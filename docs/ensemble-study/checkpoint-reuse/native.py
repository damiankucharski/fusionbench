"""Checkpoint-reuse study: fit-only native OKAPI evolution, followed by audit replay.

Use OKAPI 1d7ad2d1660627585329bc584a9c4634915e8bb1 on PYTHONPATH.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from fusionbench.eval.native_switch import serialize
from fusionbench.eval.population_trajectory import Trace, full_brier
from fusionbench.eval.suite_point_gp import _bind
from fusionbench.methods.okapi import OKAPIMethod, _task_postprocessing

GENERATIONS = 16
POPULATION = 8


def _bank(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    if x.ndim != 3 or x.shape[1:] != (POPULATION, 2) or not len(x):
        raise ValueError("Expected a nonempty (N, 8, 2) binary provider bank")
    if x.dtype.kind not in "iuf" or not np.all((x >= 0) & (x <= 1)):
        raise ValueError("Expected finite class probabilities in [0, 1]")
    if not np.allclose(x.sum(axis=2), 1., rtol=0., atol=1e-6):
        raise ValueError("Each provider's two class probabilities must sum to one")
    if y.shape != (len(x),) or y.dtype.kind not in "iub" or not np.all((y == 0) | (y == 1)):
        raise ValueError("Expected one integer 0/1 label per row")
    return x.astype(np.float64, copy=False), y.astype(np.int64, copy=False)


def _f1(positive, labels):
    predicted = positive > .5
    denominator = np.count_nonzero(predicted) + np.count_nonzero(labels)
    return 2 * np.count_nonzero(predicted & (labels == 1)) / denominator if denominator else 0.


def _fit(fit_x, fit_y, seed, objective, *, extra_founder=None, after_survival=None):
    """Search using only the fit bank and freeze each generation's winner."""
    from okapi import Okapi

    tape = Trace()
    fit_metrics = {}
    call_metrics = []

    def native_score(prediction, labels):
        positive = prediction[:, 0]
        loss, f1 = full_brier(positive, labels), _f1(positive, labels)
        call_metrics.append((loss, f1))
        return 1. - .5 * loss if objective == "brier" else f1

    class Observed(Okapi):
        def _mutate_additional_population(self):
            count = super()._mutate_additional_population()
            tape.proposals = tape.ids(self.additional_population)
            return count

        def _calculate_fitnesses(self, trees=None):
            batch = self.population if trees is None else trees
            call_metrics.clear()
            scores = super()._calculate_fitnesses(batch)
            ids = tape.ids(batch)
            tape.scored.extend(ids)
            for i, metrics in zip(ids, call_metrics, strict=True):
                old = fit_metrics.setdefault(i, metrics)
                if old != metrics:
                    raise RuntimeError("A native program's unrounded fit metrics changed between visits")
            return scores

    with _task_postprocessing("binary"):
        adapter = OKAPIMethod(seed=seed, fitness="brier")
        try:
            directory = adapter._write_split(fit_x, "fit")
            label_path = Path(adapter._tmp.name) / "labels.npy"
            np.save(label_path, fit_y)
            gp = Observed(
                directory, label_path, POPULATION, 2, 2, backend="numpy", seed=seed,
                objective_functions=(native_score,), minimize_node_count=True,
                max_nodes=31, max_depth=8, operator_point_mutation=False,
                reuse_survivor_fitness=False, selection_fitness_decimals=3,
            )
            # The default initializer supplies all eight singleton providers.
            # Scoring here gives generation zero its own snapshot without an RNG draw.
            if extra_founder is not None:
                gp.population.append(extra_founder(gp.train_tensors))
            gp.fitnesses = gp._selection_fitnesses(gp.population)
            if after_survival is not None:
                after_survival(gp, 0)
            tape.snapshot(gp.population)
            for generation in range(1, GENERATIONS + 1):
                gp.run_iteration()
                if after_survival is not None:
                    after_survival(gp, generation)
                tape.snapshot(gp.population)
        finally:
            adapter._tmp.cleanup()

    if set(fit_metrics) != set(range(len(tape.trees))):
        raise RuntimeError("A distinct proposed or surviving program lacks a native fit score")
    if any(tape.trees[i].structural_key() != key for key, i in tape.key_to_id.items()):
        raise RuntimeError("An archived native program changed after scoring")
    def selection_key(i):
        loss, f1 = fit_metrics[i]
        return (loss if objective == "brier" else -f1, tape.trees[i].nodes_count)

    winners = [min(snap["active"], key=selection_key) for snap in tape.snapshots]
    return tape, fit_metrics, winners


def _digest(data):
    packed = json.dumps(data, separators=(",", ":"), sort_keys=True, allow_nan=False).encode()
    return hashlib.sha256(packed).hexdigest()


def discover_and_score(fit_x, fit_y, audit_x, audit_y, *, seed, objective="brier"):
    """Return (scalar metrics, detail) for one binary bank under this fixed protocol.

    Both banks are (N,8,2) and labels are integer vectors. The deployed program
    minimizes unrounded fit Brier or maximizes unrounded pooled positive-class
    F1 among the final survivors. The same objective drives every generation.
    Ties favor fewer nodes and then the earlier survivor slot. Audit scoring
    starts after all sixteen generation winners plus initialization freeze.
    Brier sums across both classes. Audit extrema are hindsight diagnostics.
    The detail dict contains the two selected positive-class vectors and a
    list of per-generation diagnostic rows, starting at generation zero.
    """
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be an integer in [0, 2**32-1]")
    if objective not in ("brier", "f1"):
        raise ValueError("objective must be brier or f1")
    fit_x, fit_y = _bank(fit_x, fit_y)
    audit_x, audit_y = _bank(audit_x, audit_y)
    tape, fit_metrics, winners = _fit(fit_x, fit_y, int(seed), objective)
    selected = winners[-1]
    audit_losses = []
    audit_f1 = []
    selected_audit = None
    audit_inputs = _bind(audit_x)
    with _task_postprocessing("binary"):
        fit_prediction = np.asarray(tape.trees[selected].do_pred_on_another_tensors(current_tensors=_bind(fit_x)))[:, 0].copy()
        if (full_brier(fit_prediction, fit_y), _f1(fit_prediction, fit_y)) != fit_metrics[selected]:
            raise RuntimeError("Selected program's fit replay differs from its native score")
        for i, tree in enumerate(tape.trees):
            p = np.asarray(tree.do_pred_on_another_tensors(current_tensors=audit_inputs))[:, 0]
            if p.shape != audit_y.shape or not np.all((p >= 0) & (p <= 1)):
                raise RuntimeError("Native replay must give a finite positive-class probability per audit row")
            audit_losses.append(full_brier(p, audit_y))
            audit_f1.append(_f1(p, audit_y))
            if i == selected:
                selected_audit = p.copy()

    stages = []
    seen = set()
    scored = set()
    previous_calls = 0
    for generation, (snap, winner) in enumerate(zip(tape.snapshots, winners, strict=True)):
        active = snap["active"]
        seen.update(active if generation == 0 else snap["offered"])
        visited = tape.scored[previous_calls:snap["calls"]]
        scored.update(visited)
        if scored != seen or not set(active).issubset(seen):
            raise RuntimeError("Survivor or evaluated program missing from the proposal history")
        stages.append({
            "generation": generation, "proposals": len(snap["offered"]), "score_calls": len(visited),
            "cumulative_score_calls": snap["calls"], "n_seen": len(seen), "n_survivors": len(active),
            "selected_id": winner, "selected_nodes": tape.trees[winner].nodes_count,
            "fit_selected": fit_metrics[winner][0], "audit_selected": audit_losses[winner],
            "fit_f1": fit_metrics[winner][1], "audit_f1": audit_f1[winner],
            "audit_best_final": min(audit_losses[i] for i in active),
            "audit_best_seen": min(audit_losses[i] for i in seen),
            "audit_best_final_f1": max(audit_f1[i] for i in active),
            "audit_best_seen_f1": max(audit_f1[i] for i in seen),
        })
        previous_calls = snap["calls"]
    if len(seen) != len(tape.trees) or selected_audit is None:
        raise RuntimeError("Incomplete native trajectory")
    programs = [serialize(t.root) for t in tape.trees]
    end = stages[-1]
    metrics = {
        **{key: end[key] for key in ("fit_selected", "audit_selected", "audit_best_final", "audit_best_seen",
                                     "fit_f1", "audit_f1", "audit_best_final_f1", "audit_best_seen_f1",
                                     "n_seen", "n_survivors", "selected_nodes")},
        "objective": objective,
        "n_generations": GENERATIONS, "score_calls": len(tape.scored), "proposals": sum(s["proposals"] for s in stages),
        "selected_provider_count": len({leaf.id for leaf in tape.trees[selected].value_nodes}),
        "selected_program_sha256": _digest(programs[selected]),
        "initial_population_sha256": _digest([programs[i] for i in tape.snapshots[0]["active"]]),
        "trajectory_sha256": _digest([programs, tape.scored, tape.snapshots]),
    }
    return metrics, {"fit_selected": fit_prediction, "audit_selected": selected_audit, "trajectory": stages}

"""Native seeded-founder experiment with fit-only evolution and frozen audit replay."""

from __future__ import annotations

import re
from numbers import Integral
from pathlib import Path
from runpy import run_path

import numpy as np

from fusionbench.eval.native_switch import serialize
from fusionbench.eval.population_trajectory import full_brier
from fusionbench.eval.suite_point_gp import _bind
from fusionbench.methods.okapi import _task_postprocessing

_previous = run_path(
    str(Path(__file__).resolve().parents[1] / "checkpoint-reuse" / "native.py")
)
_fit, _bank, _f1, _digest = (_previous[k] for k in ("_fit", "_bank", "_f1", "_digest"))
ARMS = ("stock", "sham", "seeded", "carry")
TOL = 1e-12


def _index(value, size, name):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Integral)
        or not 0 <= value < size
    ):
        raise ValueError(f"{name} must be an integer in [0, {size})")
    return int(value)


def make_founder(tensors, arm, sham_provider=0):
    """Return a reserved native founder, or None for stock singleton initialization.

    Native provider IDs m0.npy, m1.npy, ... use numeric order. Other tensor
    mappings retain their iteration order. Binary and multiclass share the same
    tree construction. Supply the native postprocessor for the task on replay.
    """
    from okapi.node import MeanNode, ValueNode
    from okapi.tree import Tree

    if arm not in ARMS:
        raise ValueError(f"Unknown arm {arm!r}; expected one of {ARMS}")
    if arm == "stock":
        return None
    items = list(tensors.items())
    if not items:
        raise ValueError("A founder requires at least one provider")
    if all(isinstance(k, str) and re.fullmatch(r"m[0-9]+\.npy", k) for k, _ in items):
        items.sort(key=lambda item: int(item[0][1:-4]))
    if arm == "sham":
        i = _index(sham_provider, len(items), "sham_provider")
        items = [items[i]] * len(items)
    root_id, root_value = items[0]
    operands = [ValueNode(None, value, name) for name, value in items[1:]]
    return Tree(ValueNode([MeanNode(operands)], root_value, root_id))


def _copy_tree(tree):
    fresh = tree.copy()
    fresh.mutation_chance = tree.mutation_chance
    return fresh


def restore_founder(gp, reserved_tree, reserved_fitness):
    """Restore an absent founder in the last survivor slot using its saved fitness row."""
    if not len(gp.population) or len(gp.population) != len(gp.fitnesses):
        raise ValueError(
            "Restoration requires a scored, nonempty, aligned survivor population"
        )
    if np.shape(reserved_fitness) != gp.fitnesses[-1].shape:
        raise ValueError("Supply the complete original native selection fitness row")
    key = reserved_tree.structural_key()
    if any(tree.structural_key() == key for tree in gp.population):
        return False
    replacement = _copy_tree(reserved_tree)
    gp.fitnesses[-1] = reserved_fitness
    gp.population[-1] = replacement
    return True


def _audit_probability(audit_q, rows):
    if audit_q is None:
        return None
    q = np.asarray(audit_q)
    if q.dtype.kind not in "iuf" or not np.all((q >= 0.0) & (q <= 1.0)):
        raise ValueError("audit_q must contain finite probabilities in [0, 1]")
    if q.shape == (rows, 2):
        if not np.allclose(q.sum(axis=1), 1.0, rtol=0.0, atol=1e-6):
            raise ValueError("Two-column audit_q must sum to one for every row")
        q = q[:, 1]
    if q.shape == (rows, 1):
        q = q[:, 0]
    if q.shape != (rows,):
        raise ValueError("Expected one ground-truth positive probability per audit row")
    return q.astype(float, copy=False)


def _matches(pair, reference):
    return all(
        np.all(np.abs(p - ref) <= TOL) for p, ref in zip(pair, reference, strict=True)
    )


def _milestones(stages, field):
    present = [s["n_active_" + field] for s in stages]
    if present[0] is None:
        return None, None, None
    first = next((i for i, n in enumerate(present) if n), None)
    lost = next(
        (i for i in range(1, len(present)) if present[i - 1] and not present[i]), None
    )
    again = (
        next((i for i in range(lost + 1, len(present)) if present[i]), None)
        if lost is not None
        else None
    )
    return first, lost, again


def run_seeded(
    fit_x,
    fit_y,
    audit_x,
    audit_y,
    *,
    seed,
    arm,
    sham_provider=0,
    audit_q=None,
    signal_provider=None,
):
    """Return scalar metrics and positive-class predictions plus 17 trajectory records.

    The four arms differ only in founders and, for carry, in survivor retention.
    Each generation's deployment winner minimizes unrounded fit full-class Brier;
    ties favor fewer nodes and then the earliest surviving slot. A second terminal
    selection compares that winner to a fresh native mean, preferring the mean on
    exact ties. This finalist incurs one extra fit score outside the search.

    Optional audit_q and signal_provider are used only for retrospective diagnostics.
    Exact identity uses the canonical ordered genotype. Equivalence requires every
    fit and audit probability to agree to absolute tolerance 1e-12. Offered counts
    include repeated proposals. Seen counts are distinct genotypes since generation 0.
    """
    from okapi.node import ValueNode
    from okapi.tree import Tree

    if arm not in ARMS:
        raise ValueError(f"Unknown arm {arm!r}; expected one of {ARMS}")
    seed = _index(seed, 2**32, "seed")
    fit_x, fit_y = _bank(fit_x, fit_y)
    audit_x, audit_y = _bank(audit_x, audit_y)
    audit_q = _audit_probability(audit_q, len(audit_y))
    m = fit_x.shape[1]
    if signal_provider is not None:
        signal_provider = _index(signal_provider, m, "signal_provider")
    sham_provider = _index(sham_provider, m, "sham_provider")
    reserved = None
    reinserted = []

    def after_survival(gp, generation):
        nonlocal reserved
        if generation == 0:
            reserved = _copy_tree(gp.population[-1]), gp.fitnesses[-1].copy()
            reinserted.append(False)
        else:
            reinserted.append(restore_founder(gp, *reserved))

    def extra_founder(tensors):
        return make_founder(tensors, arm, sham_provider)

    tape, fit_metrics, winners = _fit(
        fit_x,
        fit_y,
        seed,
        "brier",
        extra_founder=None if arm == "stock" else extra_founder,
        after_survival=after_survival if arm == "carry" else None,
    )
    end_id = winners[-1]
    fit_inputs = _bind(fit_x)
    mean = make_founder(fit_inputs, "seeded")
    with _task_postprocessing("binary"):
        mean_fit = np.asarray(mean.predict())[:, 0].copy()
    mean_fit_brier = full_brier(mean_fit, fit_y)
    use_mean = mean_fit_brier <= fit_metrics[end_id][0]

    # All search states and both terminal decisions are frozen before audit replay.
    audit_inputs = _bind(audit_x)
    joint_inputs = {
        key: np.concatenate((value, audit_inputs[key]))
        for key, value in fit_inputs.items()
    }
    mean_key = mean.structural_key()
    signal_key = signal_pair = None
    if signal_provider is not None:
        key = f"m{signal_provider}.npy"
        signal_key = Tree(ValueNode(None, fit_inputs[key], key)).structural_key()
        signal_pair = (fit_x[:, signal_provider, 1], audit_x[:, signal_provider, 1])
    fields = ("mean_exact", "mean_equiv", "signal_exact", "signal_equiv")
    records = []
    selected_pair = None
    with _task_postprocessing("binary"):
        mean_audit = np.asarray(
            mean.do_pred_on_another_tensors(current_tensors=audit_inputs)
        )[:, 0].copy()
        mean_pair = mean_fit, mean_audit
        for i, tree in enumerate(tape.trees):
            positive = np.asarray(
                tree.do_pred_on_another_tensors(current_tensors=joint_inputs)
            )[:, 0]
            if positive.shape != (len(fit_y) + len(audit_y),) or not np.all(
                (positive >= 0) & (positive <= 1)
            ):
                raise RuntimeError(
                    "Expected one finite native positive probability per replay row"
                )
            fit, audit = positive[: len(fit_y)], positive[len(fit_y) :]
            if (full_brier(fit, fit_y), _f1(fit, fit_y)) != fit_metrics[i]:
                raise RuntimeError(
                    "Program fit replay differs from the original native score"
                )
            key = tree.structural_key()
            records.append(
                {
                    "audit_brier": full_brier(audit, audit_y),
                    "audit_f1": _f1(audit, audit_y),
                    "excess": full_brier(audit, audit_q)
                    if audit_q is not None
                    else None,
                    "mean_exact": key == mean_key,
                    "mean_equiv": _matches((fit, audit), mean_pair),
                    "signal_exact": key == signal_key
                    if signal_key is not None
                    else None,
                    "signal_equiv": _matches((fit, audit), signal_pair)
                    if signal_pair is not None
                    else None,
                    "provider_count": len(tree.unique_value_node_ids),
                }
            )
            if i == end_id:
                selected_pair = fit.copy(), audit.copy()

    stages = []
    seen, scored = set(), set()
    previous_calls = 0
    for generation, (snap, winner) in enumerate(
        zip(tape.snapshots, winners, strict=True)
    ):
        active, offered = snap["active"], snap["offered"]
        seen.update(active if generation == 0 else offered)
        visited = tape.scored[previous_calls : snap["calls"]]
        scored.update(visited)
        if scored != seen or not set(active).issubset(seen):
            raise RuntimeError(
                "Survivor or scored program is missing from the frozen proposal history"
            )
        w = records[winner]
        row = {
            "generation": generation,
            "proposals": len(offered),
            "score_calls": len(visited),
            "cumulative_score_calls": snap["calls"],
            "n_seen": len(seen),
            "n_survivors": len(active),
            "selected_id": winner,
            "selected_nodes": tape.trees[winner].nodes_count,
            "selected_provider_count": w["provider_count"],
            "fit_selected": fit_metrics[winner][0],
            "audit_selected": w["audit_brier"],
            "fit_f1": fit_metrics[winner][1],
            "audit_f1": w["audit_f1"],
            "audit_selected_excess": w["excess"],
            "carry_reinserted": reinserted[generation] if arm == "carry" else False,
        }
        for suffix, ids in (("final", active), ("seen", seen)):
            row[f"audit_best_{suffix}"] = min(records[i]["audit_brier"] for i in ids)
            row[f"audit_best_{suffix}_f1"] = max(records[i]["audit_f1"] for i in ids)
            row[f"audit_best_{suffix}_excess"] = (
                min(records[i]["excess"] for i in ids) if audit_q is not None else None
            )
        for field in fields:
            row["selected_" + field] = w[field]
            for group, ids in (
                ("active", active),
                ("offered", offered),
                ("seen", seen),
            ):
                row[f"n_{group}_{field}"] = (
                    sum(records[i][field] for i in ids)
                    if w[field] is not None
                    else None
                )
        if arm == "carry" and row["n_active_mean_exact"] == 0:
            raise RuntimeError("Carry failed to preserve the canonical founder")
        stages.append(row)
        previous_calls = snap["calls"]
    if len(seen) != len(tape.trees) or selected_pair is None:
        raise RuntimeError("Incomplete native trajectory")
    if stages[0]["n_survivors"] != m + (arm != "stock"):
        raise RuntimeError("Incorrect founding population size")
    singletons = [tape.trees[i] for i in tape.snapshots[0]["active"][:m]]
    if any(t.nodes_count != 1 for t in singletons) or {
        t.root.id for t in singletons
    } != set(fit_inputs):
        raise RuntimeError(
            "A native provider singleton is missing from the founding population"
        )
    if arm in ("seeded", "carry") and not stages[0]["n_active_mean_exact"]:
        raise RuntimeError("Missing canonical mean founder")
    programs = [serialize(tree.root) for tree in tape.trees]
    end = stages[-1]
    scalar = {
        **{
            k: v
            for k, v in end.items()
            if k not in ("generation", "cumulative_score_calls", "carry_reinserted")
        },
        "objective": "brier",
        "arm": arm,
        "n_generations": len(stages) - 1,
        "score_calls": len(tape.scored),
        "proposals": sum(s["proposals"] for s in stages),
        "selected_program_sha256": _digest(programs[end_id]),
        "initial_population_sha256": _digest(
            [programs[i] for i in tape.snapshots[0]["active"]]
        ),
        "trajectory_sha256": _digest([programs, tape.scored, tape.snapshots]),
        "carry_reinsertions": sum(reinserted),
        "protected_used_mean": bool(use_mean),
        "protected_chose_mean": bool(use_mean),
        "protected_extra_fit_score_calls": 1,
        "protected_selected_provider_count": m
        if use_mean
        else end["selected_provider_count"],
        "protected_selected_nodes": mean.nodes_count
        if use_mean
        else end["selected_nodes"],
        "protected_selected_program_sha256": _digest(
            serialize(mean.root) if use_mean else programs[end_id]
        ),
    }
    for field in fields:
        for label, value in zip(
            ("first_present", "first_lost", "first_reappeared"),
            _milestones(stages, field),
            strict=True,
        ):
            scalar[f"{field}_{label}"] = value
    protected_pair = mean_pair if use_mean else selected_pair
    scalar.update(
        {
            "protected_fit_selected": full_brier(protected_pair[0], fit_y),
            "protected_audit_selected": full_brier(protected_pair[1], audit_y),
            "protected_fit_f1": _f1(protected_pair[0], fit_y),
            "protected_audit_f1": _f1(protected_pair[1], audit_y),
            "protected_audit_selected_excess": full_brier(protected_pair[1], audit_q)
            if audit_q is not None
            else None,
            "protected_selected_mean_exact": use_mean or end["selected_mean_exact"],
            "protected_selected_mean_equiv": use_mean or end["selected_mean_equiv"],
            "protected_selected_signal_exact": (
                False if use_mean else end["selected_signal_exact"]
            )
            if signal_pair is not None
            else None,
            "protected_selected_signal_equiv": _matches(protected_pair, signal_pair)
            if signal_pair is not None
            else None,
        }
    )
    return scalar, {
        "fit_selected": selected_pair[0],
        "audit_selected": selected_pair[1],
        "protected_fit_selected": protected_pair[0],
        "protected_audit_selected": protected_pair[1],
        "trajectory": stages,
    }

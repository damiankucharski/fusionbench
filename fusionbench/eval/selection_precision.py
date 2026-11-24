"""Vary only the precision of native tournament and survivor fitness on fixed predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from itertools import product
from numbers import Integral
from pathlib import Path
from time import perf_counter

import numpy as np

from fusionbench.eval.full_suite_point import data_for, fingerprint
from fusionbench.eval.metrics import score
from fusionbench.eval.mutation_opportunity import evaluate_bank as evaluate_baselines
from fusionbench.eval.native_switch import serialize
from fusionbench.eval.suite_point_gp import (
    SuiteCheckpoint,
    _bind,
    _full,
    _loss,
    _targets,
    predict_tree,
)
from fusionbench.methods.okapi import OKAPIMethod, _task_postprocessing

POLICIES = {"round3": 3, "round6": 6, "raw": None}
DATA_SEEDS = range(5900, 5906)
GP_SEEDS = (709, 719)


@dataclass(frozen=True)
class PrecisionCheckpoint(SuiteCheckpoint):
    counts: dict[str, int]
    cohorts: tuple[dict, ...] = ()


def resolution(scores):
    """Count distinct native float64 fitness values at each decision precision."""
    f = np.asarray(scores, dtype=np.float64)
    if f.ndim != 1 or not f.size or not np.isfinite(f).all():
        raise ValueError("Expected a nonempty finite vector of native fitness scores")
    return {
        "distinct_raw": len(np.unique(f)),
        "distinct_6": len(np.unique(f.round(6))),
        "distinct_3": len(np.unique(f.round(3))),
        "distinct_compressed_3": len(np.unique((1 - (1 - f) * 2**-12).round(3))),
    }


def _tied_pairs(f):
    _, counts = np.unique(f, return_counts=True)
    return int(np.sum(counts * (counts - 1) // 2))


def _audit_survivors(engine, cohort, raw, state, row):
    """Replay just survivor selection, verifying membership and order of the native result."""
    from okapi.population import choose_pareto, choose_pareto_then_proximity
    from okapi.random_state import use_rng

    lookup = {id(tree): i for i, tree in enumerate(cohort)}
    actual = [lookup[id(t)] for t in engine.population]
    sizes = np.asarray([t.nodes_count for t in cohort], dtype=int)
    row["candidate_fitness"] = json.dumps(raw[:, 0].tolist())
    row["candidate_nodes"] = json.dumps(sizes.tolist())
    for name, places in POLICIES.items():
        scores = raw if places is None else raw.round(places)
        clone = np.random.RandomState(0)
        clone.set_state(state)
        with use_rng(clone):
            front, _ = choose_pareto(
                cohort,
                scores,
                len(cohort),
                engine.objectives,
                engine.minimize_node_count,
            )
            selected, _ = choose_pareto_then_proximity(
                cohort,
                scores,
                engine.objectives,
                engine.population_size,
                engine.minimize_node_count,
            )
        indices = [lookup[id(t)] for t in selected]
        if places == engine.selection_fitness_decimals and indices != actual:
            raise RuntimeError(
                "Same-precision replay disagrees with the actual native survivors"
            )
        row[f"{name}_front_size"] = len(front)
        row[f"{name}_mean_nodes"] = float(sizes[indices].mean())
        row[f"{name}_mean_raw_fitness"] = float(raw[indices, 0].mean())
        row[f"{name}_max_raw_fitness"] = float(raw[indices, 0].max())
        row[f"{name}_membership_replacements"] = len(set(actual) - set(indices))
        row[f"{name}_positions_changed"] = sum(
            a != b for a, b in zip(actual, indices, strict=True)
        )
        row[f"{name}_indices"] = json.dumps(indices)
    row["native_indices"] = json.dumps(actual)
    row["survivor_capacity"] = len(actual)


def fit_selection_precision(
    preds,
    fit_y,
    validation_preds,
    validation_y,
    *,
    seed,
    decimals,
    budgets=(600, 1200),
    audit_survival=False,
):
    """One native run per precision, with exact objective-call prefix archives.

    Cohort summaries include the initial complete scoring batch and the subsequent
    complete deduplicated batches. An unfinished batch is absent from a checkpoint.
    The optional same-cohort control compares native post-scoring survivors only.
    Its scores, RNG, survivor lists and choices never enter subsequent variation.
    """
    from okapi import Okapi
    from okapi.operators import (
        CLOSE_THRESHOLD,
        FAR_THRESHOLD,
        MAX,
        MEAN,
        MIN,
        WEIGHTED_MEAN,
    )

    budgets = tuple(budgets)
    if not budgets or any(
        isinstance(b, bool) or not isinstance(b, Integral) or b < 1 for b in budgets
    ):
        raise ValueError("Budgets must be positive integers")
    budgets = {int(b) for b in budgets}
    maximum = max(budgets)
    fit, val = _full(preds), _full(validation_preds)
    n, m, c = fit.shape
    if m < 2 or fit.shape[1:] != val.shape[1:]:
        raise ValueError(
            "Expected identical class/provider counts and at least two providers"
        )
    labels, target = _targets(fit_y, n, c)
    _, val_target = _targets(validation_y, len(val), c)
    bindings = _bind(val)
    seen = set()
    checks, cohorts = {}, []
    counts = {
        "fit_calls": 0,
        "validation_calls": 0,
        "complete_batches": 0,
        "complete_scored_rows": 0,
    }
    digest = hashlib.sha256()
    best_fit = best_validation = None
    losses = []
    generation = 0
    shadow_seconds = 0.0

    def objective(p, _):
        loss = _loss(p, target)
        losses.append(loss)
        return 1 - loss

    def remember(tree, loss):
        nonlocal best_fit, best_validation
        counts["fit_calls"] += 1
        key = tree.structural_key()
        digest.update((repr(key) + "\n").encode())
        digest.update(struct.pack("<d", loss))
        if key in seen:
            return
        seen.add(key)
        vp = tree.do_pred_on_another_tensors(current_tensors=bindings)
        vl = _loss(vp, val_target)
        counts["validation_calls"] += 1
        win_fit = best_fit is None or loss < best_fit[0]
        win_val = best_validation is None or (vl, loss) < best_validation[:2]
        if win_fit or win_val:
            frozen = tree.copy()
            frozen.mutation_chance = tree.mutation_chance
            if win_fit:
                best_fit = (loss, frozen)
            if win_val:
                best_validation = (vl, loss, frozen)

    def capture(calls):
        checks[calls] = PrecisionCheckpoint(
            fit_tree=best_fit[-1],
            val_tree=best_validation[-1],
            calls=calls,
            n_trees=len(seen),
            generation=generation,
            path_digest=digest.hexdigest(),
            elapsed_seconds=perf_counter() - started - shadow_seconds,
            fit_brier=best_fit[0],
            validation_brier=best_validation[0],
            counts=dict(counts),
        )

    class StopScoring(Exception):
        pass

    class Observed(Okapi):
        pending_audit = None

        def _calculate_fitnesses(self, trees=None):
            cohort = self.population if trees is None else trees
            initial = self.fitnesses is None
            start = counts["fit_calls"]
            end = min(start + len(cohort), maximum)
            # Cut each budget before running its following scores.
            edges = sorted({b for b in budgets if start < b <= end} | {end})
            chunks = []
            for edge in edges:
                portion = cohort[counts["fit_calls"] - start : edge - start]
                losses.clear()
                raw = super()._calculate_fitnesses(portion)
                chunks.append(raw)
                for t, loss in zip(portion, losses, strict=True):
                    remember(t, loss)
                if edge == start + len(cohort):
                    scores = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
                    f = scores[:, 0]
                    row = {
                        "kind": "initial" if initial else "survival",
                        "generation": generation,
                        "n": len(f),
                        "call_end": edge,
                        **resolution(f),
                        "tied_pairs_raw": _tied_pairs(f),
                        "tied_pairs_6": _tied_pairs(f.round(6)),
                        "tied_pairs_3": _tied_pairs(f.round(3)),
                    }
                    cohorts.append(row)
                    counts["complete_batches"] += 1
                    counts["complete_scored_rows"] += len(f)
                    if audit_survival and not initial:
                        self.pending_audit = (
                            list(cohort),
                            scores.copy(),
                            self.rng.get_state(),
                            row,
                        )
                if edge in budgets:
                    capture(edge)
                if edge == maximum and (initial or edge < start + len(cohort)):
                    raise StopScoring
            return np.concatenate(chunks) if len(chunks) > 1 else chunks[0]

    with _task_postprocessing("binary" if c == 2 else "multiclass"):
        started = perf_counter()
        adapter = OKAPIMethod(seed=seed, fitness="brier")
        try:
            directory = adapter._write_split(fit, "fit")
            gt = Path(adapter._tmp.name) / "gt.npy"
            np.save(gt, labels)
            pop = max(2, min(adapter.population_size, m))
            gp = Observed(
                directory,
                gt,
                pop,
                adapter.population_multiplier,
                max(2, min(adapter.tournament_size, pop)),
                backend="numpy",
                seed=seed,
                objective_functions=(objective,),
                minimize_node_count=adapter.minimize_node_count,
                postprocessing_function=None,
                allowed_ops=(
                    MEAN,
                    MIN,
                    MAX,
                    WEIGHTED_MEAN,
                    FAR_THRESHOLD,
                    CLOSE_THRESHOLD,
                ),
                operator_point_mutation=False,
                reuse_survivor_fitness=False,
                selection_fitness_decimals=decimals,
            )
            adapter._seed_population(gp, np.random.default_rng(seed))
            try:
                while counts["fit_calls"] < maximum:
                    generation += 1
                    gp.run_iteration()
                    if gp.pending_audit is not None:
                        audit_start = perf_counter()
                        _audit_survivors(gp, *gp.pending_audit)
                        shadow_seconds += perf_counter() - audit_start
                        gp.pending_audit = None
            except StopScoring:
                pass
        finally:
            adapter._tmp.cleanup()
    return {
        calls: replace(
            cp, cohorts=tuple(row for row in cohorts if row["call_end"] <= calls)
        )
        for calls, cp in checks.items()
    }


def panel(pair):
    from loguru import logger

    logger.disable("okapi")
    name, data_seed = pair
    scenario, splits = data_for(name, data_seed)
    classes = scenario.n_classes
    x, y, _ = splits["fit"]
    vx, vy, _ = splits["validation"]
    # The archived four-arm driver exposes the same baseline protocol on fixed
    # arrays. Passing no search arms fits only the three predefined baselines.
    rows = evaluate_baselines(name, data_seed, splits, arms=())
    cohorts = []
    for gp_seed, (policy, decimals) in product(GP_SEEDS, POLICIES.items()):
        audit = policy == "round3" and (data_seed, gp_seed) == (5900, 709)
        cp_by_budget = fit_selection_precision(
            x, y, vx, vy, seed=gp_seed, decimals=decimals, audit_survival=audit
        )
        run = {
            "scenario": name,
            "data_seed": data_seed,
            "gp_seed": gp_seed,
            "method": policy,
        }
        cohorts.extend(
            {**run, **row} for row in cp_by_budget[max(cp_by_budget)].cohorts
        )
        for calls, cp in cp_by_budget.items():
            for decision, tree in (
                ("fit", cp.fit_tree),
                ("validation", cp.validation_tree),
            ):
                metadata = {
                    **run,
                    "calls": calls,
                    "checkpoint": str(calls),
                    "generation": cp.generation,
                    "selection": decision,
                    "classes": classes,
                    "providers": x.shape[1],
                    "nodes": tree.nodes_count,
                    "archive_trees": cp.n_trees,
                    "search_seconds": cp.elapsed_seconds,
                    "path_digest": cp.path_digest,
                    "program": json.dumps(serialize(tree.root)),
                    "archive_selection_brier": cp.fit_brier
                    if decision == "fit"
                    else cp.validation_brier,
                    **cp.counts,
                }
                for split, (z, labels, oracle) in splits.items():
                    p = predict_tree(tree, z)
                    if (
                        not np.isfinite(p).all()
                        or np.any(p < -1e-14)
                        or np.any(p > 1 + 1e-14)
                        or not np.allclose(p.sum(axis=1), 1, rtol=0, atol=2e-12)
                    ):
                        raise ValueError(
                            f"Invalid native probabilities for {name}/{split}"
                        )
                    rows.append(
                        {
                            **metadata,
                            "split": split,
                            "rows": len(labels),
                            "prediction_sha256": fingerprint(p),
                            "oracle_sha256": None
                            if oracle is None
                            else fingerprint(oracle),
                            "brier": np.mean(
                                np.sum((p - np.eye(classes)[labels]) ** 2, axis=1)
                            ),
                            "conditional_excess": None
                            if oracle is None
                            else np.mean(np.sum((p - oracle) ** 2, axis=1)),
                            **score(
                                labels,
                                p,
                                classes,
                                metrics=["accuracy", "macro_f1", "log_loss"],
                            ),
                        }
                    )
    return rows, cohorts


def run(out, workers):
    import pandas as pd

    from fusionbench.scenarios.registry import STANDARD_V1_SCENARIOS as SCENARIOS

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows, cohorts = [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for results, batch_rows in pool.map(panel, product(SCENARIOS, DATA_SEEDS)):
            rows.extend(results)
            cohorts.extend(batch_rows)
    for stem, values in (("runs", rows), ("cohorts", cohorts)):
        pending = out / f"{stem}.partial.csv.gz"
        pd.DataFrame(values).to_csv(pending, index=False)
        pending.replace(out / f"{stem}.csv.gz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--workers", default=2, type=int)
    args = parser.parse_args()
    run(args.out, args.workers)

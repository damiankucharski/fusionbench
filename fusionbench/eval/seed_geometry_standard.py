"""Compare native seed-tree shape on the fixed standard FusionBench panel."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import struct
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import product
from numbers import Integral
from pathlib import Path
from time import perf_counter

import numpy as np

from fusionbench.eval.full_suite_point import data_for, fingerprint
from fusionbench.eval.metrics import score
from fusionbench.eval.mutation_opportunity import evaluate_bank as evaluate_baselines
from fusionbench.eval.native_switch import serialize
from fusionbench.eval.seed_geometry import (
    append_geometry_seeds,
    make_seed_plans,
    replace_deep_seeds,
)
from fusionbench.eval.suite_point_gp import (
    SuiteCheckpoint,
    _bind,
    _full,
    _loss,
    _targets,
    predict_tree,
)
from fusionbench.methods.okapi import OKAPIMethod, _task_postprocessing

SHAPES = ("stock", "branch", "chain")
DATA_SEEDS = range(7000, 7006)
GP_SEEDS = (907, 919)


@dataclass(frozen=True)
class GeometryCheckpoint(SuiteCheckpoint):
    diversity: dict
    initial_rows: tuple[dict, ...]
    validation_calls: int


def validate_gp_validation_top_k(value):
    if value is not None and (
        isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1
    ):
        raise ValueError("gp_validation_top_k must be None or a positive integer")
    return None if value is None else int(value)


def _diversity(rows):
    extra = [r for r in rows if r["origin"] == "seed"]
    predictions = tuple(r["fit_prediction_sha256"] for r in rows)
    return {
        "initial_trees": len(rows),
        "initial_singletons": len(rows) - len(extra),
        "initial_seed_trees": len(extra),
        "initial_unique_genotypes": len({r["genotype_sha256"] for r in rows}),
        "initial_unique_fit_predictions": len(set(predictions)),
        "seed_unique_fit_predictions": len({r["fit_prediction_sha256"] for r in extra}),
        "initial_prediction_hashes": predictions,
    }


def fit_seed_geometry(
    preds, fit_y, validation_preds, validation_y, *, seed, shape, budgets=(600, 1200),
    fitness=None, on_candidate=None, discovery_fitness=None, gp_validation_top_k=None,
):
    """Return prefix champions and the complete native pre-evolution initial archive.

    Training diversity counts distinct byte-exact native float64 prediction matrices
    among the specified actually scored trees. Every fit forward is reused from the
    native score callback. By default each structural program receives one
    validation pass. gp_validation_top_k restricts validation at each checkpoint
    to its k best unrounded discovery scores, keeping first visits on ties.
    The default ranks by lowest raw discovery Brier. A custom fitness or
    discovery_fitness ranks by its supplied higher-is-better discovery score.
    Shared candidates across checkpoints reuse their cached validation score.
    on_candidate(tree, fit_prediction) observes each distinct program's first
    fit evaluation, in traversal order. Copy the tree to retain it. A supplied
    fitness(native_prediction, labels) controls evolution and both champions.
    OKAPI requires unit-interval selection scores, so custom scores enter
    native selection as tied ranks within the fully reevaluated population.
    Archive comparisons use their original finite values. The reported Brier
    fields always describe the selected trees' Brier losses. An optional
    discovery_fitness replaces only the stock GP training score on its usual
    unit scale with three-decimal tournament selection. Validation champions
    still maximize raw held-out Brier skill with raw fit Brier as a tiebreaker.
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

    top_k = validate_gp_validation_top_k(gp_validation_top_k)
    if fitness is not None and discovery_fitness is not None:
        raise ValueError("Specify either a common fitness or a discovery-only fitness")
    if shape not in (*SHAPES, "mixed_branch", "mixed_chain"):
        raise ValueError(f"Unknown shape {shape!r}")
    budgets = tuple(budgets)
    if not budgets or any(
        isinstance(b, bool) or not isinstance(b, Integral) or b < 1 for b in budgets
    ):
        raise ValueError("Budgets must be positive integers")
    cuts = {int(b) for b in budgets}
    x, v = _full(preds), _full(validation_preds)
    n, m, c = x.shape
    if m < 2 or x.shape[1:] != v.shape[1:]:
        raise ValueError(
            "Expected matching class/provider counts and at least two providers"
        )
    labels, target = _targets(fit_y, n, c)
    val_labels, vt = _targets(validation_y, len(v), c)
    inputs = _bind(v)
    seen = set()
    fit_vectors = set()
    digest = hashlib.sha256()
    visits = 0
    generation = 0
    best_fit = best_val = None
    shortlist = []
    validation_cache = {}
    checks = {}
    initial_rows = []
    batch_losses = []
    batch_hashes = []
    batch_predictions = []
    collecting_init = True

    def objective(p, _):
        val = _loss(p, target)
        batch_losses.append(val)
        batch_hashes.append(fingerprint(p))
        if on_candidate is not None:
            batch_predictions.append(p)
        if discovery_fitness is not None:
            return discovery_fitness(p, labels)
        return 1 - val if fitness is None else custom_score(p, labels)

    def custom_score(p, y):
        value = np.asarray(fitness(p, y))
        if value.ndim != 0 or not np.isfinite(value):
            raise ValueError("fitness must return a finite scalar")
        return float(value)

    def validation(t):
        vp = t.do_pred_on_another_tensors(current_tensors=inputs)
        vl = _loss(vp, vt)
        return (-vl if fitness is None else custom_score(vp, val_labels)), vl

    def remember(t, loss, fit_score, batch_index):
        nonlocal visits, best_fit, best_val
        key = t.structural_key()
        visits += 1
        digest.update((repr(key) + "\n").encode())
        digest.update(struct.pack("<d", loss))
        if key in seen:
            return
        seen.add(key)
        if on_candidate is not None:
            on_candidate(t, batch_predictions[batch_index])
        wf = best_fit is None or fit_score > best_fit[0]
        validation_tiebreak = -loss if discovery_fitness is not None else fit_score
        if top_k is not None:
            # The heap root is the least eligible pair (lowest score, latest visit).
            priority = (fit_score, -visits)
            eligible = len(shortlist) < top_k or priority > shortlist[0][:2]
            if wf or eligible:
                ref = t.copy()
                ref.mutation_chance = t.mutation_chance
                if wf:
                    best_fit = (fit_score, loss, ref)
                if eligible:
                    entry = (*priority, validation_tiebreak, ref)
                    if len(shortlist) < top_k:
                        heapq.heappush(shortlist, entry)
                    else:
                        heapq.heapreplace(shortlist, entry)
            return
        val_score, vl = validation(t)
        wv = best_val is None or (val_score, validation_tiebreak) > best_val[:2]
        if wf or wv:
            ref = t.copy()
            ref.mutation_chance = t.mutation_chance
            if wf:
                best_fit = (fit_score, loss, ref)
            if wv:
                best_val = (val_score, validation_tiebreak, vl, ref)

    def capture(label):
        nonlocal best_val
        if top_k is not None:
            best_order = None
            for _, neg_visit, tie, ref in shortlist:
                if neg_visit not in validation_cache:
                    validation_cache[neg_visit] = validation(ref)
                val_score, vl = validation_cache[neg_visit]
                order = (val_score, tie, neg_visit)
                if best_order is None or order > best_order:
                    best_order = order
                    best_val = (val_score, tie, vl, ref)
        checks[label] = GeometryCheckpoint(
            fit_tree=best_fit[-1],
            val_tree=best_val[-1],
            calls=visits,
            n_trees=len(seen),
            generation=generation,
            path_digest=digest.hexdigest(),
            elapsed_seconds=perf_counter() - start,
            fit_brier=best_fit[-2],
            validation_brier=best_val[-2],
            diversity={
                **_diversity(initial_rows),
                "archive_unique_fit_predictions": len(fit_vectors),
            },
            initial_rows=tuple(initial_rows),
            validation_calls=len(seen) if top_k is None else len(validation_cache),
        )

    class Stop(Exception):
        pass

    class Observed(Okapi):
        def _calculate_fitnesses(self, trees=None):
            nonlocal collecting_init
            trees = self.population if trees is None else trees
            start_calls = visits
            stop_at = min(start_calls + len(trees), last)
            boundaries = sorted(
                {b for b in cuts if start_calls < b < stop_at} | {stop_at}
            )
            batches = []
            for end in boundaries:
                subset = trees[visits - start_calls : end - start_calls]
                batch_losses.clear()
                batch_hashes.clear()
                batch_predictions.clear()
                batches.append(super()._calculate_fitnesses(subset))
                for j, (tree, loss) in enumerate(
                    zip(subset, batch_losses, strict=True)
                ):
                    fit_vectors.add(batch_hashes[j])
                    if collecting_init:
                        key = repr(tree.structural_key()).encode()
                        initial_rows.append(
                            {
                                "index": visits,
                                "origin": "singleton"
                                if visits < singleton_count
                                else "seed",
                                "nodes": tree.nodes_count,
                                "fit_prediction_sha256": batch_hashes[j],
                                "genotype_sha256": hashlib.sha256(key).hexdigest(),
                                "native_fit_brier": loss,
                                "program": json.dumps(
                                    serialize(tree.root), separators=(",", ":")
                                ),
                            }
                        )
                    fit_score = -loss if fitness is None and discovery_fitness is None else batches[-1][j, 0]
                    remember(tree, loss, fit_score, j)
                if end in cuts:
                    capture(end)
                if collecting_init and end == init_count:
                    capture("initial")
                    collecting_init = False
                if end == last:
                    raise Stop
            scores = batches[0] if len(batches) == 1 else np.concatenate(batches)
            if fitness is not None:
                unique, ranks = np.unique(scores[:, 0], return_inverse=True)
                scores = ranks[:, None] / max(1, len(unique) - 1)
            return scores

    with _task_postprocessing("binary" if c == 2 else "multiclass"):
        start = perf_counter()
        adapter = OKAPIMethod(seed=seed, fitness="brier")
        try:
            p = adapter._write_split(x, "fit")
            gt = Path(adapter._tmp.name) / "labels.npy"
            np.save(gt, labels)
            pop = max(2, min(m, adapter.population_size))
            gp = Observed(
                p,
                gt,
                pop,
                adapter.population_multiplier,
                max(2, min(adapter.tournament_size, pop)),
                seed=seed,
                backend="numpy",
                allowed_ops=(
                    MEAN,
                    MIN,
                    MAX,
                    WEIGHTED_MEAN,
                    FAR_THRESHOLD,
                    CLOSE_THRESHOLD,
                ),
                minimize_node_count=adapter.minimize_node_count,
                selection_fitness_decimals=3 if fitness is None else None,
                reuse_survivor_fitness=False,
                operator_point_mutation=False,
                objective_functions=(objective,),
                postprocessing_function=None,
            )
            singleton_count = len(gp.population)
            if shape in ("stock", "mixed_branch", "mixed_chain"):
                adapter._seed_population(gp, np.random.default_rng(seed))
                if shape != "stock":
                    replace_deep_seeds(gp, seed, shape, singleton_count)
            else:
                append_geometry_seeds(
                    gp, make_seed_plans(gp.ids, seed, adapter.n_seeds), shape
                )
            init_count = len(gp.population)
            last = max(max(cuts), init_count)
            try:
                while True:
                    generation += 1
                    gp.run_iteration()
            except Stop:
                pass
        finally:
            adapter._tmp.cleanup()
    return checks


def panel(pair):
    from loguru import logger

    logger.disable("okapi")
    name, bank = pair
    scenario, splits = data_for(name, bank)
    x, y, _ = splits["fit"]
    vx, vy, _ = splits["validation"]
    classes = scenario.n_classes
    rows = evaluate_baselines(name, bank, splits, arms=())
    seeds = []
    for search_seed, shape in product(GP_SEEDS, SHAPES):
        base = {
            "scenario": name,
            "data_seed": bank,
            "gp_seed": search_seed,
            "method": shape,
        }
        results = fit_seed_geometry(x, y, vx, vy, seed=search_seed, shape=shape)
        seeds.extend({**base, **row} for row in results["initial"].initial_rows)
        for horizon, cp in results.items():
            div = {
                k: v
                for k, v in cp.diversity.items()
                if k != "initial_prediction_hashes"
            }
            for decision, tree in (("fit", cp.fit_tree), ("validation", cp.val_tree)):
                meta = {
                    **base,
                    **div,
                    "classes": classes,
                    "providers": x.shape[1],
                    "checkpoint": str(horizon),
                    "selection": decision,
                    "calls": cp.calls,
                    "generation": cp.generation,
                    "nodes": tree.nodes_count,
                    "fit_calls": cp.calls,
                    "validation_calls": cp.n_trees,
                    "archive_trees": cp.n_trees,
                    "path_digest": cp.path_digest,
                    "search_seconds": cp.elapsed_seconds,
                    "archive_selection_brier": cp.fit_brier
                    if decision == "fit"
                    else cp.validation_brier,
                    "program": json.dumps(serialize(tree.root), separators=(",", ":")),
                }
                for split, (z, labels, oracle) in splits.items():
                    p = predict_tree(tree, z)
                    if (
                        p.shape != (len(labels), classes)
                        or not np.isfinite(p).all()
                        or np.any(p < -1e-14)
                        or np.any(p > 1 + 1e-14)
                        or not np.allclose(p.sum(axis=1), 1, atol=2e-12, rtol=0)
                    ):
                        raise ValueError(
                            f"Invalid probability predictions for {name}/{split}"
                        )
                    rows.append(
                        {
                            **meta,
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
                                metrics=["macro_f1", "accuracy", "log_loss"],
                            ),
                        }
                    )
    return rows, seeds


def run(out, workers):
    import pandas as pd

    from fusionbench.scenarios.registry import STANDARD_V1_SCENARIOS as SCENARIOS

    out = Path(out)
    out.mkdir(exist_ok=True, parents=True)
    rows, seeds = [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for r, s in pool.map(panel, product(SCENARIOS, DATA_SEEDS)):
            rows.extend(r)
            seeds.extend(s)
    for name, items in (("standard", rows), ("standard-initial", seeds)):
        pending = out / f"{name}.partial.csv.gz"
        pd.DataFrame(items).to_csv(pending, index=False)
        pending.replace(out / f"{name}.csv.gz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    run(args.out, args.workers)

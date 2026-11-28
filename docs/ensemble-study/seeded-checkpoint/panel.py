"""Reproduce the 24-scenario, two-bank secondary seeded-checkpoint panel."""

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from native_seed import make_founder, restore_founder
from threadpoolctl import threadpool_limits

from fusionbench import generate_dataset
from fusionbench.eval.suite_point_gp import predict_tree
from fusionbench.methods.averaging import SimpleAverage, TopN
from fusionbench.methods.native_pattern import (
    RegularizedNativePatternOKAPI,
    _full_probability,
)
from fusionbench.methods.okapi import OKAPIMethod, _task_postprocessing
from fusionbench.methods.stacking import StackingLR
from fusionbench.transfer import load_preset

HERE = Path(__file__).resolve().parent
SEEDS = (285201, 285202)
ARMS = ("stock", "sham", "seeded", "carry")
METHODS = (*ARMS, "protected", "mean", "top_n", "prob_lr", "regularized")
GENERATIONS = 16


def loss(p, onehot):
    return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))


def matches(p, q):
    return bool(np.max(np.abs(p - q)) <= 1e-12)


def digest(key):
    return hashlib.sha256(repr(key).encode()).hexdigest()


def search(fit_x, fit_y, arm, sham_provider):
    """Fit and freeze both choices before any test covariates or outcomes arrive."""
    from okapi import Okapi
    from okapi.operators import (
        CLOSE_THRESHOLD,
        FAR_THRESHOLD,
        MAX,
        MEAN,
        MIN,
        WEIGHTED_MEAN,
    )

    _, m, c = fit_x.shape
    task = "binary" if c == 2 else "multiclass"
    targets = np.eye(c)[fit_y]
    batch_risks, archive, snapshots = [], {}, []
    calls = 0
    proposals = 0

    def objective(p, _):
        risk = loss(_full_probability(p), targets)
        batch_risks.append(risk)
        return 1 - 0.5 * risk

    class Observed(Okapi):
        def _calculate_fitnesses(self, trees=None):
            nonlocal calls
            trees = self.population if trees is None else trees
            batch_risks.clear()
            scores = super()._calculate_fitnesses(trees)
            calls += len(trees)
            for tree, risk in zip(trees, batch_risks, strict=True):
                archive.setdefault(tree.structural_key(), (tree, risk, generation))
            return scores

        def _mutate_additional_population(self):
            nonlocal proposals
            count = super()._mutate_additional_population()
            proposals += len(self.additional_population)
            return count

    def record(restored):
        snapshots.append(
            {
                "generation": generation,
                "calls": calls,
                "proposals": proposals,
                "restored": restored,
                "keys": tuple(tree.structural_key() for tree in gp.population),
            }
        )

    generation = 0
    capacity = min(8, m)
    adapter = OKAPIMethod(seed=0)
    try:
        directory = adapter._write_split(fit_x, "fit")
        gt = Path(adapter._tmp.name) / "gt.npy"
        np.save(gt, fit_y)
        with _task_postprocessing(task):
            gp = Observed(
                directory,
                gt,
                capacity,
                2,
                2,
                backend="numpy",
                seed=0,
                objective_functions=(objective,),
                minimize_node_count=True,
                max_depth=8,
                max_nodes=max(31, m + 1),
                allowed_ops=(
                    MEAN,
                    MIN,
                    MAX,
                    WEIGHTED_MEAN,
                    FAR_THRESHOLD,
                    CLOSE_THRESHOLD,
                ),
            )
            singleton_ids = tuple(
                tree.unique_value_node_ids[0] for tree in gp.population
            )
            mean = make_founder(gp.train_tensors, "seeded")
            mean_key = mean.structural_key()
            extra = make_founder(gp.train_tensors, arm, sham_provider=sham_provider)
            if extra is not None:
                assert gp._within_caps(extra) and extra.nodes_count == m + 1
                gp.population.append(extra)
            gp.fitnesses = gp._selection_fitnesses(gp.population)
            stored = gp.fitnesses[-1].copy() if arm == "carry" else None
            record(False)
            for generation in range(1, GENERATIONS + 1):
                gp.run_iteration()
                restored = arm == "carry" and restore_founder(gp, mean, stored)
                if len(gp.population) > capacity:
                    raise RuntimeError("Survivors exceed configured capacity")
                record(restored)
            # min() preserves native survivor order when the first two fields tie.
            final_keys = snapshots[-1]["keys"]
            best = min(
                final_keys, key=lambda k: (archive[k][1], archive[k][0].nodes_count)
            )
            protected = None
            if arm == "seeded":
                protected = (
                    best if archive[best][1] < archive[mean_key][1] else mean_key
                )
    finally:
        adapter._tmp.cleanup()
    return {
        "archive": archive,
        "snapshots": snapshots,
        "mean_key": mean_key,
        "best": best,
        "protected": protected,
        "singleton_ids": singleton_ids,
    }


def audit(result, arm, bank, average):
    """Audit every unique evaluated tree and the frozen choices on the held-out bank."""
    archive = result["archive"]
    targets = np.eye(bank.n_classes)[bank.test_y]
    audited = {}
    for key, (tree, _, _) in archive.items():
        test = predict_tree(tree, bank.test_preds)
        equal = matches(test, average[1]) and matches(
            predict_tree(tree, bank.fit_preds), average[0]
        )
        audited[key] = (loss(test, targets), equal)
    trace = []
    for snap in result["snapshots"]:
        keys = snap["keys"]
        gen = snap["generation"]
        trace.append(
            {
                "arm": arm,
                "generation": gen,
                "survivors": len(keys),
                "objective_evaluations": snap["calls"],
                "offspring_proposals": snap["proposals"],
                "genotype_mean_active": result["mean_key"] in keys,
                "functional_mean_active": any(audited[k][1] for k in keys),
                "restored": snap["restored"],
                "best_active_test_brier": min(audited[k][0] for k in keys),
                "active_digest": digest(keys),
            }
        )
    deployed = [(arm, result["best"])]
    if arm == "seeded":
        deployed.append(("protected", result["protected"]))
    scores = []
    for name, key in deployed:
        tree, risk, _ = archive[key]
        scores.append(
            {
                "method": name,
                "shared_search": arm,
                "fit_brier": risk,
                "test_brier": audited[key][0],
                "selected_is_mean": audited[key][1],
                "unique_providers": len(tree.unique_value_node_ids),
                "selected_nodes": tree.nodes_count,
                "selected_digest": digest(key),
                "unique_evaluated": len(archive),
                "best_ever_test_brier": min(p[0] for p in audited.values()),
                "best_final_test_brier": trace[-1]["best_active_test_brier"],
                "objective_evaluations": trace[-1]["objective_evaluations"],
                "offspring_proposals": trace[-1]["offspring_proposals"],
                "restorations": sum(int(r["restored"]) for r in trace),
                "initial_singleton_ids": " ".join(result["singleton_ids"]),
            }
        )
    return scores, trace


def evaluate(job):
    idx, scenario, seed = job
    with threadpool_limits(limits=1):
        bank = generate_dataset(scenario, n_per_split=512, seed=seed)
        m, c = bank.fit_preds.shape[1:]
        sham = int(
            np.random.default_rng(np.random.SeedSequence([285000, idx, seed])).integers(
                m
            )
        )
        average = [
            SimpleAverage().predict(p) for p in (bank.fit_preds, bank.test_preds)
        ]
        context = {
            "scenario_id": scenario,
            "data_seed": seed,
            "n_classes": c,
            "pool_size": m,
            "max_nodes": max(31, m + 1),
            "search_seed": 0,
            "sham_provider": sham,
        }
        scores, traces = [], []
        for arm in ARMS:
            start = perf_counter()
            result = search(bank.fit_preds, bank.fit_y, arm, sham)
            fit_seconds = perf_counter() - start
            arm_scores, trace = audit(result, arm, bank, average)
            scores.extend(
                context | row | {"fit_seconds": fit_seconds} for row in arm_scores
            )
            traces.extend(context | row for row in trace)

        for name, model in (
            ("mean", SimpleAverage()),
            ("top_n", TopN(metric="brier")),
            ("prob_lr", StackingLR(c_value=1.0)),
            ("regularized", RegularizedNativePatternOKAPI(seed=0, split_seed=0)),
        ):
            start = perf_counter()
            model.fit(bank.fit_preds, bank.fit_y, task=bank.task, n_classes=c)
            fit_seconds = perf_counter() - start
            predictions = [model.predict(p) for p in (bank.fit_preds, bank.test_preds)]
            count = m
            if name == "top_n":
                count = int(model.n_selected_)
            if name == "regularized":
                count = len(model.trees_[model.chosen_].unique_value_node_ids)
            scores.append(
                context
                | {
                    "method": name,
                    "shared_search": name,
                    "fit_brier": loss(predictions[0], np.eye(c)[bank.fit_y]),
                    "test_brier": loss(predictions[1], np.eye(c)[bank.test_y]),
                    "selected_is_mean": all(
                        matches(p, q) for p, q in zip(predictions, average, strict=True)
                    ),
                    "unique_providers": count,
                    "fit_seconds": fit_seconds,
                }
            )
    return scores, traces


def compare(table, family_set):
    """Pair banks inside each fixed scenario before averaging scenarios equally."""
    table = table[table.scenario_id.isin(family_set)]
    wide = table.pivot(
        index=["scenario_id", "data_seed"], columns="method", values="test_brier"
    )
    order = pd.MultiIndex.from_product([family_set, SEEDS], names=wide.index.names)
    wide = wide.reindex(order)
    f = len(family_set)
    draws = np.random.default_rng(97822).integers(0, 2, size=(10000, f, 2))
    pairs = [
        ("seeded", "sham"),
        ("seeded", "stock"),
        ("protected", "seeded"),
        ("carry", "protected"),
        ("carry", "seeded"),
        ("carry", "sham"),
    ]
    pairs += [
        (arm, ref)
        for arm in ("sham", "seeded", "protected", "carry")
        for ref in ("mean", "top_n", "prob_lr", "regularized")
    ]
    rows = []
    for a, b in pairs:
        values = (wide[a] - wide[b]).to_numpy().reshape(f, 2)
        draws_means = values[np.arange(f)[None, :, None], draws].mean(axis=(1, 2))
        lower, upper = np.quantile(draws_means, [0.025, 0.975])
        rows.append(
            {
                "families": f,
                "method": a,
                "comparator": b,
                "mean_delta": values.mean(),
                "ci_lower": lower,
                "ci_upper": upper,
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    scenarios = tuple(load_preset("standard-v3")["scenarios"])
    if len(scenarios) != 24:
        raise ValueError("Expected the frozen standard-v3 panel of 24 families")
    jobs = [
        (i, scenario, seed) for i, scenario in enumerate(scenarios) for seed in SEEDS
    ]
    start = perf_counter()
    rows, traces = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for scores, trace in pool.map(evaluate, jobs):
            rows.extend(scores)
            traces.extend(trace)
    scores, logs = pd.DataFrame(rows), pd.DataFrame(traces)
    expected = {(s, d, m) for s in scenarios for d in SEEDS for m in METHODS}
    if (
        set(
            scores[["scenario_id", "data_seed", "method"]].itertuples(
                index=False, name=None
            )
        )
        != expected
    ):
        raise RuntimeError("Missing panel scores")
    keys = ["scenario_id", "data_seed"]
    if (
        len(scores) != len(expected)
        or len(logs) != len(jobs) * len(ARMS) * (GENERATIONS + 1)
        or logs.duplicated(keys + ["arm", "generation"]).any()
        or not np.isfinite(scores[["fit_brier", "test_brier"]]).all().all()
    ):
        raise RuntimeError("Duplicate, missing or nonfinite results")
    adjusted = scores.loc[scores.max_nodes.gt(31), "scenario_id"].unique().tolist()
    if adjusted != ["needle"]:
        raise RuntimeError(f"Unexpected cap exception: {adjusted}")
    actual = scores[scores.method.isin(ARMS)].groupby(keys)
    if actual.initial_singleton_ids.nunique().max() != 1:
        raise RuntimeError("Unmatched initial singleton choices")
    if not logs[logs.arm.eq("carry")].genotype_mean_active.all():
        raise RuntimeError("Carry failed to retain the canonical mean")
    fixed = scores.pivot(index=keys, columns="method", values="fit_brier")
    if (fixed.protected > fixed[["mean", "seeded"]].min(axis=1) + 1e-12).any():
        raise RuntimeError("Protected deployment failed its training-fit bound")
    contrasts = pd.DataFrame(
        compare(scores, scenarios)
        + compare(scores, [s for s in scenarios if s != "needle"])
    )
    means = scores.pivot_table(
        index="scenario_id", columns="method", values="test_brier"
    ).reindex(scenarios)
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    scores.to_csv(out / "panel-scores.csv", index=False)
    logs.to_csv(out / "panel-trajectories.csv", index=False)
    means.to_csv(out / "panel-family-means.csv")
    contrasts.to_csv(out / "panel-contrasts.csv", index=False)
    report = {
        "scenarios": len(scenarios),
        "banks": len(jobs),
        "scores": len(scores),
        "trajectories": len(logs),
        "search_fits": len(jobs) * len(ARMS),
        "workers": args.workers,
        "numeric_threads_per_worker": 1,
        "elapsed_seconds": perf_counter() - start,
        "family_equal_mean_test_brier": means.mean().to_dict(),
        "methods": list(METHODS),
        "bootstrap_seed": 97822,
        "bootstrap_repetitions": 10000,
        "max_nodes_exception": adjusted,
        "accounting": "Protected and seeded share one search and report its same realized fit cost. Objective calls include rescored survivors. Test oracle summaries were computed after deployment choices.",
    }
    (out / "panel-summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

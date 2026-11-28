"""Standard-v4 regression of two native search paths and their frozen terminal rules."""

import argparse
import hashlib
import json
import platform
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits

from fusionbench import generate_dataset
from fusionbench.eval.transfer_grid import brier
from fusionbench.methods.averaging import SimpleAverage, TopN
from fusionbench.methods.native_pattern import RegularizedNativePatternOKAPI
from fusionbench.methods.stacking import StackingLR
from fusionbench.transfer import load_preset

SEEDS = (287300, 287301)
N_ROWS = 512
NATIVE = (
    "pooled_final",
    "pooled_pooled",
    "pooled_fold",
    "fold_pooled",
    "fold_fold",
    "anchor",
)
CONTROLS = ("mean", "top_n", "prob_lr", "regularized")
METHODS = (*NATIVE, *CONTROLS)


def digest(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def references(bank):
    x, y = bank.fit_preds, bank.fit_y
    n_classes = x.shape[-1]
    task = "binary" if n_classes == 2 else "multiclass"
    models = {
        "mean": SimpleAverage(),
        "top_n": TopN(metric="brier"),
        "prob_lr": StackingLR(c_value=1.0),
        "regularized": RegularizedNativePatternOKAPI(seed=0, split_seed=0),
    }
    rows = []
    for method, model in models.items():
        start = perf_counter()
        model.fit(x, y, task=task, n_classes=n_classes)
        row = {
            "method": method,
            "fit_seconds": perf_counter() - start,
            "unique_providers": x.shape[1],
            "fit_brier": brier(model.predict(x), y),
        }
        if method == "top_n":
            row["unique_providers"] = model.n_selected_
        if method == "regularized":
            row["family"] = model.chosen_
            row["objective_evaluations"] = model.counts_["total_score_calls"]
            row["unique_providers"] = len(
                model.trees_[model.chosen_].unique_value_node_ids
            )
        predicted = model.predict(bank.test_preds)
        rows.append(
            {
                **row,
                "test_brier": brier(predicted, bank.test_y),
                "predictions_sha256": digest(predicted),
            }
        )
    return rows


def evaluate(job):
    from search import fit_search, score_search

    scenario, seed = job
    with threadpool_limits(limits=1):
        bank = generate_dataset(scenario, seed=seed, n_per_split=N_ROWS)
        x, y = bank.fit_preds, bank.fit_y
        context = {
            "scenario_id": scenario,
            "data_seed": seed,
            "search_seed": 0,
            "fold_seed": seed + 3_000_000,
            "n_fit": len(y),
            "n_test": len(bank.test_y),
            "provider_count": x.shape[1],
            "n_classes": x.shape[2],
            "fit_inputs_sha256": digest(x),
            "fit_labels_sha256": digest(y),
            "test_inputs_sha256": digest(bank.test_preds),
            "test_labels_sha256": digest(bank.test_y),
        }
        groups = np.arange(len(y))
        # Every terminal choice reuses its path's one fitted search.
        trained = {}
        for path in ("pooled", "fold"):
            start = perf_counter()
            result = fit_search(
                x, y, groups=groups, data_seed=seed, search_seed=0, path=path
            )
            trained[path] = (result, perf_counter() - start)
        rows, traces = [], []
        for path, (result, elapsed) in trained.items():
            selected, trajectory = score_search(result, bank.test_preds, bank.test_y)
            for row in selected:
                row = dict(row)
                row["test_brier"] = row.pop("audit_brier")
                rows.append({**row, "path": path, "fit_seconds": elapsed})
            traces.extend({**context, **stage} for stage in trajectory)
        if {row["method"] for row in rows} != set(NATIVE) or len(rows) != len(NATIVE):
            raise ValueError("Incomplete deployment menu")
        if len({row["anchor_digest"] for row in traces}) != 1:
            raise ValueError("The two native searches used different anchors")
        rows.extend(references(bank))
        return [{**context, **row} for row in rows], traces


def summarize(scores, families):
    methods = list(METHODS)
    index = pd.MultiIndex.from_product(
        [families, SEEDS], names=["scenario_id", "data_seed"]
    )
    cells = scores.pivot(
        index=["scenario_id", "data_seed"], columns="method", values="test_brier"
    )
    cells = cells.reindex(index=index, columns=methods)
    if not np.isfinite(cells).all().all() or scores.shape[0] != len(index) * len(
        methods
    ):
        raise ValueError("Incomplete family-by-seed-by-method grid")
    draws = np.random.default_rng(97825).integers(2, size=(10_000, len(families), 2))
    family_idx = np.arange(len(families))[None, :, None]

    def estimate(series):
        values = series.to_numpy().reshape(len(families), 2)
        lo, hi = np.quantile(
            values[family_idx, draws].mean(axis=(1, 2)), [0.025, 0.975]
        )
        return {
            "estimate": values.mean(),
            "ci_low": lo,
            "ci_high": hi,
            "n_families": len(families),
        }

    levels = [{"method": method, **estimate(cells[method])} for method in methods]
    pairs = [
        ("fold_fold", "pooled_pooled"),
        ("fold_fold", "pooled_fold"),
        ("pooled_fold", "pooled_pooled"),
        ("fold_pooled", "pooled_pooled"),
        ("fold_fold", "fold_pooled"),
        ("pooled_pooled", "pooled_final"),
        ("pooled_fold", "pooled_final"),
        *(("fold_fold", ref) for ref in ("pooled_final", "anchor", *CONTROLS)),
    ]
    per_family, effects = [], []
    for method, reference in pairs:
        difference = cells[method] - cells[reference]
        means = difference.groupby(level="scenario_id").mean()
        per_family.extend(
            {
                "scenario_id": f,
                "method": method,
                "reference": reference,
                "estimate": means[f],
            }
            for f in families
        )
        effects.append(
            {
                "method": method,
                "reference": reference,
                **estimate(difference),
                "n_lower": int((means < -1e-12).sum()),
                "n_tie": int((abs(means) <= 1e-12).sum()),
                "n_higher": int((means > 1e-12).sum()),
            }
        )
    return {
        "panel-means": pd.DataFrame(levels),
        "panel-contrasts": pd.DataFrame(effects),
        "panel-family-means": cells.groupby(level="scenario_id")
        .mean()
        .loc[families]
        .reset_index(),
        "panel-family-contrasts": pd.DataFrame(per_family),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=Path(__file__).resolve().parent / "results"
    )
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    families = list(load_preset("standard-v4")["scenarios"])
    if len(families) != 25 or len(set(families)) != 25:
        raise ValueError("Expected exactly 25 distinct standard-v4 families")
    jobs = [(family, seed) for family in families for seed in SEEDS]
    start = perf_counter()
    rows, stages = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for selected, traces in pool.map(evaluate, jobs):
            rows.extend(selected)
            stages.extend(traces)
    scores, trajectories = pd.DataFrame(rows), pd.DataFrame(stages)
    tables = summarize(scores, families)
    if len(trajectories) != len(jobs) * 2 * 17:
        raise ValueError("Unexpected number of native trace rows")
    args.out.mkdir(parents=True, exist_ok=True)
    tables.update({"panel-scores": scores, "panel-trajectories": trajectories})
    for name, data in tables.items():
        data.to_csv(args.out / (name + ".csv"), index=False)
    manifest = {
        "suite": "standard-v4",
        "families": families,
        "seeds": list(SEEDS),
        "search_seed": 0,
        "fold_seed_offset": 3_000_000,
        "n_fit": N_ROWS,
        "n_test": N_ROWS,
        "bootstrap_draws": 10_000,
        "bootstrap_seed": 97825,
        "workers": args.workers,
        "numeric_threads_per_worker": 1,
        "fitted_gp_paths": 2 * len(jobs),
        "deployments": len(scores),
        "trajectory_rows": len(trajectories),
        "seconds": perf_counter() - start,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
    }
    (args.out / "panel-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(tables["panel-contrasts"].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

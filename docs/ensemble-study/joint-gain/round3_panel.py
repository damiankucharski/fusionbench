"""Compare matched native breeding and deployment rules on the 25 fixed standard-v4 families."""

import argparse
import json
import platform
import sys
from pathlib import Path
from runpy import run_path
from time import perf_counter

import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits

from fusionbench import generate_dataset
from fusionbench.eval.suite_point_gp import predict_tree
from fusionbench.eval.transfer_grid import brier
from fusionbench.transfer import load_preset

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "fold-promotion"
sys.path.insert(0, str(PREVIOUS))
previous = run_path(str(PREVIOUS / "panel.py"))
references, digest = previous["references"], previous["digest"]

SEEDS = (288400, 288401, 288402, 288403)
PATHS = ("pooled", "fold", "joint")
RULES = ("pooled", "fold", "veto", "joint")
NATIVE = tuple(f"{p}_{r}" for p in PATHS for r in RULES) + ("anchor", "pooled_final")
CONTROLS = ("mean", "top_n", "prob_lr", "regularized")
METHODS = NATIVE + CONTROLS
PRIMARY = ("joint_joint", "fold_joint")
PAIRS = (
    PRIMARY,
    ("joint_joint", "fold_fold"),
    ("joint_joint", "pooled_final"),
    ("joint_joint", "pooled_pooled"),
    ("joint_joint", "pooled_joint"),
    ("joint_joint", "joint_pooled"),
    ("joint_joint", "joint_fold"),
    ("joint_joint", "joint_veto"),
    ("fold_joint", "fold_fold"),
    ("pooled_joint", "pooled_pooled"),
    ("joint_pooled", "fold_pooled"),
    *(("joint_joint", ref) for ref in ("anchor", *CONTROLS)),
)


def evaluate(family, seed):
    from search import fit_search, score_search

    bank = generate_dataset(family, seed=seed, n_per_split=512)
    x, y = bank.fit_preds, bank.fit_y
    context = {
        "scenario_id": family,
        "data_seed": seed,
        "search_seed": 0,
        "fold_seed": seed + 3_000_000,
        "n_fit": len(y),
        "n_test": len(bank.test_y),
        "provider_count": x.shape[1],
        "n_classes": x.shape[-1],
        "fit_inputs_sha256": digest(x),
        "fit_labels_sha256": digest(y),
        "test_inputs_sha256": digest(bank.test_preds),
        "test_labels_sha256": digest(bank.test_y),
    }
    groups = np.arange(len(y))
    trained = {}
    for path in PATHS:
        started = perf_counter()
        result = fit_search(
            x,
            y,
            groups=groups,
            data_seed=seed,
            search_seed=0,
            path=path,
            joint_study=True,
        )
        trained[path] = (result, perf_counter() - started)
    rows, stages = [], []
    for path, (result, elapsed) in trained.items():
        selected, trace = score_search(result, bank.test_preds, bank.test_y)
        # Confirm every reported deployment with one direct audit forward per distinct genotype.
        checked = {}
        for row in selected:
            key = result["choices"][row["method"]]
            if key not in checked:
                predictions = predict_tree(
                    result["archive"][key]["tree"], bank.test_preds
                )
                checked[key] = (brier(predictions, bank.test_y), digest(predictions))
            risk, prediction_hash = checked[key]
            if abs(risk - row["audit_brier"]) > 1e-12:
                raise ValueError(
                    "Fresh held-out scoring disagreed with the shared search evaluator"
                )
            row = {k: v for k, v in row.items() if v is not None and k != "audit_brier"}
            rows.append(
                {
                    **context,
                    **row,
                    "test_brier": risk,
                    "predictions_sha256": prediction_hash,
                    "path_fit_seconds": elapsed,
                }
            )
        stages.extend(
            {**context, **{k: v for k, v in s.items() if v is not None}} for s in trace
        )
    if len(rows) != len(NATIVE) or {r["method"] for r in rows} != set(NATIVE):
        raise ValueError("Incomplete native deployment menu")
    for field in ("anchor_digest", "fold_digest", "founder_digest"):
        if len({s[field] for s in stages}) != 1:
            raise ValueError(f"Unmatched {field} across paths")
    for ref in references(bank):
        rows.append({**context, **ref})
    return rows, stages


def summarize(scores, families):
    index = pd.MultiIndex.from_product(
        [families, SEEDS], names=["scenario_id", "data_seed"]
    )
    cells = scores.pivot(
        index=["scenario_id", "data_seed"], columns="method", values="test_brier"
    ).reindex(index=index, columns=METHODS)
    if (
        scores.shape[0] != len(index) * len(METHODS)
        or not np.isfinite(cells).all().all()
    ):
        raise ValueError("Missing or duplicate standard-panel results")
    draws = np.random.default_rng(97828).integers(
        len(SEEDS), size=(10_000, len(families), len(SEEDS))
    )
    family_index = np.arange(len(families))[None, :, None]

    def estimate(series):
        matrix = series.to_numpy().reshape(len(families), len(SEEDS))
        low, high = np.quantile(
            matrix[family_index, draws].mean(axis=(1, 2)), [0.025, 0.975]
        )
        return {
            "estimate": matrix.mean(),
            "ci_low": low,
            "ci_high": high,
            "n_families": len(families),
        }

    levels = [{"method": m, **estimate(cells[m])} for m in METHODS]
    contrasts, by_family = [], []
    for method, reference in PAIRS:
        diffs = cells[method] - cells[reference]
        means = diffs.groupby(level="scenario_id").mean()
        row = {
            "method": method,
            "reference": reference,
            **estimate(diffs),
            "n_better": int((means < -1e-12).sum()),
            "n_tied": int((abs(means) <= 1e-12).sum()),
            "n_worse": int((means > 1e-12).sum()),
        }
        contrasts.append(row)
        for f in families:
            seeds = diffs.loc[f]
            by_family.append(
                {
                    "scenario_id": f,
                    "method": method,
                    "reference": reference,
                    "estimate": means[f],
                    "n_seed_better": int((seeds < -1e-12).sum()),
                    "n_seed_tied": int((abs(seeds) <= 1e-12).sum()),
                    "n_seed_worse": int((seeds > 1e-12).sum()),
                }
            )
    return {
        "means": pd.DataFrame(levels),
        "contrasts": pd.DataFrame(contrasts),
        "family-means": cells.groupby(level="scenario_id")
        .mean()
        .loc[families]
        .reset_index(),
        "family-contrasts": pd.DataFrame(by_family),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "results")
    args = parser.parse_args()
    families = list(load_preset("standard-v4")["scenarios"])
    if len(set(families)) != 25 or len(families) != 25:
        raise ValueError("Expected exactly the standard-v4 25-family panel")
    started = perf_counter()
    rows, stages = [], []
    with threadpool_limits(limits=1):
        for family in families:
            for seed in SEEDS:
                selected, trace = evaluate(family, seed)
                rows.extend(selected)
                stages.extend(trace)
            print(f"Finished {family}", flush=True)
    scores, traces = pd.DataFrame(rows), pd.DataFrame(stages)
    if len(traces) != len(families) * len(SEEDS) * len(PATHS) * 17:
        raise ValueError("Incomplete native generation trace")
    tables = summarize(scores, families)
    final = traces[traces.generation == 16]
    tables["path-costs"] = (
        final.groupby("path")[
            ["objective_evaluations", "offspring_proposals", "unique_evaluated"]
        ]
        .mean()
        .reset_index()
    )
    tables["scores"] = scores
    args.out.mkdir(parents=True, exist_ok=True)
    for name, values in tables.items():
        values.to_csv(args.out / ("r3-panel-" + name + ".csv"), index=False)
    traces.to_csv(
        args.out / "r3-panel-trajectories.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    manifest = {
        "suite": "standard-v4",
        "families": families,
        "seeds": list(SEEDS),
        "search_seed": 0,
        "fold_seed_offset": 3_000_000,
        "n_fit": 512,
        "n_test": 512,
        "bootstrap_seed": 97828,
        "bootstrap_draws": 10_000,
        "workers": 1,
        "numerical_threads": 1,
        "live_searches": len(final),
        "score_rows": len(scores),
        "trajectory_rows": len(traces),
        "seconds": perf_counter() - started,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
    }
    (args.out / "r3-panel-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(tables["contrasts"].iloc[0].to_string(), flush=True)


if __name__ == "__main__":
    main()

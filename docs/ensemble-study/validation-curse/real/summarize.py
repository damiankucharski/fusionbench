"""Summarize paired real banks, preserving the shared Digits dataset as one unit."""

import argparse
import json
from pathlib import Path

import pandas as pd

STUDY = Path(__file__).resolve().parent
INDEX = ["dataset", "pool", "replicate"]
METHODS = [
    "mean",
    "top_n",
    "prob_lr",
    "logit_lr",
    "incumbent",
    "search_z0",
    "search_z1",
    "inner_incumbent",
    "raw_plus_mean",
    "guarded",
]


def add_aggregates(frame):
    # Equal representation within the pool and within the underlying dataset.
    by_pool = frame.groupby(["dataset", "pool"]).mean(numeric_only=True)
    four = by_pool.groupby("dataset").mean().mean()
    six = by_pool.mean()
    report = by_pool.droplevel("dataset")
    report.loc["four_dataset_macro"] = four
    report.loc["six_pool_macro"] = six
    return report


def summarize(out):
    scores = pd.read_csv(out / "scores.csv.gz")
    gate = pd.read_csv(out / "gate.csv")
    oracle = pd.read_csv(out / "final-family-oracle.csv")
    brier = scores.pivot(index=INDEX, columns="method", values="test_brier")
    accuracy = scores.pivot(index=INDEX, columns="method", values="test_accuracy")
    if len(brier) != 48 or brier.isna().any().any():
        raise ValueError("Expected a complete paired panel of 48 partitions")
    groups = [
        ("search_z1", "search_z0"),
        ("search_z1", "mean"),
        ("search_z1", "top_n"),
        ("search_z1", "prob_lr"),
        ("search_z1", "logit_lr"),
        ("guarded", "raw_plus_mean"),
        ("guarded", "incumbent"),
        ("guarded", "inner_incumbent"),
        ("inner_incumbent", "incumbent"),
    ]
    deltas = pd.DataFrame({f"{a}_minus_{b}": brier[a] - brier[b] for a, b in groups})
    for metric, table in [
        ("brier", brier[METHODS]),
        ("accuracy", accuracy[METHODS]),
        ("differences", deltas),
    ]:
        add_aggregates(table).to_csv(out / f"pool-{metric}.csv", index_label="pool")

    def contrast_detail(d):
        eps = 1e-12
        return {
            "n": len(d),
            "better": int((d < -eps).sum()),
            "tied": int((d.abs() <= eps).sum()),
            "worse": int((d > eps).sum()),
            "min_difference": d.min(),
            "mean_difference": d.mean(),
            "max_difference": d.max(),
        }

    pd.DataFrame(
        [
            {"pool": pool, "contrast": name, **contrast_detail(part)}
            for name, col in deltas.items()
            for pool, part in col.groupby("pool")
        ]
    ).to_csv(out / "paired-direction.csv", index=False)

    counts = (
        scores[scores.method.isin(METHODS) & scores.choice.notna()]
        .groupby(["pool", "method", "choice"])
        .size()
        .rename("n")
    )
    counts.to_csv(out / "selected-frequencies.csv")
    pd.concat(
        {
            column: add_aggregates(
                gate.pivot(index=INDEX, columns="method", values=column)
            )
            for column in [
                "outer_n",
                "gate_minus_mean",
                "test_minus_mean",
                "relative_optimism",
            ]
        },
        names=["statistic", "pool"],
    ).to_csv(out / "pool-gate.csv")
    pd.concat(
        {
            column: add_aggregates(
                oracle.pivot(index=INDEX, columns="search_method", values=column)
            )
            for column in [
                "test_brier",
                "test_three_oracle_brier",
                "selection_regret",
                "test_oracle_if_mean_allowed",
            ]
        },
        names=["statistic", "pool"],
    ).to_csv(out / "pool-oracle.csv")

    search = scores[scores.method.isin(["incumbent", "search_z0", "search_z1"])]
    sums = search.groupby("method")[["score_calls", "fit_seconds"]].agg(
        ["sum", "mean", "min", "max"]
    )
    sums.to_csv(out / "budget.csv")
    pred = scores.pivot(index=INDEX, columns="method", values="prediction_sha256")
    meta = {
        "partitions": len(brier),
        "incumbent_z0_prediction_hash_matches": int(
            (pred.incumbent == pred.search_z0).sum()
        ),
        "incumbent_z0_risk_exact": int((brier.incumbent == brier.search_z0).sum()),
        "z0_z1_prediction_hash_matches": int((pred.search_z1 == pred.search_z0).sum()),
        "raw_guarded_prediction_hash_matches": int(
            (pred.raw_plus_mean == pred.guarded).sum()
        ),
        "secondary_terminals_select_mean": {
            m: int(((gate.method == m) & (gate.selected == "mean")).sum())
            for m in ["raw_plus_mean", "guarded"]
        },
        "primary_wins_ties_losses": contrast_detail(deltas.search_z1_minus_search_z0),
        "secondary_wins_ties_losses": contrast_detail(
            deltas.guarded_minus_raw_plus_mean
        ),
    }
    (out / "summary.json").write_text(
        json.dumps(meta, indent=2, default=lambda x: x.item()) + "\n"
    )
    return meta


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=STUDY / "results")
    args = p.parse_args()
    print(json.dumps(summarize(args.out), indent=2, default=lambda x: x.item()))

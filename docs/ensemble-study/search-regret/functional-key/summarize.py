"""Reproduce means and paired intervals from the raw functional-key audit."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

LOSSES = ["fit", "validation", "iid", "shift", "fit_regret", "exact_hit"]
SEARCH = [
    "calls",
    "completed_generations",
    "observed_joins",
    "proposals",
    "redundant_structural_firsts",
    "distinct_structures",
    "distinct_functions",
    "mean_essential",
    "max_essential",
    "nodes",
    "structural_mask_seconds",
    "functional_mask_seconds",
    "audit_wall_seconds",
]


def summarize(path):
    rows = pd.read_csv(path / "gp.csv")
    keys = ["checkpoint", "champion", "data_seed", "gp_seed"]
    assert len(rows) == 12 * 4 * 2 * 2 * 3
    assert not rows.duplicated(keys + ["policy"]).any()
    numeric = LOSSES + SEARCH
    by_seed = rows.groupby(["checkpoint", "champion", "data_seed", "policy"])[
        numeric
    ].mean()
    by_seed.groupby(["checkpoint", "champion", "policy"]).mean().to_csv(
        path / "means.csv"
    )
    paired = rows.pivot(index=keys, columns="policy", values=numeric)
    effects = paired.xs("functional", level="policy", axis=1) - paired.xs(
        "structural", level="policy", axis=1
    )
    seeds = effects.groupby(["checkpoint", "champion", "data_seed"]).mean()
    summary = []
    for (checkpoint, champion), group in seeds.groupby(["checkpoint", "champion"]):
        for metric in numeric:
            values = group[metric]
            center = values.mean()
            half = t.ppf(0.975, len(values) - 1) * values.sem()
            summary.append(
                {
                    "checkpoint": checkpoint,
                    "champion": champion,
                    "metric": metric,
                    "mean_difference": center,
                    "low_95": center - half,
                    "high_95": center + half,
                    "n_datasets": len(values),
                }
            )
    pd.DataFrame(summary).to_csv(path / "paired.csv", index=False)
    for metric in ("tree", "prediction_sha256"):
        wide = rows.pivot(index=keys, columns="policy", values=metric)
        print(
            metric,
            (wide.structural != wide.functional)
            .groupby(["checkpoint", "champion"])
            .sum()
            .to_dict(),
        )
    fit = effects.xs("fit", level="champion")["fit_regret"]
    print("fit regret sign counts (negative, zero, positive)")
    print(
        fit.groupby("checkpoint").agg(
            lambda x: tuple(np.sum(np.sign(x) == k) for k in (-1, 0, 1))
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    summarize(parser.parse_args().path)

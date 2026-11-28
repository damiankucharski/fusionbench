"""Check the confirmation run's paired target laws without fitting any method."""

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "results"
WORLDS = ("best_world", "mean_world")


def summarize(scores):
    # The retrospective portfolio oracle uses target labels and is outside the bound.
    rows = scores.loc[
        (scores.cohort == "nfl") & (scores.method != "portfolio_oracle")
    ].copy()
    if rows.empty or rows.duplicated(["family", "seed", "method"]).any():
        raise ValueError("Expected pairs of distinct target worlds per method and seed")
    ids = ["method", "seed"]
    source_cols = [
        "fit_x_sha256",
        "fit_y_sha256",
        "audit_x_sha256",
        "audit_uniform_sha256",
        "fit_hash",
        "test_hash",
        "action",
        "common_rule_excess_bound",
    ]
    if (
        rows[source_cols].isna().any().any()
        or not (rows.groupby(ids)[source_cols].nunique() == 1).all().all()
    ):
        raise ValueError(
            "The two target worlds must share inputs, choices and forecasts"
        )

    rows["expected_excess"] = rows.expected_brier - rows.bayes_brier
    rows["observed_excess"] = rows.brier - rows.observed_bayes_brier
    values = ["expected_excess", "observed_excess", "brier", "common_rule_excess_bound"]
    wide = rows.pivot(index=ids, columns="family", values=values)
    if (
        set(wide.columns.get_level_values("family")) != set(WORLDS)
        or not np.isfinite(wide.to_numpy()).all()
    ):
        raise ValueError("Missing world or non-finite risk in a pair")
    paired = pd.DataFrame(index=wide.index)
    for world in WORLDS:
        paired[world + "_expected_excess"] = wide["expected_excess", world]
        paired[world + "_observed_brier"] = wide["brier", world]
    paired["mean_expected_excess"] = wide["expected_excess"].mean(axis=1)
    paired["mean_observed_excess"] = wide["observed_excess"].mean(axis=1)
    paired["common_rule_excess_bound"] = wide["common_rule_excess_bound", WORLDS[0]]
    paired["excess_above_bound"] = (
        paired.mean_expected_excess - paired.common_rule_excess_bound
    )
    if (paired.excess_above_bound < -1e-12).any():
        raise ValueError("A shared forecast violates the exact two-world risk identity")
    by_method = paired.groupby(level="method").mean()
    by_method.insert(0, "paired_draws", paired.groupby(level="method").size())
    return paired, by_method


if __name__ == "__main__":
    paired, by_method = summarize(pd.read_csv(OUT / "r3-scores.csv"))
    if not (by_method.paired_draws == 16).all():
        raise ValueError("The confirmation protocol requires 16 paired draws")
    paired.to_csv(OUT / "r3-witness-pairs.csv")
    by_method.to_csv(OUT / "r3-witness-summary.csv")

"""Summarize fresh synthetic banks separately from overlapping outer-training folds."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

CONTRASTS = (
    ("retype", "identity"),
    ("retype", "refresh"),
    ("refresh", "identity"),
    ("identity", "stock"),
    ("retype", "stock"),
    ("refresh_independent", "refresh"),
    ("refresh_independent", "identity"),
    ("retype", "refresh_independent"),
)
SCORES = ["brier", "conditional_excess", "macro_f1", "accuracy", "log_loss", "nodes"]


def estimates(values):
    x = pd.Series(values).dropna()
    mean = x.mean()
    sd = x.std()
    margin = t.ppf(0.975, len(x) - 1) * sd / np.sqrt(len(x)) if len(x) > 1 else np.nan
    return {
        "banks": len(x),
        "mean": mean,
        "sd": sd,
        "low_95": mean - margin,
        "high_95": mean + margin,
    }


def synthetic(raw, folder):
    groups = ["scenario", "split", "selection", "checkpoint"]
    bank_methods = (
        raw.groupby([*groups, "method", "data_seed"])[SCORES].mean().reset_index()
    )
    long = bank_methods.melt(
        id_vars=[*groups, "method", "data_seed"],
        value_vars=SCORES,
        var_name="metric",
        value_name="value",
    )
    stats = [
        {**dict(zip([*groups, "method", "metric"], key)), **estimates(block.value)}
        for key, block in long.groupby([*groups, "method", "metric"])
        if block.value.notna().any()
    ]
    pd.DataFrame(stats).to_csv(folder / "methods.csv.gz", index=False)
    comparisons = []
    for key, block in long[long.checkpoint.ne("fixed")].groupby([*groups, "metric"]):
        table = block.pivot(index="data_seed", columns="method", values="value")
        for a, b in CONTRASTS:
            deltas = table[a] - table[b]
            if deltas.notna().any():
                comparisons.append(
                    {
                        **dict(zip([*groups, "metric"], key)),
                        "a": a,
                        "b": b,
                        **estimates(deltas),
                    }
                )
    pd.DataFrame(comparisons).to_csv(folder / "paired.csv.gz", index=False)
    # Aggregate bank first across families; do not treat two restarts or 19 families as six new data replications.
    scenario_bank = bank_methods.groupby(
        ["split", "selection", "checkpoint", "method", "data_seed"]
    )[SCORES].mean()
    aggregate = []
    for key, block in scenario_bank.reset_index().groupby(
        ["split", "selection", "checkpoint"]
    ):
        if key[2] == "fixed":
            continue
        for metric in SCORES:
            tab = block.pivot(index="data_seed", columns="method", values=metric)
            for a, b in CONTRASTS:
                dif = tab[a] - tab[b]
                if dif.notna().any():
                    aggregate.append(
                        {
                            **dict(zip(["split", "selection", "checkpoint"], key)),
                            "metric": metric,
                            "a": a,
                            "b": b,
                            **estimates(dif),
                            "bank_differences": ",".join(f"{x:.12g}" for x in dif),
                        }
                    )
    pd.DataFrame(aggregate).to_csv(folder / "suite-contrasts.csv", index=False)
    grouped = raw[
        raw.split.eq("iid") & raw.selection.eq("fit") & raw.checkpoint.ne("fixed")
    ]
    costs = [
        "calls",
        "archive_trees",
        "completed_generations",
        "generation",
        "joined_trees",
        "dedup_removed",
        "structural_mutations",
        "point_eligible_structural_mutations",
        "point_opportunities",
        "point_original_parametric",
        "point_program_changes",
        "point_family_changes",
        "nodes",
        "search_seconds",
    ]
    grouped.groupby(["checkpoint", "method"])[costs].mean().to_csv(folder / "costs.csv")
    return pd.DataFrame(aggregate)


def external(raw, folder):
    iid = raw[raw.split.eq("iid")]
    fields = ["checkpoint", "selection", "method"]
    by_fold = (
        iid.groupby([*fields, "data_seed"])
        .agg(brier=("brier", "mean"), size=("rows", "first"))
        .reset_index()
    )
    by_fold.to_csv(folder / "wine-folds.csv", index=False)
    pooled = []
    for key, block in by_fold.groupby(fields):
        assert sum(block["size"]) == 178
        pooled.append(
            {
                **dict(zip(fields, key)),
                "pooled_brier": np.average(block.brier, weights=block["size"]),
                "sd_fold_brier": block.brier.std(),
            }
        )
    pd.DataFrame(pooled).to_csv(folder / "wine-pooled.csv", index=False)
    contrasts = []
    for key, block in by_fold[by_fold.checkpoint.ne("fixed")].groupby(fields[:2]):
        by_arm = block.pivot(index="data_seed", columns="method", values="brier")
        weights = block.groupby("data_seed")["size"].first()
        for a, b in CONTRASTS:
            d = by_arm[a] - by_arm[b]
            contrasts.append(
                {
                    **dict(zip(fields[:2], key)),
                    "a": a,
                    "b": b,
                    "pooled_difference": np.average(d, weights=weights),
                    "sd_fold_differences": d.std(),
                    **{f"fold{i}": f for i, f in d.items()},
                }
            )
    pd.DataFrame(contrasts).to_csv(folder / "wine-contrasts.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    folder = parser.parse_args().directory
    load = lambda f: pd.read_csv(folder / f, float_precision="round_trip")
    synthetic(
        pd.concat([load("runs.csv.gz"), load("refresh-independent-runs.csv.gz")]),
        folder,
    )
    external(
        pd.concat(
            [load("wine-runs.csv.gz"), load("wine-refresh-independent-runs.csv.gz")]
        ),
        folder,
    )

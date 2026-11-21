"""Report paired precision results over independent data replicates on a fixed suite."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

CONTRASTS = (("raw", "round3"), ("round6", "round3"), ("raw", "round6"))
METRICS = ("brier", "conditional_excess", "log_loss", "accuracy", "macro_f1", "nodes")


def interval(x):
    n = x.count()
    mean = x.mean()
    sd = x.std()
    d = t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    return {"banks": n, "mean": mean, "sd": sd, "low_95": mean - d, "high_95": mean + d}


def run(folder):
    raw = pd.read_csv(
        folder / "runs.csv.gz", float_precision="round_trip", dtype={"checkpoint": str}
    )
    keys = ["scenario", "checkpoint", "selection", "split"]
    block = (
        raw.groupby([*keys, "method", "data_seed"])[list(METRICS)].mean().reset_index()
    )
    long = block.melt(
        id_vars=[*keys, "method", "data_seed"],
        value_vars=METRICS,
        var_name="metric",
        value_name="value",
    )
    methods = []
    for key, panel in long.groupby([*keys, "method", "metric"]):
        if panel.value.notna().any():
            methods.append(
                {**dict(zip([*keys, "method", "metric"], key)), **interval(panel.value)}
            )
    pd.DataFrame(methods).to_csv(folder / "methods.csv.gz", index=False)
    pairs = []
    for key, panel in long[long.checkpoint.ne("fixed")].groupby([*keys, "metric"]):
        m = panel.pivot(index="data_seed", columns="method", values="value")
        for a, b in CONTRASTS:
            d = m[a] - m[b]
            if d.notna().any():
                pairs.append(
                    {**dict(zip([*keys, "metric"], key)), "a": a, "b": b, **interval(d)}
                )
    pd.DataFrame(pairs).to_csv(folder / "paired.csv.gz", index=False)
    macro = []
    for key, panel in long[long.checkpoint.ne("fixed")].groupby(
        ["checkpoint", "selection", "split", "metric"]
    ):
        m = panel.groupby(["data_seed", "method"]).value.mean().unstack()
        for a, b in CONTRASTS:
            d = m[a] - m[b]
            if d.notna().any():
                macro.append(
                    {
                        **dict(
                            zip(["checkpoint", "selection", "split", "metric"], key)
                        ),
                        "a": a,
                        "b": b,
                        **interval(d),
                        "bank_differences": ",".join(f"{x:.15g}" for x in d),
                    }
                )
    pd.DataFrame(macro).to_csv(folder / "suite-contrasts.csv", index=False)
    cohorts = pd.read_csv(folder / "cohorts.csv.gz", float_precision="round_trip")
    ix = ["distinct_raw", "distinct_6", "distinct_3", "distinct_compressed_3"]
    resolution = cohorts.groupby(["scenario", "method"])[ix].sum()
    resolution["fraction_raw_levels_merged_by_3"] = (
        1 - resolution.distinct_3 / resolution.distinct_raw
    )
    resolution["fraction_raw_levels_merged_by_6"] = (
        1 - resolution.distinct_6 / resolution.distinct_raw
    )
    resolution.to_csv(folder / "cohort-resolution.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    run(parser.parse_args().folder)

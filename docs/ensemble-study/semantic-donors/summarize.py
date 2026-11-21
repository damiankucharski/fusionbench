"""Summarize per-population crossover opportunities and exact finite-law performance."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t


def stats(v):
    sd = v.std()
    d = t.ppf(0.975, len(v) - 1) * sd / np.sqrt(len(v))
    return {"mean": v.mean(), "sd": sd, "low_95": v.mean() - d, "high_95": v.mean() + d}


def summarize(folder):
    f = pd.read_csv(folder / "focal.csv.gz")
    for k, a, b in (
        (
            "live_root_join",
            "root_joins_parents_free_strict_union_pairs",
            "root_joins_parents_free_strict_intersection_pairs",
        ),
        (
            "live_subtree_join",
            "subtree_joins_parents_free_strict_union_pairs",
            "subtree_joins_parents_free_strict_intersection_pairs",
        ),
        (
            "past_root_join",
            "historical_root_union_pairs",
            "historical_root_intersection_pairs",
        ),
    ):
        f[k] = f[a].gt(0) | f[b].gt(0)
    f["tournament_positive"] = f.p_tournament_new.gt(0)
    rows = []
    for (law, target, generation), block in f.groupby(["law", "target", "generation"]):
        for metric in (
            "target_present",
            "new_available",
            "tournament_positive",
            "p_tournament_new",
            "p_uniform_new",
            "live_root_join",
            "live_subtree_join",
            "past_root_join",
        ):
            banks = block.groupby("data_seed")[metric].mean()
            rows.append(
                {
                    "law": law,
                    "target": target,
                    "generation": generation,
                    "metric": metric,
                    "starts": len(block),
                    "starts_nonzero": block[metric].gt(0).sum(),
                    "banks_nonzero": banks.gt(0).sum(),
                    "minimum_start": block[metric].min(),
                    "maximum_start": block[metric].max(),
                    **stats(banks),
                }
            )
    pd.DataFrame(rows).to_csv(folder / "opportunities.csv", index=False)
    r = pd.read_csv(folder / "runs.csv.gz")
    measures = [
        "fit_brier",
        "iid_test_brier",
        "shift_test_brier",
        "iid_risk",
        "shift_risk",
        "fit_minus_family_erm",
        "iid_minus_family",
    ]
    models = []
    for key, part in r.groupby(["law", "method", "selector"]):
        for field in measures:
            bank = part.groupby("data_seed")[field].mean()
            models.append(
                {
                    **dict(zip(["law", "method", "selector"], key)),
                    "metric": field,
                    **stats(bank),
                }
            )
    pd.DataFrame(models).to_csv(folder / "model-scores.csv", index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path)
    summarize(p.parse_args().folder)

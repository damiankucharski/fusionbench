"""Exploratory transfer of fit improvements over mean across the 64 paired fits.

Positive score differences favor mean. Count each A/B orientation once, treating
absolute differences <= 1e-12 as ties. Average both orientations within bank
before averaging banks. Audit counts use conditional excess on fresh scenes.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from analyze import SEEDS, TOL, prepare

METRICS = ["fit_brier", "audit_brier", "audit_excess"]
RESULTS = Path(__file__).resolve().parent / "results"


def summarize(frame):
    frame = prepare(frame)
    keys = ["seed", "law", "checkpoint_block", "fit_block"]
    reference = frame.loc[frame.method == "mean", keys + METRICS]
    joined = frame.merge(
        reference, on=keys, validate="many_to_one", suffixes=("", "_mean")
    )
    for metric in METRICS:
        joined[metric] -= joined.pop(metric + "_mean")
    if not np.isfinite(joined[METRICS].to_numpy()).all():
        raise ValueError("Missing or nonfinite paired risk difference")

    rows = []
    for (law, allocation, arm), group in joined.groupby(
        ["law", "allocation", "method"]
    ):
        bank = group.groupby("seed")
        if bank.ngroups != len(SEEDS) or not (bank.size() == 2).all():
            raise ValueError("Expected two orientations per bank")
        counts = {}
        for name, metric in (("fit", "fit_brier"), ("audit", "audit_excess")):
            values = group[metric].to_numpy()
            counts[name] = {
                "better": values < -TOL,
                "tied": np.abs(values) <= TOL,
                "worse": values > TOL,
            }
        row = {
            "law": law,
            "allocation": allocation,
            "arm": arm,
            "n_fits": len(group),
            "n_banks": bank.ngroups,
            "n_fit_tied": counts["fit"]["tied"].sum(),
            "n_audit_tied": counts["audit"]["tied"].sum(),
        }
        for fit, fit_mask in counts["fit"].items():
            for audit, audit_mask in counts["audit"].items():
                row[f"n_fit_{fit}_audit_{audit}"] = (fit_mask & audit_mask).sum()
        paired = bank[METRICS].mean().mean()
        row.update({f"delta_{metric}": value for metric, value in paired.items()})
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    summarize(pd.read_csv(RESULTS / "scores.csv")).to_csv(
        RESULTS / "paired-mean-decisions.csv", index=False
    )

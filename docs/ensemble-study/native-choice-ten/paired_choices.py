"""Count changed decisions under paired source sizes or optimizer seeds."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "results"


def describe(r):
    frame = pd.read_csv(OUT / f"r{r:02}.csv", float_precision="round_trip")
    ref = {7: "large", 8: "seed0"}[r]
    frame = frame[frame.cohort == "standard"].set_index(["method", "family", "seed"])
    first = frame[frame.variant == ref].sort_index()
    second = frame[frame.variant == "primary"].sort_index()
    if not first.index.equals(second.index) or not first.target_hash.equals(
        second.target_hash
    ):
        raise ValueError(
            "Intervention comparisons must be paired on families and tests"
        )
    records = []
    for method, a in first.groupby(level="method"):
        b = second.xs(method, level="method", drop_level=False)
        records.append(
            {
                "method": method,
                "pairs": len(a),
                "decisions_changed": int(a.action.ne(b.action).sum()),
                "native_to_simple": int(
                    (a.action.eq("native") & b.action.isin(("mean", "top_n"))).sum()
                ),
                "simple_to_native": int(
                    (b.action.eq("native") & a.action.isin(("mean", "top_n"))).sum()
                ),
                "different_predictions": int(a.test_hash.ne(b.test_hash).sum()),
                "mean_absolute_brier_change": float(np.abs(a.brier - b.brier).mean()),
            }
        )
    return pd.DataFrame(records)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("round", type=int, choices=(7, 8))
    r = p.parse_args().round
    describe(r).to_csv(OUT / f"r{r:02}-paired-choices.csv", index=False)

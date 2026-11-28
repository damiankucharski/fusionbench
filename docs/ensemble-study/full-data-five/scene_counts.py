"""Describe event counts in the locked confirmation cohort without fitting any model."""

from pathlib import Path

import numpy as np
import pandas as pd
from banks_full import _clustered

HERE = Path(__file__).resolve().parent


def describe(seed, sample, y, groups):
    _, group_index = np.unique(groups, return_inverse=True)
    positives = np.bincount(group_index, weights=y)
    return {
        "seed": seed,
        "sample": sample,
        "rows": len(y),
        "positive_rows": int(np.sum(y)),
        "scenes": len(positives),
        "zero_positive_scenes": int(np.count_nonzero(positives == 0)),
    }


def main():
    rows = []
    base = 380000 + 1000 * 5
    for bank in _clustered(5, range(base + 300, base + 332)):
        seed, family = bank.context["seed"], bank.context["family"]
        rows.append(describe(seed, family, bank.y, bank.groups))
        if family == "scene16":
            # Both training allocations share this exact audit set.
            rows.append(describe(seed, "audit", bank.audit_y, bank.audit_groups))
    table = pd.DataFrame(rows)
    table.to_csv(HERE / "results" / "round5-scene-counts.csv", index=False)
    for sample, frame in table.groupby("sample", sort=False):
        cols = ["rows", "positive_rows", "scenes", "zero_positive_scenes"]
        print(sample)
        print(frame[cols].agg(["min", "median", "max", "sum"]).to_string())


if __name__ == "__main__":
    main()

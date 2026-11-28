"""Does the size of the source advantage predict an improvement on a fresh bank?

Descriptive analysis of the final, already-fitted choices. Membership in a bin
uses only grouped out-of-fold training evidence. No bins influence selection.
"""

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
ROUND = 10
FIELDS = ["cohort", "family", "seed", "variant"]


def classify(gain, se):
    if gain < 0:
        return "simple_trails"
    if gain < se:
        return "gain_below_one_se"
    return "gain_reaches_one_se"


def main():
    raw = pd.read_csv(OUT / "r10.csv", float_precision="round_trip")
    losses = raw.pivot(index=FIELDS, columns="method", values="brier")
    with gzip.open(OUT / "r10-source.jsonl.gz", "rt") as stream:
        traces = [json.loads(row) for row in stream]
    records = []
    for trace in traces:
        e = trace["evidence"]
        key = tuple(trace[k] for k in FIELDS)
        simple = e["best_simple"]
        actual = losses.loc[key]
        gain = e["simple_gain"]
        se = e["paired_se"]["native"][simple]
        realized = actual["native"] - actual[simple]
        records.append(
            {
                **dict(zip(FIELDS, key, strict=True)),
                "best_simple": simple,
                "bin": classify(gain, se),
                "source_gain": gain,
                "paired_se": se,
                "audit_gain": realized,
                "audit_better": realized > 1e-12,
                "audit_worse": realized < -1e-12,
            }
        )
    detail = pd.DataFrame(records)
    detail.to_csv(OUT / "r10-transfer-banks.csv", index=False)
    detail["panel"] = np.where(detail.cohort == "standard", "standard", detail.family)
    grouped = (
        detail.groupby(["panel", "bin"])
        .agg(
            banks=("seed", "size"),
            mean_source_gain=("source_gain", "mean"),
            mean_audit_gain=("audit_gain", "mean"),
            audit_better=("audit_better", "sum"),
            audit_worse=("audit_worse", "sum"),
        )
        .reset_index()
    )
    grouped.to_csv(OUT / "r10-transfer-bins.csv", index=False)


if __name__ == "__main__":
    main()

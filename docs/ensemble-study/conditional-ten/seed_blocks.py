"""Check R10 standard-panel uncertainty with joint bootstrap draws of the common seed indices."""

from pathlib import Path

import numpy as np
import pandas as pd
from analyze import committed_selection, standard_weights

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
PANELS = ("standard", "equal_cluster", "multi_site", "rule_design")


def run():
    method = committed_selection()["chosen"]
    raw = pd.read_csv(OUT / "r10.csv", float_precision="round_trip")
    published = pd.read_csv(
        OUT / "r10-contrasts.csv", float_precision="round_trip"
    ).set_index(["group", "method", "reference"], verify_integrity=True)
    raw = raw.loc[
        (raw.cohort == "standard")
        & (raw.world == "test")
        & raw.method.isin([method, "native", "lr"])
    ]
    matrix = raw.pivot(
        index=["seed", "family"], columns="method", values="brier"
    ).sort_index()
    expected_seeds = np.arange(1150400, 1150408)
    weights = dict(standard_weights())
    if (
        not np.isfinite(matrix).all().all()
        or not matrix.groupby(level="seed").size().eq(30).all()
    ):
        raise ValueError(
            "Every standard source draw needs Brier scores on all paired methods"
        )
    np.testing.assert_array_equal(matrix.index.unique("seed"), expected_seeds)
    if set(matrix.index.unique("family")) != set(weights["standard"]):
        raise ValueError("Expected all 30 standard-v6 scenarios")
    draws = np.random.default_rng(113010).integers(8, size=(10000, 8))
    records = []
    for ref in ("native", "lr"):
        diffs = (matrix[method] - matrix[ref]).unstack("family")
        for group in PANELS:
            shares = pd.Series(weights[group])
            per_seed = diffs.loc[:, shares.index].to_numpy() @ shares.to_numpy()
            lo, hi = np.quantile(per_seed[draws].mean(axis=1), [0.025, 0.975])
            # Native can be the locked action after a fallback. The published table
            # intentionally omits self-comparisons, which equal zero identically.
            expected = (
                (0.0, 0.0, 0.0)
                if method == ref
                else published.loc[
                    (group, method, ref), ["delta", "lo", "hi"]
                ].to_numpy(dtype=float)
            )
            np.testing.assert_allclose(per_seed.mean(), expected[0], rtol=0, atol=2e-14)
            records.append(
                {
                    "group": group,
                    "method": method,
                    "reference": ref,
                    "families": len(shares),
                    "seed_blocks": len(per_seed),
                    "resamples": len(draws),
                    "delta": per_seed.mean(),
                    "registered_lo": expected[1],
                    "registered_hi": expected[2],
                    "joint_seed_lo": lo,
                    "joint_seed_hi": hi,
                }
            )
    return pd.DataFrame(records)


if __name__ == "__main__":
    run().to_csv(OUT / "r10-seed-blocks.csv", index=False)

"""Paired fixed-family bootstrap and within-bank endpoint risk."""

import numpy as np
import pandas as pd


def weights(index, r):
    sizes = pd.Series(1, index=index).groupby(level="family").sum()
    if not len(sizes) or sizes.nunique() != 1 or index.to_list() != sorted(index):
        raise ValueError("Expect equal numbers of ordered seeds within fixed families")
    n_families, n = len(sizes), int(sizes.iloc[0])
    draw = np.random.default_rng(83000 + r).integers(n, size=(10_000, n_families, n))
    counts = np.stack([(draw == i).sum(axis=2) for i in range(n)], axis=2)
    return counts.reshape(10_000, -1) / len(index)


def intervals(w, d):
    lo, hi = np.quantile(w @ d.to_numpy(), [0.025, 0.975], axis=0)
    return pd.DataFrame({"delta": d.mean(), "lo": lo, "hi": hi})


def robust_panel(raw, family):
    """Take each method's worst observed endpoint separately within the same data seed."""
    rows = raw[
        (raw.cohort == "focus")
        & (raw.family == family)
        & raw.world.isin(("s0", "s100"))
    ]
    endpoints = rows.pivot(
        index=["family", "seed", "method"], columns="world", values="brier"
    )
    if set(endpoints) != {"s0", "s100"} or endpoints.isna().any().any():
        raise ValueError(f"Incomplete paired endpoints for {family}")
    return endpoints.max(axis=1).unstack("method").sort_index()

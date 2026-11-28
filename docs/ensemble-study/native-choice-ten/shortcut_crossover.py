"""Secondary follow-up to round five: where do native and mean change order?"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "results"


def compute(r):
    raw = pd.read_csv(OUT / f"r{r:02}.csv", float_precision="round_trip")
    data = raw[raw.family.isin(("shortcut_100", "shortcut_050", "shortcut_000"))]
    panel = data.pivot(index="seed", columns=["family", "method"], values="brier")
    simple = panel.xs("mean", level="method", axis=1)
    native = panel.xs("native", level="method", axis=1)
    gaps = native - simple
    err = np.max(
        np.abs((gaps.shortcut_100 + gaps.shortcut_000) / 2 - gaps.shortcut_050)
    )
    if err > 5e-12:
        raise AssertionError(
            "The midpoint is inconsistent with the two known mixture laws"
        )
    endpoints = gaps[["shortcut_000", "shortcut_100"]].to_numpy()

    def crossing(v):
        # Root of (1-s)*delta_at_zero + s*delta_at_one = 0.
        if not np.all((v[..., 0] > 0) & (v[..., 1] < 0)):
            raise ValueError("A paired resampling draw does not bracket a crossover")
        return v[..., 0] / (v[..., 0] - v[..., 1])

    ids = np.random.default_rng(91000 + r).integers(
        len(panel), size=(10_000, len(panel))
    )
    interval = np.quantile(crossing(endpoints[ids].mean(axis=1)), [0.025, 0.975])
    return {
        "source_draws": len(panel),
        "fraction_active_at_equal_brier": float(crossing(endpoints.mean(axis=0))),
        "paired_interval": interval.tolist(),
        "max_gap_midpoint_error": float(err),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("round", type=int, choices=(5, 10))
    r = parser.parse_args().round
    result = compute(r)
    if r == 10:
        discovery = compute(5)
        result["difference_from_round5"] = (
            result["fraction_active_at_equal_brier"]
            - discovery["fraction_active_at_equal_brier"]
        )
    (OUT / f"r{r:02}-crossover.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result)

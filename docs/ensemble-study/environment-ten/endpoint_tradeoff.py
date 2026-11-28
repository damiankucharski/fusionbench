"""Descriptive crossover of two fitted rules as target shortcut activity changes."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"


def main():
    name = json.loads((HERE / "selection.json").read_text())["chosen"]
    raw = pd.read_csv(OUT / "r10.csv", float_precision="round_trip")
    focus = raw[
        (raw.cohort == "focus")
        & raw.world.isin(["s0", "s25", "s50", "s75", "s100"])
        & raw.method.isin([name, "native"])
    ]
    rows = []
    for family, f in focus.groupby("family"):
        piv = f.pivot(index="seed", columns=["method", "world"], values="brier")
        delta = piv[name] - piv["native"]
        endpoints = delta[["s0", "s100"]].to_numpy()
        for i, world in enumerate(("s25", "s50", "s75"), 1):
            np.testing.assert_allclose(
                delta[world],
                (1 - i / 4) * endpoints[:, 0] + i / 4 * endpoints[:, 1],
                rtol=0,
                atol=2e-12,
            )
        lo, hi = endpoints.mean(0)
        if lo < 0 < hi:
            frac = -lo / (hi - lo)
            rng = np.random.default_rng(94010)
            draws = rng.integers(len(endpoints), size=(10000, len(endpoints)))
            dl, dh = endpoints[draws].mean(1).T
            valid = (dl < 0) & (dh > 0)
            limits = (
                np.quantile(-dl[valid] / (dh[valid] - dl[valid]), [0.025, 0.975])
                if valid.any()
                else [None, None]
            )
            good = int(valid.sum())
        else:
            frac = None
            limits = [None, None]
            good = 0
        rows.append(
            {
                "family": family,
                "method": name,
                "sources": len(endpoints),
                "delta_s0": lo,
                "delta_s100": hi,
                "active_fraction_at_equal_risk": frac,
                "ci_lo_given_crossing": limits[0],
                "ci_hi_given_crossing": limits[1],
                "bootstrap_draws_with_interior_crossing": good,
            }
        )
    pd.DataFrame(rows).to_csv(OUT / "r10-endpoint-tradeoff.csv", index=False)


if __name__ == "__main__":
    main()

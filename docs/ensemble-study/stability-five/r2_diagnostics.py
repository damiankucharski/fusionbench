"""Isolate the effect of the coefficient penalty from the probability-pool restriction."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from fusionbench.scenarios.stratified_pool import StratifiedPool

OUT = Path(__file__).resolve().parent / "results"


def main():
    cells = pd.read_csv(OUT / "round2-scores.csv")
    fits = [
        json.loads(line)
        for line in (OUT / "round2-fits.jsonl").read_text().splitlines()
    ]
    contrasts = []
    groups = list(cells[cells.cohort != "standard"].groupby("family"))
    groups.append(("standard", cells[cells.cohort == "standard"]))
    for group, frame in groups:
        wide = frame.pivot(
            index=["family", "seed"], columns="method", values="test_brier"
        ).sort_index()
        diff = wide.simplex - wide.simplex_free
        delta = diff.unstack().to_numpy()
        nf, ns = delta.shape
        lo = hi = None
        if group != "digits_449":
            indices = np.random.default_rng(89002).integers(ns, size=(10000, nf, ns))
            lo, hi = np.quantile(
                delta[np.arange(nf)[None, :, None], indices].mean((1, 2)),
                [0.025, 0.975],
            )
        contrasts.append(
            {"group": group, "penalty_minus_free": diff.mean(), "lo": lo, "hi": hi}
        )
    pd.DataFrame(contrasts).to_csv(OUT / "round2-penalty.csv", index=False)
    rows = []
    for group in ("equal", "weak", "strong"):
        records = [
            t for t in fits if t["cohort"] == "controlled" and t["family"] == group
        ]
        x, mass, q = StratifiedPool(records[0]["specialization"]).support()
        for arm in ("simplex", "simplex_free"):
            w = np.array([t["variants"][arm]["weights"] for t in records])
            pred = w @ x[:, :, 1].T
            excess = 2 * np.mean((pred - q) ** 2, axis=0) @ mass
            bias = 2 * (np.mean(pred, axis=0) - q) ** 2 @ mass
            variance = 2 * np.var(pred, axis=0) @ mass
            truth = cells[
                (cells.cohort == "controlled")
                & (cells.family == group)
                & (cells.method == arm)
            ].excess.mean()
            np.testing.assert_allclose(
                [bias + variance, excess], truth, rtol=0, atol=1e-12
            )
            rows.append(
                {
                    "group": group,
                    "method": arm,
                    "excess": excess,
                    "squared_bias": bias,
                    "fit_draw_variance": variance,
                    "mean_weight_on_provider_zero": np.mean(w[:, 0]),
                    "mean_active_providers": (w > 0).sum(axis=1).mean(),
                }
            )
    pd.DataFrame(rows).to_csv(OUT / "round2-dispersion.csv", index=False)


if __name__ == "__main__":
    main()

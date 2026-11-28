"""Decompose the round-one mean's exact excess Brier on the public scene inputs."""

from itertools import takewhile

import numpy as np
import pandas as pd
from banks_study import banks
from stats_four import CELLS, CODES, OUT, weights

from fusionbench.scenarios.grouped_rare_event import GroupedRareEvent


def geometry(b):
    p = b.audit_x[:, :, 1]
    q = b.q
    err = p - q[:, None]
    m = err.shape[1]
    single = 2 * np.square(err).mean()
    diversity = 2 * np.square(p - p.mean(1, keepdims=True)).mean()
    pair_sum = np.square(err.sum(1)) - np.square(err).sum(1)
    pair = 2 * pair_sum.mean() / (m * (m - 1))
    excess = 2 * np.square(err.mean(1)).mean()
    irreducible = 2 * (q * (1 - q)).mean()
    np.testing.assert_allclose(excess, single - diversity, atol=2e-15)
    np.testing.assert_allclose(excess, single / m + (m - 1) / m * pair, atol=2e-15)
    return {
        "family": b.context["family"],
        "seed": b.context["seed"],
        "mean_individual_excess": single,
        "diversity_discount": diversity,
        "pair_error_product": pair,
        "mean_excess": excess,
        "irreducible": irreducible,
        "expected_brier_mean": excess + irreducible,
        "specialist_excess_inside": 2
        * np.square(err[b.audit_regime, GroupedRareEvent.specialist_index]).mean(),
        "specialist_excess_outside": 2
        * np.square(err[~b.audit_regime, GroupedRareEvent.specialist_index]).mean(),
        "mean_excess_inside": 2 * np.square(err[b.audit_regime].mean(1)).mean(),
        "mean_excess_outside": 2 * np.square(err[~b.audit_regime].mean(1)).mean(),
    }


def main():
    scores = pd.read_csv(OUT / "r1-scores.csv", float_precision="round_trip")
    scores = scores.query("cohort=='factorial' and method=='mean'").set_index(
        ["family", "seed"], verify_integrity=True
    )
    data = pd.DataFrame(
        [
            geometry(b)
            for b in takewhile(lambda b: b.context["cohort"] == "factorial", banks(1))
        ]
    )
    data = data.set_index(["family", "seed"], verify_integrity=True).sort_index()
    assert len(data) == 64 and data.index.equals(scores.sort_index().index)
    np.testing.assert_allclose(
        data.expected_brier_mean, scores.sort_index().expected_brier, rtol=0, atol=2e-12
    )
    data.to_csv(OUT / "r1-geometry.csv")
    rows = []
    for k, factor in enumerate("HCS"):
        signs = (2 * np.array(CODES)[:, k] - 1) / 4
        for name in data:
            cells = data[name].unstack("family").loc[:, CELLS]
            d = cells.to_numpy() @ signs
            draws = weights(1, len(d), 1) @ d
            lo, hi = np.quantile(draws, [0.025, 0.975])
            rows.append(
                {
                    "factor": factor,
                    "quantity": name,
                    "delta": d.mean(),
                    "lo": lo,
                    "hi": hi,
                    "seeds": len(d),
                }
            )
    pd.DataFrame(rows).to_csv(OUT / "r1-geometry-effects.csv", index=False)


if __name__ == "__main__":
    main()

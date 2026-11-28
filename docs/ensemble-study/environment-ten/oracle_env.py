"""Target-law audit of information lost when a provider is omitted. Never used by a fit."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from banks_env import focus_banks

OUT = Path(__file__).resolve().parent / "results"


def conditional_mean(x, mass, q, keep):
    """Bayes probabilities conditional on a subset of observed votes on the finite support."""
    observed = x[:, keep, 1]
    _, cells = np.unique(observed, axis=0, return_inverse=True)
    cell_mass = np.bincount(cells, weights=mass)
    positive = np.bincount(cells, weights=mass * q)
    return np.divide(
        positive, cell_mass, out=np.full_like(positive, 0.5), where=cell_mass > 0
    )[cells]


def risk(probability, target):
    return float(
        2 * target.mass @ ((probability - target.q) ** 2 + target.q * (1 - target.q))
    )


def main(r):
    records = []
    for family, seed, bank, worlds in focus_banks(r):
        ids = np.arange(bank.x.shape[1])
        low = worlds["s0"]
        keep = ids[ids != bank.shortcut_index]
        robust = conditional_mean(low.x, low.mass, low.q, keep)
        np.testing.assert_allclose(robust, low.q, atol=1e-12, rtol=0)
        for name, t in worlds.items():
            np.testing.assert_array_equal(t.x, low.x)
            available = risk(t.q, t)
            safe = risk(robust, t)
            if name.startswith("s") and name != "specialist80":
                np.testing.assert_allclose(safe, risk(robust, low), atol=1e-12, rtol=0)

            def omitted(idx):
                p = conditional_mean(t.x, t.mass, t.q, ids[ids != idx])
                return risk(p, t) - available

            records.append(
                {
                    "family": family,
                    "seed": seed,
                    "world": name,
                    "full_bayes_brier": available,
                    "minimax_s0_s100_rule_brier": safe,
                    "information_cost_shortcut": omitted(bank.shortcut_index),
                    "information_cost_specialist": omitted(bank.specialist_index),
                }
            )
    frame = pd.DataFrame(records)
    frame.groupby(["family", "world"]).mean(numeric_only=True).drop(
        columns="seed"
    ).to_csv(OUT / f"r{r:02}-oracle.csv")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("round", choices=range(1, 11), type=int)
    main(p.parse_args().round)

"""Exact fresh-row full-class excess Brier of fixed predictors in round two.

The six independent provider scores are uniform on {1,...,63}/64. Enumerate
both designated input ticks A,B and average the other four analytically.
For a predictor T and law q this computes 2 E[(T-q)^2], integrating over X.
The empirical founder selected as the fit anchor is a separate random choice.
"""

import csv
from fractions import Fraction
from itertools import product
from pathlib import Path

OUTPUT = Path(__file__).with_name("results") / "r2-population-founders.csv"


def population_rows():
    ticks = range(1, 64)
    mu = Fraction(sum(ticks), len(ticks))
    variance = sum((i - mu) ** 2 for i in ticks) / len(ticks)
    # A, B, a representative of four other singletons, the all-provider
    # arithmetic mean, and the achievable two-input MIN witness.
    multiplicities = {"p_A": 1, "p_B": 1, "distractor": 4, "mean6": 1, "min_AB": 1}
    for law in ("singleton", "min"):
        squared_errors = {p: Fraction(0) for p in multiplicities}
        for a, b in product(ticks, repeat=2):
            minimum = min(a, b)
            q = a if law == "singleton" else minimum
            squared_errors["p_A"] += (a - q) ** 2
            squared_errors["p_B"] += (b - q) ** 2
            squared_errors["min_AB"] += (minimum - q) ** 2
            # E[(T-q)^2 | A,B] = (E[T | A,B]-q)^2 + Var(T | A,B).
            squared_errors["distractor"] += (mu - q) ** 2 + variance
            squared_errors["mean6"] += (
                (a + b + 4 * mu - 6 * q) ** 2 + 4 * variance
            ) / 36
        for predictor, total in squared_errors.items():
            excess = 2 * total / (len(ticks) ** 2 * 64**2)
            yield {
                "law": law,
                "predictor": predictor,
                "in_founders": predictor != "min_AB",
                "multiplicity": multiplicities[predictor],
                "excess_fraction": str(excess),
                "population_excess_brier": float(excess),
            }


if __name__ == "__main__":
    rows = list(population_rows())
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

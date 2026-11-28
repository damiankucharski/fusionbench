"""Enumerate both context counts with exact integer product-binomial weights."""

import argparse
import csv
import io
from decimal import Decimal, localcontext
from fractions import Fraction
from itertools import product
from math import comb
from pathlib import Path

SIZES = (16, 64, 256)
LAWS = {"null": (5, 5), "strong": (1, 9), "overshoot": (4, 6), "boundary": (3, 7)}
# Probabilities in tenths. The anchor wins all score-zero ties.
PROGRAMS = [(5, 5)] + [p for p in product((1, 5, 9), repeat=2) if p != (5, 5)]
RESULTS = Path(__file__).with_name("results.csv")


def count_weights(m, q):
    """Unnormalized binomial probabilities with common denominator 10**m."""
    return [comb(m, k) * q**k * (10 - q)**(m - k) for k in range(m + 1)]


def policy_table(m):
    """Return selected program indices for every pair of label counts."""
    with localcontext() as ctx:
        ctx.prec = 64
        # The empirical full-Brier gain has integer numerator in units 1/(50*n).
        # The range of this numerator on a single row is 20*abs(p-5), p in tenths.
        charges = []
        base = []
        slopes = []
        for p in PROGRAMS:
            d1, d2 = p[0] - 5, p[1] - 5
            squared = d1**2 + d2**2
            charges.append((Decimal(200 * m * squared) * Decimal(160).ln()).sqrt())
            base.append(-m * (10 * (d1 + d2) + squared))
            slopes.append((20 * d1, 20 * d2))

        decisions = {}
        for k1, k2 in product(range(m + 1), repeat=2):
            gains = [intercept + b1 * k1 + b2 * k2
                     for intercept, (b1, b2) in zip(base, slopes, strict=True)]
            lower = [g - charge for g, charge in zip(gains, charges, strict=True)]
            decisions[k1, k2] = (max(range(len(gains)), key=gains.__getitem__),
                                 max(range(len(lower)), key=lower.__getitem__))
        return decisions


def true_gain(p, law):
    """Full Brier improvement, averaged over two equally frequent contexts."""
    def expected_loss(v, q):
        return Fraction(q * (10 - v)**2 + (10 - q) * v**2, 500)

    return sum(expected_loss(5, q) - expected_loss(v, q) for v, q in zip(p, law, strict=True)) / 2


def summarize(m, law, decisions):
    weights1, weights2 = (count_weights(m, q) for q in law)
    mass = [[0] * len(PROGRAMS) for _ in range(2)]
    for (k1, k2), (raw, certified) in decisions.items():
        mass[0][raw] += weights1[k1] * weights2[k2]
        mass[1][certified] += weights1[k1] * weights2[k2]
    denominator = 10 ** (2 * m)
    gains = [true_gain(p, law) for p in PROGRAMS]
    result = []
    for counts in mass:
        assert sum(counts) == denominator
        probabilities = [Fraction(w, denominator) for w in counts]
        expected_gain = sum(p * g for p, g in zip(probabilities, gains, strict=True))
        positive = sum(p for p, g in zip(probabilities, gains, strict=True) if g > 0)
        harmful = sum(p for p, g in zip(probabilities, gains, strict=True) if g < 0)
        wrong = 1 - probabilities[0] - positive
        result.append((expected_gain, wrong, harmful, positive))
    # Independent check. On this Cartesian grid raw Brier chooses a non-anchor
    # in a context iff its empirical positive rate falls outside [0.3, 0.7].
    stay = [sum(w for k, w in enumerate(ws) if 3*m <= 10*k <= 7*m) for ws in (weights1, weights2)]
    assert mass[0][0] == stay[0] * stay[1]
    assert result[1][1] <= Fraction(1, 20)
    return result


def render():
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(("n", "law", "q1", "q2", "rule", "expected_true_gain", "p_wrong", "p_harmful", "p_beneficial"))
    for n in SIZES:
        m = n // 2
        decisions = policy_table(m)
        for name, law in LAWS.items():
            for rule, values in zip(("raw", "certified"), summarize(m, law, decisions), strict=True):
                writer.writerow((n, name, *(q / 10 for q in law), rule, *(format(float(v), ".12g") for v in values)))
    return out.getvalue()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="write results.csv (the default verifies its contents)")
    args = parser.parse_args()
    table = render()
    if args.write:
        RESULTS.write_text(table, encoding="utf-8")
    elif RESULTS.read_text(encoding="utf-8") != table:
        raise SystemExit("results.csv differs from the enumerated table")
    print(table, end="")

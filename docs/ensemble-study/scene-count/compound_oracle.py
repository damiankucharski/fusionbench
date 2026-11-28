"""Exact full-class Brier on the four endpoints of the cloned-shortcut law.

Enumerate label and independent votes, constructing cloned and reversed reports
before conditioning. The target posterior knows the deployment law. No model
is fitted and no target observations are used to choose a rule.
"""

import csv
from fractions import Fraction as Q
from itertools import product
from math import prod
from pathlib import Path

import numpy as np

from fusionbench.scenarios.cloned_shortcut import ClonedShortcut

STABLE, SHORTCUT = Q(3, 4), Q(4, 5)
STATES = (0, 1)


def joint(clone, reverse):
    """Return exact P(observed votes, y) on the positive-probability support."""
    rates = (STABLE,) * 2 + (SHORTCUT,) * (1 if clone else 6)
    masses = {}
    for votes in product(STATES, repeat=len(rates)):
        shortcuts = votes[2:] * (6 if clone else 1)
        observed = votes[:2] + tuple(v ^ reverse for v in shortcuts)
        masses[observed] = tuple(
            Q(1, 2)
            * prod(p if v == y else 1 - p for v, p in zip(votes, rates, strict=True))
            for y in STATES
        )
    assert sum(map(sum, masses.values())) == 1
    return masses


def evaluate(clone, reverse):
    source, target = joint(clone, 0), joint(clone, reverse)
    assert source.keys() == target.keys()
    votes = sorted(target)
    rates = (STABLE,) * 2 + (SHORTCUT,) * 6
    reports = [
        [p if v else 1 - p for v, p in zip(row, rates, strict=True)] for row in votes
    ]
    mean = [sum(row) / 8 for row in reports]
    positive = np.asarray(reports, dtype=float)
    predictions = np.stack((1 - positive, positive), axis=2)
    scenario = ClonedShortcut(
        clone_probability=clone, test_reversal_probability=reverse
    )
    posteriors = []
    for split, law in (("fit", source), ("test", target)):
        q = [law[v][1] / sum(law[v]) for v in votes]
        public = scenario.bayes_oracle(predictions, {"split": split})[:, 1]
        np.testing.assert_allclose(
            public, np.asarray(q, dtype=float), rtol=0, atol=1e-14
        )
        posteriors.append(q)

    def risk(probabilities):
        return sum(
            2 * target[v][y] * (p - y) ** 2
            for v, p in zip(votes, probabilities, strict=True)
            for y in STATES
        )

    source_brier, target_brier = map(risk, posteriors)
    return {
        "clone": clone,
        "reverse": reverse,
        "supported_votes": len(votes),
        "test_mean_brier": float(risk(mean)),
        "test_source_posterior_brier": float(source_brier),
        "test_target_posterior_brier": float(target_brier),
        "source_minus_target_posterior_brier": float(source_brier - target_brier),
    }


if __name__ == "__main__":
    rows = [evaluate(c, r) for c, r in product(STATES, repeat=2)]
    output = Path(__file__).with_name("results") / "compound-oracle.csv"
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

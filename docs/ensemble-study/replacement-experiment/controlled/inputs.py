"""Small complete-support coupled probability laws for replacement search."""

from dataclasses import dataclass
from itertools import product

import numpy as np

POOLS = ("null", "signal")
POLICIES = ("ordinary", "open_raw", "open_kl", "strict_kl")
SEED_BASE = 4_782_000
SEARCH_BASE = 4_783_000


def probabilities(bits):
    pos = (1 + 6 * np.asarray(bits, dtype=float)) / 8
    return np.stack((1 - pos, pos), axis=-1)


SUPPORT = probabilities(list(product((0, 1), repeat=3)))
INDEX_WEIGHTS = np.array([4, 2, 1])


def support_targets(bank):
    a, b, c = np.random.default_rng([SEED_BASE + bank, 17]).permutation(3)
    p = SUPPORT[:, :, 1]
    ab = np.maximum(p[:, a], p[:, b])
    structure = {
        "max_ab": ab,
        "min_ac": np.minimum(p[:, a], p[:, c]),
        "min_bc": np.minimum(p[:, b], p[:, c]),
    }
    return (
        {"null": p.mean(axis=1), "signal": np.minimum(ab, p[:, c])},
        structure,
        (int(a), int(b), int(c)),
    )


@dataclass
class Sample:
    predictions: np.ndarray
    indices: np.ndarray
    labels: dict


def generate(bank, stream, size):
    rng = np.random.default_rng([SEED_BASE + bank, stream])
    bits = rng.integers(0, 2, size=(size, 3))
    u = rng.random(size)
    indices = bits @ INDEX_WEIGHTS
    targets, _, _ = support_targets(bank)
    return Sample(
        SUPPORT[indices],
        indices,
        {law: (u < q[indices]).astype(np.int64) for law, q in targets.items()},
    )


def full_brier(pos, y):
    return 2 * np.square(np.asarray(pos) - y)


def score(pos_on_support, q, test, law):
    p = pos_on_support[test.indices]
    y = test.labels[law]
    excess = full_brier(pos_on_support, q).mean()
    bayes = np.mean(2 * q * (1 - q))
    return {
        "exact_excess": float(excess),
        "exact_brier": float(excess + bayes),
        "test_brier": float(full_brier(p, y).mean()),
        # Binary class-0 wins a tie for a canonical [1-p,p] probability row.
        "test_accuracy": float(np.mean((p > 0.5) == y)),
        "exact_accuracy": float(np.where(pos_on_support > 0.5, q, 1 - q).mean()),
    }

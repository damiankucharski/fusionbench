"""Label-efficient auditing of the finite-pool Brier difference of two fixed predictions."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from scipy.stats import beta


def _probabilities(f, g):
    p, q = np.asarray(f, dtype=float), np.asarray(g, dtype=float)
    if p.ndim != 1 or q.shape != p.shape or not len(p):
        raise ValueError(
            "Expected two nonempty equal-length vectors of binary probabilities"
        )
    if not all(
        np.isfinite(a).all() and (a >= 0).all() and (a <= 1).all() for a in (p, q)
    ):
        raise ValueError("Probabilities must be finite and lie in [0,1]")
    return p, q


def _labels(y, n):
    labels = np.asarray(y)
    if labels.shape != (n,) or not np.isin(labels, (0, 1)).all():
        raise ValueError("Expected one binary label for each requested row")
    return labels.astype(float, copy=False)


@dataclass(frozen=True)
class PairAudit:
    """A label-blind sampling plan. Repeated records retain their sampling multiplicity."""

    indices: np.ndarray
    signed_weights: np.ndarray
    center: float
    mean_gap: float
    proposal: str

    def estimate(self, labels):
        y = _labels(labels, len(self.indices))
        if not len(y):
            return self.center
        return float(self.center + self.signed_weights @ (1 - 2 * y) / len(y))

    def exact_interval(self, labels, confidence=0.95):
        """Fixed-budget binomial interval for a gap-proportional, with-replacement audit."""
        if self.proposal != "gap":
            raise ValueError(
                "The exact sign interval requires gap-proportional sampling"
            )
        if not 0 < confidence < 1:
            raise ValueError("Confidence must lie strictly between zero and one")
        y = _labels(labels, len(self.indices))
        if self.mean_gap == 0:
            return 0.0, 0.0
        n = len(y)
        successes = int(np.count_nonzero(self.signed_weights * (1 - 2 * y) > 0))
        alpha = 1 - confidence
        lo = 0 if successes == 0 else beta.ppf(alpha / 2, successes, n - successes + 1)
        hi = (
            1
            if successes == n
            else beta.ppf(1 - alpha / 2, successes + 1, n - successes)
        )
        return tuple(float(self.center + self.mean_gap * (2 * v - 1)) for v in (lo, hi))


def draw_pair_audit(f, g, draws, *, rng, proposal="gap"):
    """Plan draws before seeing labels to estimate mean((f-y)^2 - (g-y)^2)."""
    p, q = _probabilities(f, g)
    if isinstance(draws, bool) or not isinstance(draws, Integral) or draws < 1:
        raise ValueError("Draw count must be a positive integer")
    if proposal not in {"gap", "uniform"}:
        raise ValueError("Expected 'gap' or 'uniform'")
    delta = p - q
    magnitude = np.abs(delta)
    mean_gap = float(magnitude.mean())
    center = float(np.mean(delta * (p + q - 1)))
    if mean_gap == 0:
        return PairAudit(
            np.array([], dtype=int),
            np.array([], dtype=float),
            center,
            mean_gap,
            proposal,
        )
    if proposal == "uniform":
        indices = rng.integers(len(p), size=draws)
        weights = delta[indices]
    else:
        indices = rng.choice(
            len(p), size=draws, replace=True, p=magnitude / magnitude.sum()
        )
        weights = np.sign(delta[indices]) * mean_gap
    return PairAudit(indices, weights, center, mean_gap, proposal)


def deterministic_gap_requests(f, g, unique_budget):
    """Return the nonzero-gap rows giving the smallest worst-case interval width."""
    p, q = _probabilities(f, g)
    if (
        isinstance(unique_budget, bool)
        or not isinstance(unique_budget, Integral)
        or unique_budget < 0
    ):
        raise ValueError("Unique label budget must be a nonnegative integer")
    gap = np.abs(p - q)
    order = np.argsort(-gap, kind="stable")
    return order[gap[order] > 0][:unique_budget]


def sharp_pair_bounds(f, g, seen, labels):
    """Exact attainable endpoints as yet-unresolved binary labels vary arbitrarily."""
    p, q = _probabilities(f, g)
    indices = np.asarray(seen)
    if indices.ndim != 1 or (indices.size and indices.dtype.kind not in "iu"):
        raise ValueError("Expected a vector of unique row indices")
    indices = indices.astype(int, copy=False)
    if (
        len(set(indices.tolist())) != len(indices)
        or (indices < 0).any()
        or (indices >= len(p)).any()
    ):
        raise ValueError("Invalid or repeated row indices")
    y = _labels(labels, len(indices))
    delta = p - q
    open_rows = np.ones(len(p), dtype=bool)
    open_rows[indices] = False
    center = float(np.mean(delta * (p + q - 1)) + delta[indices] @ (1 - 2 * y) / len(p))
    radius = float(np.abs(delta[open_rows]).sum() / len(p))
    return center - radius, center + radius

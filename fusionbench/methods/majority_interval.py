"""Identify the interval of rotating third votes consistent with binary majorities."""

from dataclasses import dataclass
from numbers import Real

import numpy as np


def _vote_data(x):
    raw = np.asarray(x)
    if raw.ndim == 1 and raw.size == 0:
        raw = raw.reshape(0, 4)
    if raw.ndim != 2 or raw.shape[1] != 4 or raw.dtype.kind not in "biuf":
        raise ValueError("x must be a finite real matrix with four columns")
    raw = raw.astype(np.float64, copy=False)
    if not np.isfinite(raw).all():
        raise ValueError("x must be finite")
    shared = (raw[:, :2] >= 0).sum(axis=1)
    v, w = raw[:, 2], raw[:, 3]
    # On [0, pi/2) the rotating vote has the sign of v + w*tan(theta).
    # atan2 avoids overflow and cancellation in the ratio -v/w.
    crossing = np.arctan2(np.where(w > 0, -v, v), np.abs(w))
    return shared, v, w, crossing


def _hard_rule(shared, v, w, offset):
    third = np.where(w == 0, v >= 0, np.where(w > 0, offset >= 0, offset <= 0))
    return np.where(shared == 1, third, shared == 2)


@dataclass(frozen=True)
class MajorityInterval:
    """Closure of feasible angles in radians, with exact endpoint inclusion flags."""

    lower: float
    upper: float
    lower_included: bool
    upper_included: bool

    def predict_midpoint(self, x):
        """Binary majority at the midpoint, with a zero projection voting positive."""
        shared, v, w, crossing = _vote_data(x)
        # Twice the angle offset locates the mathematical midpoint even if the
        # floating-point midpoint would round to an excluded endpoint.
        offset = (self.upper - crossing) - (crossing - self.lower)
        return _hard_rule(shared, v, w, offset)

    def predict_minimax_positive(self, x):
        """Equal-weight average of the hard rules at the two closure endpoints.

        Both endpoints use the positive-on-zero convention, including an
        endpoint that a strict training constraint excludes.
        """
        shared, v, w, crossing = _vote_data(x)
        low = _hard_rule(shared, v, w, self.lower - crossing)
        high = _hard_rule(shared, v, w, self.upper - crossing)
        return (low.astype(float) + high) * 0.5

    def predict_uniform_positive(self, x):
        """Probability for an angle uniform on the interval, or fixed at a singleton."""
        shared, v, w, crossing = _vote_data(x)
        if self.lower == self.upper:
            return _hard_rule(shared, v, w, self.lower - crossing).astype(float)
        transition = np.clip(crossing, self.lower, self.upper)
        positive_length = np.where(
            w > 0, self.upper - transition, transition - self.lower
        )
        third = np.where(w == 0, v >= 0, positive_length / (self.upper - self.lower))
        return np.where(shared == 1, third, shared == 2)


def fit_majority_interval(x, y, angle_max):
    """Infer all consistent angles in [0, angle_max] from final binary labels.

    The four columns give the two shared scores followed by v and w; the third
    score is v*cos(theta) + w*sin(theta). A zero score votes positive. Negative
    labels at a rotating boundary give strict constraints. Contradictions,
    including a singleton excluded by a strict constraint, raise ValueError.
    With an empty sample, the entire input angle interval remains feasible.
    """
    if (
        isinstance(angle_max, bool)
        or not isinstance(angle_max, Real)
        or not 0 < angle_max < np.pi / 2
    ):
        raise ValueError("angle_max must be in (0, pi/2)")
    shared, v, w, crossing = _vote_data(x)
    labels = np.asarray(y)
    if (
        labels.shape != shared.shape
        or labels.dtype.kind not in "biuf"
        or not np.isin(labels, (0, 1)).all()
    ):
        raise ValueError("y must contain one binary label per row")
    positive = labels.astype(bool)
    split = shared == 1
    if np.any(~split & (positive != (shared == 2))):
        raise ValueError("label contradicts the shared majority")
    if np.any(split & (w == 0) & (positive != (v >= 0))):
        raise ValueError("label contradicts the constant third vote")

    active = split & (w != 0)
    constrains_lower = active & ((w > 0) == positive)
    constrains_upper = active & ~constrains_lower
    lower = float(np.max(crossing[constrains_lower], initial=0.0))
    upper = float(np.min(crossing[constrains_upper], initial=float(angle_max)))
    strict = active & ~positive
    lower_included = not np.any(strict & (crossing == lower))
    upper_included = not np.any(strict & (crossing == upper))
    if lower > upper or (lower == upper and not (lower_included and upper_included)):
        raise ValueError("training labels have no common feasible angle")
    return MajorityInterval(lower, upper, lower_included, upper_included)

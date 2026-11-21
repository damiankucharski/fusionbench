"""Use a dominant covariance direction to recover three-sign majority axes."""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import null_space

from fusionbench.methods.majority_moments import (
    _labeled_inputs,
    fit_angular_majority_moments,
)


@dataclass(frozen=True)
class SpikedMajority:
    axes: np.ndarray
    route: str
    agreement: float
    eigenratio: float
    contradictions: int


def _anchor_axes(x, signed, anchor, mismatch):
    if not np.any(mismatch):
        raise ValueError(
            "anchor route needs contradictions with the oriented leading direction"
        )
    coordinates = null_space(anchor[None, :])
    residual = x[mismatch] @ coordinates
    bisector = np.mean(signed[mismatch, None] * residual, axis=0)
    length = np.linalg.norm(bisector)
    tol = np.finfo(float).eps * max(residual.shape) * np.max(np.abs(residual))
    if length <= tol:
        raise ValueError("contradictions do not identify a signed residual mean")
    bisector /= length

    # When the anchor contradicts the majority, the other two votes agree.
    # After removing their sum direction, their difference is the direction
    # with smallest conditional variance. Its sign only exchanges the axes.
    complement = null_space(bisector[None, :])
    projected = residual @ complement
    projected -= projected.mean(axis=0)
    covariance = projected.T @ projected / len(projected)
    values, vectors = np.linalg.eigh(covariance)
    if values[1] - values[0] <= np.finfo(float).eps * len(values) * values[-1]:
        raise ValueError(
            "contradictions do not identify a unique minimum-variance residual direction"
        )
    difference = coordinates @ (complement @ vectors[:, 0])
    common = coordinates @ bisector
    return np.stack(
        (anchor, (common + difference) / np.sqrt(2), (common - difference) / np.sqrt(2))
    )


def fit_spiked_majority(x, y, force=None) -> SpikedMajority:
    """Estimate three oriented axes from raw rows and a final binary task label.

    The spike is the leading direction of the uncentered second moment. For
    a leading-to-second eigenvalue ratio below 2, fit angular moments as usual.
    Otherwise a sign agreement of at least 0.625 with labels selects the anchor
    route; lower agreement drops that direction and fits residual angular
    moments. An explicit force selects one of these routes for comparisons.

    The anchor interpretation assumes that the oriented spike is a true
    concept axis and that the conditional residual moments distinguish the
    other two. Contradictions counts training rows whose majority label differs
    from the vote along the oriented spike. A zero projection votes positive.
    """
    if force not in (None, "angular", "anchor", "drop"):
        raise ValueError("force must be one of angular, anchor, drop, or None")
    x, signed = _labeled_inputs(x, y)
    if x.shape[1] < 4:
        raise ValueError("expected at least four input dimensions")
    largest = np.max(np.abs(x))
    if largest == 0:
        raise ValueError("zero inputs have no leading covariance direction")
    # A single positive factor leaves eigenvectors and eigenvalue ratios unchanged.
    scaled = x / largest
    moment = scaled.T @ scaled / len(scaled)
    values, vectors = np.linalg.eigh(moment)
    if values[-2] <= np.finfo(float).eps * len(values) * values[-1]:
        raise ValueError("at least two nondegenerate input directions are required")
    eigenratio = float(values[-1] / values[-2])
    anchor = vectors[:, -1]
    projections = scaled @ anchor
    mismatch = (projections >= 0) != (signed > 0)
    reverse_mismatch = (-projections >= 0) != (signed > 0)
    if reverse_mismatch.sum() < mismatch.sum():
        anchor = -anchor
        mismatch = reverse_mismatch
    contradictions = int(mismatch.sum())
    agreement = (len(x) - contradictions) / len(x)

    route = force
    if route is None:
        if eigenratio < 2:
            route = "angular"
        else:
            route = "anchor" if agreement >= 0.625 else "drop"
    if route == "angular":
        axes = fit_angular_majority_moments(x, y).axes
    elif route == "anchor":
        axes = _anchor_axes(scaled, signed, anchor, mismatch)
    else:
        coordinates = null_space(anchor[None, :])
        axes = fit_angular_majority_moments(x @ coordinates, y).axes @ coordinates.T
    return SpikedMajority(axes, route, agreement, eigenratio, contradictions)

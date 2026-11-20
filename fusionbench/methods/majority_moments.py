"""Recover the axes of a three-sign majority under standard Gaussian inputs."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MajorityMoments:
    """Estimated majority directions, specified by their shared axis and planar phase."""

    sum_direction: np.ndarray
    plane: np.ndarray
    phase: float

    @property
    def axes(self) -> np.ndarray:
        """Three oriented orthonormal rows, identified up to a permutation."""
        return self.with_phase_offset(0.0)

    def with_phase_offset(self, radians: float) -> np.ndarray:
        """Rotate the three axes together about their shared direction."""
        radians = _finite_array(radians, "radians")
        if radians.ndim != 0:
            raise ValueError("radians must be a finite real scalar")
        angles = self.phase + float(radians) + 2 * np.pi * np.arange(3) / 3
        planar = np.column_stack((np.cos(angles), np.sin(angles))) @ self.plane
        return self.sum_direction / np.sqrt(3.0) + np.sqrt(2.0 / 3.0) * planar


def _finite_array(value, name):
    array = np.asarray(value)
    if array.dtype.kind not in "biuf" or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite real values")
    return array.astype(np.float64, copy=False)


def majority_axes_from_moments(first, third) -> MajorityMoments:
    """Recover majority directions from signed first and symmetric third Hermite moments.

    For standard normal x and y the signed majority of three orthonormal
    projections, the first moment specifies their sum. Perpendicular to that
    sum, the left singular vectors of the flattened third moment span a plane.
    Its third angular harmonic has three equal minima pointing toward the
    projections of the three original axes.
    """
    first = _finite_array(first, "first")
    third = _finite_array(third, "third")
    if first.ndim != 1 or first.size < 3 or third.shape != (first.size,) * 3:
        raise ValueError("expected moments of shape (d,) and (d, d, d) with d >= 3")
    if not (
        np.allclose(third, third.swapaxes(0, 1))
        and np.allclose(third, third.swapaxes(1, 2))
    ):
        raise ValueError("third must be a symmetric tensor")
    first_norm = np.linalg.norm(first)
    if first_norm == 0 or not np.isfinite(first_norm):
        raise ValueError("first moment has no finite, nonzero direction")
    direction = first / first_norm
    flat = third.reshape(first.size, -1)
    projected = flat - direction[:, None] * (direction @ flat)
    left, singular, _ = np.linalg.svd(projected, full_matrices=False)
    tolerance = np.finfo(np.float64).eps * max(flat.shape) * np.linalg.norm(flat)
    if singular[1] <= tolerance:
        raise ValueError(
            "third moment does not identify two independent planar directions"
        )

    # Reproject and orthonormalize to suppress SVD roundoff when the signal is small.
    basis = left[:, :2]
    basis, _ = np.linalg.qr(basis - direction[:, None] * (direction @ basis))
    p, q = basis.T

    def contraction(a, b, c):
        return np.einsum("ijk,i,j,k->", third, a, b, c, optimize=True)

    cosine = (contraction(p, p, p) - 3 * contraction(p, q, q)) / 4
    sine = (3 * contraction(p, p, q) - contraction(q, q, q)) / 4
    if np.hypot(cosine, sine) <= tolerance:
        raise ValueError("third angular harmonic has no identifiable phase")
    phase = (np.arctan2(sine, cosine) + np.pi) / 3
    return MajorityMoments(direction, basis.T, float(phase))


def fit_majority_moments(x, y) -> MajorityMoments:
    """Estimate three majority axes using raw standard-normal inputs and 0/1 task labels.

    Inputs are assumed to share the specified zero mean and identity covariance.
    The estimator directly averages y_signed * x and y_signed * H3(x) using
    y_signed = 2*y - 1. It reads neither local concept labels nor expert predictions.
    """
    x = _finite_array(x, "x")
    labels = _finite_array(y, "y")
    if x.ndim != 2 or min(x.shape) < 3:
        raise ValueError("expected x with shape (n, d), n >= 3 and d >= 3")
    if labels.shape != (len(x),) or not np.isin(labels, (0, 1)).all():
        raise ValueError("expected one binary 0/1 task label per row")
    signed = 2 * labels - 1
    first = (signed @ x) / len(x)
    third = np.einsum("n,ni,nj,nk->ijk", signed, x, x, x, optimize=True) / len(x)
    first_identity = first[:, None, None] * np.eye(x.shape[1])
    third -= (
        first_identity
        + first_identity.transpose(1, 0, 2)
        + first_identity.transpose(1, 2, 0)
    )
    return majority_axes_from_moments(first, third)

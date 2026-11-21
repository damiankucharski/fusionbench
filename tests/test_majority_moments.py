import itertools

import numpy as np
import pytest

from fusionbench.methods.majority_moments import (
    fit_angular_majority_moments,
    fit_majority_moments,
    majority_axes_from_moments,
)


def test_population_moments_recover_rotated_oriented_axes_and_phase():
    rng = np.random.default_rng(82)
    axes = np.linalg.qr(rng.normal(size=(9, 3)))[0].T
    c = np.sqrt(2 / np.pi)
    first = (c / 2) * axes.sum(axis=0)
    third = np.zeros((9, 9, 9))
    for axis in axes:
        third -= (c / 2) * np.einsum("i,j,k->ijk", axis, axis, axis)
    for a, b, d in itertools.permutations(axes):
        third -= (c**3 / 2) * np.einsum("i,j,k->ijk", a, b, d)
    for sign in [-1, 1]:
        estimator = majority_axes_from_moments(sign * first, sign * third)
        match = estimator.axes @ (sign * axes.T)
        np.testing.assert_allclose(
            np.sort(match, axis=0), [[0] * 3, [0] * 3, [1] * 3], atol=1e-12
        )
        turned = estimator.with_phase_offset(np.pi / 3)
        np.testing.assert_allclose(turned @ turned.T, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(
            turned.sum(axis=0), estimator.axes.sum(axis=0), atol=1e-12
        )
        np.testing.assert_allclose(
            np.max(turned @ estimator.axes.T, axis=1), 2.0 / 3.0, atol=1e-12
        )


def test_fit_uses_unadjusted_gaussian_hermite_moments_and_signed_binary_labels():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(64, 5)) + 0.3
    labels = rng.integers(0, 2, size=len(x), dtype=np.uint8)
    first = np.zeros(5)
    third = np.zeros((5, 5, 5))
    identity = np.eye(5)
    for row, label in zip(x, labels, strict=True):
        signed = 2 * int(label) - 1
        first += signed * row
        hermite = np.einsum("i,j,k->ijk", row, row, row)
        for pattern in ["i,jk->ijk", "j,ik->ijk", "k,ij->ijk"]:
            hermite -= np.einsum(pattern, row, identity)
        third += signed * hermite
    expected = majority_axes_from_moments(first / len(x), third / len(x))
    fitted = fit_majority_moments(x, labels)
    # The decomposition may choose a different basis phase or order.
    np.testing.assert_allclose(
        np.max(expected.axes @ fitted.axes.T, axis=1), 1.0, atol=1e-12
    )
    with pytest.raises(ValueError, match="binary"):
        fit_majority_moments(x, 2 * labels.astype(float) - 1)


def test_degeneracy_and_invalid_input_are_reported():
    with pytest.raises(ValueError, match="nonzero"):
        majority_axes_from_moments(np.zeros(3), np.zeros((3, 3, 3)))
    with pytest.raises(ValueError, match="two independent"):
        majority_axes_from_moments(np.ones(3), np.zeros((3, 3, 3)))
    with pytest.raises(ValueError, match="finite"):
        fit_majority_moments(np.full((3, 3), np.nan), [0, 0, 1])
    with pytest.raises(ValueError, match="shape"):
        fit_majority_moments(np.ones((2, 3)), [0, 1])
    with pytest.raises(ValueError, match="symmetric"):
        third = np.zeros((3, 3, 3))
        third[0, 1, 2] = 1
        majority_axes_from_moments(np.ones(3), third)
    # y-weighted Hermite tensor proportional to sym(e ⊗ I) has zero planar phase.
    tensor = np.einsum("i,jk->ijk", [1, 0, 0], np.eye(3))
    with pytest.raises(ValueError, match="angular harmonic"):
        majority_axes_from_moments(
            [1, 0, 0], tensor + tensor.swapaxes(0, 1) + tensor.transpose(1, 2, 0)
        )


def test_angular_fit_invariant_to_arbitrary_row_radii_even_at_extreme_scales():
    rng = np.random.default_rng(19)
    x = rng.normal(size=(144, 5))
    labels = np.sum(x[:, :3] > 0, axis=1) >= 2
    reference = fit_angular_majority_moments(x, labels)
    # Radius is a deterministic function of direction and ranges from 1e-200 to 1e200.
    radii = np.where(x[:, 0] > 0, 1e200, 1e-200)
    actual = fit_angular_majority_moments(radii[:, None] * x, labels)
    # The three axes are only identified up to a permutation.
    match = np.argmax(actual.axes @ reference.axes.T, axis=1)
    np.testing.assert_allclose(actual.axes, reference.axes[match], atol=1e-12)
    with pytest.raises(ValueError, match="nonzero direction"):
        fit_angular_majority_moments(np.vstack((x, np.zeros(5))), np.r_[labels, False])


def test_angular_identity_weight_against_explicit_row_moments():
    rng = np.random.default_rng(34)
    x = rng.normal(size=(128, 6))
    labels = np.sum(x[:, :3] > 0, axis=1) >= 2
    fitted = fit_angular_majority_moments(x, labels)
    dimension = x.shape[1]
    h = np.sqrt(dimension) * x / np.linalg.norm(x, axis=1, keepdims=True)
    signed = 2 * labels.astype(float) - 1
    first = np.average(signed[:, None] * h, axis=0)
    third = np.zeros((dimension,) * 3)
    for row, sign in zip(h, signed, strict=True):
        term = np.einsum("i,j,k->ijk", row, row, row)
        for pattern in ["i,jk->ijk", "j,ik->ijk", "k,ij->ijk"]:
            term -= (dimension / (dimension + 1)) * np.einsum(
                pattern, row, np.eye(dimension)
            )
        third += sign * term / len(h)
    expected = majority_axes_from_moments(first, third)
    match = np.argmax(fitted.axes @ expected.axes.T, axis=1)
    np.testing.assert_allclose(fitted.axes, expected.axes[match], atol=1e-12)

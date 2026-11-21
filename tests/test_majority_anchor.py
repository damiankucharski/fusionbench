from itertools import product

import numpy as np
import pytest

from fusionbench.methods.majority_anchor import fit_spiked_majority
from fusionbench.methods.majority_moments import fit_angular_majority_moments


@pytest.fixture
def corners():
    x = np.array(list(product((-1.0, 1.0), repeat=4)))
    y = np.sum(x[:, :3] > 0, axis=1) >= 2
    return x, y


def test_informative_spike_recovers_true_axes_including_after_rotation(corners):
    x, y = corners
    x = x * [3, 1, 1, 1]
    rotation = np.linalg.qr(np.random.default_rng(11).normal(size=(4, 4)))[0]
    fit = fit_spiked_majority(x @ rotation, y)
    assert fit.route == "anchor"
    assert fit.contradictions == 4
    assert fit.agreement == 0.75
    assert fit.eigenratio == pytest.approx(9.0)
    scores = fit.axes @ rotation[:3].T
    np.testing.assert_allclose(
        np.sort(scores, axis=0), [[0] * 3, [0] * 3, [1] * 3], atol=1e-12
    )


def test_uninformative_spike_is_dropped(corners):
    x, y = corners
    x = x * [1, 1, 1, 3]
    fit = fit_spiked_majority(x, y)
    assert fit.route == "drop"
    assert fit.contradictions == 8
    assert fit.agreement == 0.5
    np.testing.assert_allclose(fit.axes[:, 3], 0, atol=1e-12)
    np.testing.assert_allclose(fit.axes @ fit.axes.T, np.eye(3), atol=1e-12)


def test_no_spike_and_route_override(corners):
    x, y = corners
    fit = fit_spiked_majority(x, y)
    assert fit.route == "angular"
    np.testing.assert_allclose(fit.axes, fit_angular_majority_moments(x, y).axes)
    spiked = x * [1, 1, 1, 3]
    forced = fit_spiked_majority(spiked, y, force="angular")
    assert forced.route == "angular"
    np.testing.assert_allclose(
        forced.axes, fit_angular_majority_moments(spiked, y).axes
    )


def test_shape_label_and_degeneracy_errors(corners):
    x, y = corners
    with pytest.raises(ValueError, match="four input dimensions"):
        fit_spiked_majority(x[:, :3], y)
    with pytest.raises(ValueError, match="binary"):
        fit_spiked_majority(x, np.full(len(x), 2))
    with pytest.raises(ValueError, match="contradictions"):
        fit_spiked_majority(x * [3, 1, 1, 1], x[:, 0] > 0)
    with pytest.raises(ValueError, match="force"):
        fit_spiked_majority(x, y, "bogus")

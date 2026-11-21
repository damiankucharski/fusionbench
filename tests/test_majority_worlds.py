import itertools
import math

import numpy as np
import pytest

from fusionbench.methods.majority_worlds import (
    shared_majority_probability,
    two_world_bounds,
)


def test_posterior_equals_mixture_for_all_binary_corners_and_broadcasted_weights():
    a, b, c0, c1 = np.array(list(itertools.product((0, 1), repeat=4))).T
    w = np.array([0, 0.2, 0.5, 0.8, 1])[:, None]
    expected = (1 - w) * (a + b + c0 >= 2) + w * (a + b + c1 >= 2)
    actual = shared_majority_probability(a.astype(bool), b, c0, c1, w)
    np.testing.assert_allclose(actual, expected, atol=1e-15)
    assert shared_majority_probability(True, False, False, True, 0.3) == 0.3


@pytest.mark.parametrize("theta", [0, math.pi / 8, math.pi / 2, math.pi * 0.8, math.pi])
def test_gaussian_training_matches_testing_law(theta):
    n = 12
    risk = two_world_bounds(theta, 1, n)
    p = theta / (2 * math.pi)
    probability = (1 - p) ** n
    assert risk == pytest.approx(
        {
            "p_train": p,
            "p_test": p,
            "unresolved": probability,
            "world_error": probability / 2,
            "target_error": probability * p / 2,
            "target_brier": probability * p / 4,
        }
    )


@pytest.mark.parametrize(
    "theta", [0, math.pi / 6, math.pi / 2, 5 * math.pi / 6, math.pi]
)
def test_two_degrees_of_freedom_closed_form_and_no_observations(theta):
    risk = two_world_bounds(theta, np.int64(2), np.int64(0))
    assert risk["p_train"] == pytest.approx((1 - math.cos(theta)) / 4)
    assert risk["unresolved"] == 1.0


def test_rare_disagreements_survive_small_probability_rounding():
    # For k=2, p_train=(1-cos(theta))/4=sin(theta/2)^2/2.
    theta, n = 1e-9, 10**19
    p = math.sin(theta / 2) ** 2 / 2
    risk = two_world_bounds(theta, 2, n)
    assert risk["p_train"] == pytest.approx(p, rel=1e-12, abs=0)
    assert risk["unresolved"] == pytest.approx(math.exp(n * math.log1p(-p)))
    extreme = two_world_bounds(1e-155, 2, 10**310)
    assert extreme["unresolved"] == pytest.approx(math.exp(-0.125), rel=1e-12)
    assert two_world_bounds(0, 2, 10**1000)["unresolved"] == 1
    assert two_world_bounds(math.pi / 2, 2, 10**1000)["unresolved"] == 0


@pytest.mark.parametrize(
    "args",
    [
        (-1, 2, 0),
        (math.nan, 2, 0),
        (4, 2, 0),
        (0, 0, 0),
        (0, 2.5, 0),
        (0, 2, -1),
        (0, 2, True),
    ],
)
def test_invalid_risk_arguments(args):
    with pytest.raises(ValueError):
        two_world_bounds(*args)


@pytest.mark.parametrize(
    "args",
    [
        (0, 2, 0, 0, 0.5),
        (0, 0, np.nan, 0, 0.5),
        (0, 1, 0, 0, np.inf),
        (0, 1, 0, 0, -0.2),
    ],
)
def test_invalid_posterior_arguments(args):
    with pytest.raises(ValueError):
        shared_majority_probability(*args)

import numpy as np
import pytest

from fusionbench.methods.majority_interval import fit_majority_interval


def test_feasible_interval_and_analytic_posterior():
    # A decreasing vote labeled negative imposes theta > pi/4; an increasing
    # vote labeled negative imposes theta < atan2(2, 1).
    fit = fit_majority_interval([[1, -1, 1, -1], [-1, 1, -2, 1]], [0, 0], 1.4)
    assert fit.lower == pytest.approx(np.pi / 4)
    assert fit.upper == pytest.approx(np.arctan2(2, 1))
    assert not fit.lower_included and not fit.upper_included
    threshold = fit.lower + (fit.upper - fit.lower) / 4
    v, w = -np.sin(threshold), np.cos(threshold)
    raw = [[1, -1, v, w], [1, -1, -v, -w], [-1, -1, v, w], [1, 1, -v, -w]]
    np.testing.assert_array_equal(fit.predict_midpoint(raw), [1, 0, 0, 1])
    np.testing.assert_array_equal(fit.predict_minimax_positive(raw), [0.5, 0.5, 0, 1])
    np.testing.assert_allclose(fit.predict_uniform_positive(raw), [0.75, 0.25, 0, 1])


def test_shared_scores_ties_constants_and_empty_observations():
    fit = fit_majority_interval([], [], 1)
    assert (fit.lower, fit.upper) == (0, 1)
    assert fit.lower_included and fit.upper_included
    # Signed zeros in shared or rotating scores all use the positive convention.
    raw = [[0, -1, -0.0, 0], [-1, 0, -1, 0], [-0.0, 0, -5, 0], [-1, -1, 2, 0]]
    for predict in (
        fit.predict_midpoint,
        fit.predict_minimax_positive,
        fit.predict_uniform_positive,
    ):
        np.testing.assert_array_equal(predict(raw), [1, 0, 1, 0])
        assert predict(np.empty((0, 4))).shape == (0,)
    with pytest.raises(ValueError, match="shared"):
        fit_majority_interval([[1, 0, -90, 0]], [0], 1)
    with pytest.raises(ValueError, match="constant"):
        fit_majority_interval([[-1, 0, 0, 0]], [0], 1)


def test_closed_singleton_and_excluded_boundary():
    point = fit_majority_interval([[-1, 1, -1, 1], [-1, 1, 1, -1]], [1, 1], 1.3)
    assert point.lower == point.upper == np.pi / 4
    assert point.lower_included and point.upper_included
    for predict in (
        point.predict_midpoint,
        point.predict_minimax_positive,
        point.predict_uniform_positive,
    ):
        np.testing.assert_array_equal(predict([[-1, 1, -1, 1], [-1, 1, 1, -1]]), [1, 1])
    with pytest.raises(ValueError, match="no common feasible"):
        fit_majority_interval([[-1, 1, -1, 1], [-1, 1, -1, 1]], [1, 0], 1.3)

    strict = fit_majority_interval([[-1, 1, 0, -1]], [0], 1)
    assert strict.lower == 0 and not strict.lower_included
    raw = [[-1, 1, 0, -1]]
    assert strict.predict_midpoint(raw)[0] == 0
    assert strict.predict_uniform_positive(raw)[0] == 0
    assert strict.predict_minimax_positive(raw)[0] == 0.5


def test_fit_boundary_precision_and_outside_domain():
    # atan2 maps this exact 1:1 ratio to pi/4, a closed boundary of the domain.
    point = fit_majority_interval([[-1, 1, -1e300, 1e300]], [1], np.pi / 4)
    assert point.lower == point.upper
    # Adjacent floating-point endpoints leave a real-valued open interval.
    limit = np.nextafter(np.pi / 4, 2)
    interval = fit_majority_interval([[-1, 1, 1, -1]], [0], limit)
    assert interval.lower < interval.upper and not interval.lower_included
    assert interval.predict_midpoint([[-1, 1, 1, -1]])[0] == 0
    with pytest.raises(ValueError, match="no common feasible"):
        fit_majority_interval([[-1, 1, -100, 1e-300]], [1], 1.1)


@pytest.mark.parametrize("angle", [None, 0, -1, np.pi / 2, np.inf, np.nan, True])
def test_reject_invalid_domain(angle):
    with pytest.raises(ValueError, match="angle_max"):
        fit_majority_interval([], [], angle)


@pytest.mark.parametrize(
    ("x", "y"),
    [
        ([[1, 2]], [0]),
        ([[1, 2, np.nan, 3]], [1]),
        ([[1, 2, 3, 4]], [2]),
        ([[1, 2, 3, 4]], []),
    ],
)
def test_reject_invalid_features_and_labels(x, y):
    with pytest.raises(ValueError):
        fit_majority_interval(x, y, 1)

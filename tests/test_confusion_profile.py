import numpy as np
import pytest

from fusionbench.eval.confusion_profile import runner_up_profile


def test_strongest_wrong_includes_correct_and_incorrect_top_decisions():
    # All four rows prefer wrong class 1. Each half's smoothed distribution is (.75, .25).
    p = np.array([[0.65, 0.30, 0.05], [0.30, 0.55, 0.15]] * 2)[:, None, :]
    out = runner_up_profile(p, np.zeros(4, dtype=int))
    assert out["top_two_margin"] == pytest.approx(0.30)
    assert out["wrong_class_repeatability"] == pytest.approx(0.25)


def test_wrong_class_ties_split_the_vote_uniformly():
    p = np.tile([[[0.4, 0.3, 0.3, 0.0]]], (4, 1, 1))
    out = runner_up_profile(p, np.zeros(4, dtype=int))
    assert out["top_two_margin"] == pytest.approx(0.1)
    assert out["wrong_class_repeatability"] == pytest.approx(1 / 16)
    permutation = [3, 2, 0, 1]
    assert runner_up_profile(p[:, :, permutation], np.full(4, permutation.index(0))) == pytest.approx(out)


def test_halves_are_within_class_and_clip_only_after_macro_averaging():
    y = np.array([0, 1, 0, 1, 0, 0, 2, 0, 0])
    p = np.full((len(y), 2, 3), 1 / 3)
    p[y == 0, 0] = [0.6, 0.3, 0.1]  # normalized agreement +9/25
    p[y == 1, 0] = [[0.3, 0.6, 0.1], [0.1, 0.6, 0.3]]  # normalized agreement -1/9
    p[y == 2, 0] = [0.9, 0.0, 0.1]  # singleton excluded; second provider always ties
    out = runner_up_profile(p, y)
    assert out["wrong_class_repeatability"] == pytest.approx((9 / 25 - 1 / 9) / 4)
    assert runner_up_profile(p[y == 1], y[y == 1])["wrong_class_repeatability"] == 0.0


def test_binary_no_eligible_class_and_odd_halves():
    binary = np.array([[[0.1, 0.9]], [[0.5, 0.5]]])
    assert runner_up_profile(binary, [0, 1]) == pytest.approx({"top_two_margin": 0.4, "wrong_class_repeatability": 0})
    p = np.tile([[[0.0, 1.0, 0.0]]], (3, 1, 1))
    assert runner_up_profile(p, [0, 1, 2])["wrong_class_repeatability"] == 0.0
    assert runner_up_profile(p, [0, 0, 0])["wrong_class_repeatability"] == pytest.approx((2 / 4) * (1 / 3))


@pytest.mark.parametrize("p,y", [
    (np.ones((1, 3)), [0]),
    (np.zeros((0, 1, 3)), []),
    (np.full((2, 1, 3), 0.1), [0, 0]),
    (np.array([[[-0.1, 1.1]]]), [0]),
    (np.full((1, 1, 2), np.nan), [0]),
    (np.full((1, 1, 2), 0.5), [2]),
    (np.full((1, 1, 2), 0.5), [0.5]),
    (np.full((2, 1, 2), 0.5), [0]),
])
def test_rejects_invalid_banks(p, y):
    with pytest.raises(ValueError):
        runner_up_profile(p, y)

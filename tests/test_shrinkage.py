import numpy as np
import pytest

from fusionbench.methods.okapi import _brier_objective
from fusionbench.methods.shrinkage import brier_tree_weight


def test_shrinkage_estimates_an_interior_binary_weight():
    # Fraction positive 3/4 is the optimal constant forecast. Blending 0.1
    # and 0.9 requires a tree weight of 0.8125.
    anchor = np.tile([.9, .1], (4, 1))
    tree = anchor[:, ::-1]
    assert brier_tree_weight(tree, anchor, [1, 1, 1, 0]) == pytest.approx(.8125)


def test_multiclass_weight_beats_grid_and_handles_coincident_forecasts():
    rng = np.random.default_rng(41)
    tree = rng.dirichlet(np.ones(4), size=50)
    anchor = rng.dirichlet(np.ones(4), size=50)
    labels = rng.integers(4, size=50)
    w = brier_tree_weight(tree, anchor, labels)
    target = np.eye(4)[labels]
    def risk(weight):
        return np.mean((weight * tree + (1 - weight) * anchor - target) ** 2)
    assert risk(w) <= min(risk(t) for t in np.linspace(0, 1, 1001)) + 1e-15
    assert brier_tree_weight(tree, tree, labels) == 0


def test_gp_brier_fitness_is_bounded_for_wrong_and_perfect_forecasts():
    labels = np.array([0, 1])
    for perfect in (np.array([[0.], [1.]]), np.array([[1., 0.], [0., 1.]])):
        assert _brier_objective(perfect, labels) == 1.
        assert _brier_objective(1. - perfect, labels) == 0.

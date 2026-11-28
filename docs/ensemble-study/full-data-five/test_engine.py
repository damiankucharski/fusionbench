"""Tests for full-row parity and attribution of the selection effect."""

import numpy as np
import pytest
from engine import BASE_OPTIONS, fit_round1, partition, select_on_heldout

from fusionbench.eval.suite_point_gp import predict_tree
from fusionbench.methods.okapi import OKAPIMethod


@pytest.mark.parametrize("classes", [2, 4])
def test_round1_control_and_shared_discovery_fit(classes):
    rng = np.random.default_rng(32990)
    x = rng.dirichlet(np.ones(classes), (128, 5))
    y = rng.integers(classes, size=len(x))
    task = "binary" if classes == 2 else "multiclass"
    got = fit_round1(x, y)
    native = OKAPIMethod(seed=0, **BASE_OPTIONS).fit(x, y, task=task, n_classes=classes)
    target_x = rng.dirichlet(np.ones(classes), (17, 5))
    np.testing.assert_array_equal(got["full"].predict(target_x), native.predict(target_x))
    assert got["full"].best_tree.structural_key() == native.best_tree.structural_key()

    discovery, heldout = partition(len(x))
    assert len(discovery) == 89 and len(heldout) == 39
    np.testing.assert_array_equal(np.sort(np.r_[discovery, heldout]), np.arange(len(x)))
    reduced, split = got["reduced"], got["split"]
    assert reduced.okapi is split.okapi
    np.testing.assert_array_equal(reduced.okapi.gt_tensor.ravel(), y[discovery])
    np.testing.assert_array_equal(got["full"].okapi.gt_tensor.ravel(), y)

    reference = np.eye(classes)[y[heldout]]
    losses = []
    for tree in [reduced.best_tree, *reduced.okapi.population]:
        prediction = predict_tree(tree, x[heldout])
        losses.append(np.square(prediction - reference).sum(1).mean())
    assert split.selection_["winner_heldout_brier"] == pytest.approx(min(losses), abs=1e-14)
    assert split.selection_["champion_heldout_brier"] == pytest.approx(losses[0], abs=1e-14)


def test_heldout_tie_retains_the_discovery_winner():
    rng = np.random.default_rng(71)
    x = rng.dirichlet([1, 1], (96, 4))
    y = rng.integers(2, size=96)
    model = OKAPIMethod(seed=0, **BASE_OPTIONS).fit(x, y, task="binary", n_classes=2)
    tied = np.full_like(x, 0.5)
    selected = select_on_heldout(model, tied, y)
    assert selected.selection_["winner_index"] == 0
    assert selected.best_tree is model.best_tree


@pytest.mark.parametrize("num_groups", [16, 128])
def test_whole_scene_partition(num_groups):
    groups = np.tile(np.arange(num_groups), 128 // num_groups)
    discovery, heldout = partition(128, groups)
    assert len(np.unique(groups[discovery])) == 7 * num_groups // 10
    assert set(groups[discovery]).isdisjoint(groups[heldout])
    np.testing.assert_array_equal(np.sort(np.r_[discovery, heldout]), np.arange(128))


def test_one_hundred_provider_node_budget_is_shared():
    rng = np.random.default_rng(12934)
    x = rng.dirichlet([1, 1], (128, 100))
    y = rng.integers(2, size=len(x))
    models = fit_round1(x, y)
    assert {model.max_nodes for model in models.values()} == {101}
    native = OKAPIMethod(seed=0, **{**BASE_OPTIONS, "max_nodes": 101})
    native.fit(x, y, task="binary", n_classes=2)
    assert native.best_tree.structural_key() == models["full"].best_tree.structural_key()

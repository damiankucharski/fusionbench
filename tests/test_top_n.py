import numpy as np
import pytest

from fusionbench.cli import _comparison_names
from fusionbench.eval.metrics import HIGHER_BETTER, score
from fusionbench.eval.runner import evaluate_dataset
from fusionbench.methods.averaging import TopN
from fusionbench.methods.registry import default_methods
from fusionbench.types import Dataset


def binary(p):
    return np.stack([1 - p, p], axis=-1)


def test_recovers_better_late_prefix_after_downturn_and_rejects_weak_provider():
    # Each column is a provider's error probability on the four fit examples.
    errors = np.array([[.4,.9,.3,.2], [.5,.8,.6,.2], [.9,.1,.2,.7], [1,1,1,1]]).T
    y = np.array([1,0,1,0])
    p = binary(np.where(y[:,None] == 1, 1-errors, errors))
    model = TopN(metric="brier").fit(p, y, task="binary", n_classes=2)
    assert model.ranking_.tolist() == [0,1,2,3]
    assert model.prefix_scores_[1] > model.prefix_scores_[0] > model.prefix_scores_[2]
    assert model.n_selected_ == 3
    query = binary(np.array([[.1,.7,.4,.99]]))
    np.testing.assert_allclose(model.predict(query), [[.6,.4]])


@pytest.mark.parametrize("metric", HIGHER_BETTER)
def test_multiclass_matches_independent_enumeration(metric):
    rng = np.random.default_rng(14061)
    p = rng.dirichlet([2,2,2], (30, 4))
    y = np.tile(np.arange(3), 10)
    model = TopN(metric=metric).fit(p, y, task="multiclass", n_classes=3)
    direction = -1 if HIGHER_BETTER[metric] else 1
    rank = sorted(range(4), key=lambda i: direction * score(y,p[:,i],3)[metric])
    profiles = [score(y,p[:,rank[:n]].mean(1),3)[metric] for n in range(1,5)]
    np.testing.assert_array_equal(model.ranking_, rank)
    np.testing.assert_allclose(model.prefix_scores_, profiles)
    assert model.n_selected_ == 1+np.argmin(direction * np.array(profiles))


def test_ties_keep_order_and_smallest_prefix_and_all_can_win():
    tie = binary(np.full((4,3), .25))
    y = np.array([0,1,0,1])
    model = TopN("brier").fit(tie, y, task="binary", n_classes=2)
    assert list(model.ranking_) == [0,1,2] and model.n_selected_ == 1
    p = binary(np.array([[1,0,0],[0,1,0],[0,0,1]], dtype=float))
    model.fit(p, np.zeros(3, dtype=int), task="binary", n_classes=2)
    assert model.n_selected_ == 3


def test_runner_selects_by_dataset_metric_and_reports_size():
    y = np.array([0,1,0,1])
    p = binary(np.tile([.2,.5,.8], (4,1)))
    ds = Dataset("external","binary",2,["a","b","c"],p,y,p,y,meta={"primary_metric":"brier"})
    row = evaluate_dataset(ds, default_methods(["top_n"])).iloc[0]
    assert row.fit_metric == "brier" and row.n_selected == 1
    assert np.isfinite(row.test_brier)


@pytest.mark.parametrize("names,expected", [
    (["okapi"], ["okapi","simple_average","top_n","stacking_lr"]),
    (["stacking_logit","okapi_shrunk","top_n","top_n"], ["stacking_logit","okapi_shrunk","top_n","simple_average"]),
    (["simple_average"], ["simple_average"]),
    (None, None),
])
def test_comparison_methods(names, expected):
    assert _comparison_names(names) == expected

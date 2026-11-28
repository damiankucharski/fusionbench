import numpy as np
import pytest

from fusionbench import get_scenario
from fusionbench.methods.native_pattern import RegularizedNativePatternOKAPI
from fusionbench.methods.search_guard import SearchGuardedNativeOKAPI, row_brier
from fusionbench.methods.validation_guard import GuardedNativeOKAPI

pytestmark = pytest.mark.skipif(
    not SearchGuardedNativeOKAPI.available(), reason="requires native paired fitness"
)


@pytest.mark.parametrize("name", ["committee", "multiclass_committee"])
def test_zero_guard_identical_selection_and_paths_and_cost(name):
    ds = get_scenario(name).build(84, 74010)
    models = [
        RegularizedNativePatternOKAPI(seed=57, split_seed=82),
        SearchGuardedNativeOKAPI(seed=57, split_seed=82, z=0),
    ]
    for m in models:
        m.fit(ds.fit_preds, ds.fit_y, task=ds.task, n_classes=ds.n_classes)
    old, new = models
    assert new.chosen_ == old.chosen_
    for key in (
        "gp_path_digest",
        "score_calls",
        "total_score_calls",
        "validation_scores",
        "discovery_sha256",
        "validation_sha256",
    ):
        assert new.counts_[key] == old.counts_[key]
    np.testing.assert_array_equal(
        new.predict(ds.test_preds), old.predict(ds.test_preds)
    )


def test_positive_objective_uses_split_aligned_paired_half_brier_and_raw_validation():
    ds = get_scenario("committee").build(96, 5714)
    m = SearchGuardedNativeOKAPI(seed=818, split_seed=9)
    fit = m._training_fitness(ds.fit_preds, ds.fit_y)
    candidate = ds.fit_preds[:, 0, :]
    gains = row_brier(ds.fit_preds.mean(1), ds.fit_y) - row_brier(candidate, ds.fit_y)
    want = 1 - 0.5 * (
        row_brier(candidate, ds.fit_y).mean() + gains.std(ddof=1) / np.sqrt(len(gains))
    )
    assert fit(candidate[:, 1:2], ds.fit_y) == pytest.approx(want, abs=1e-14)
    m.fit(ds.fit_preds, ds.fit_y, task=ds.task, n_classes=ds.n_classes)
    from fusionbench.eval.suite_point_gp import predict_tree

    val = m.validation_indices_
    assert m.counts_["refit_rows"] == 96 and len(val) == 29
    for name, tree in m.discovery_trees_.items():
        p = predict_tree(tree, ds.fit_preds[val])
        assert m.counts_["validation_scores"][name] == pytest.approx(
            1 - 0.5 * row_brier(p, ds.fit_y[val]).mean()
        )


def test_outer_compares_one_frozen_menu_disjoint_from_whole_inner_fit():
    ds = get_scenario("committee").build(90, 5841)

    def fit(y):
        return GuardedNativeOKAPI(seed=99, split_seed=88).fit(
            ds.fit_preds, y, task=ds.task, n_classes=ds.n_classes
        )

    first = fit(ds.fit_y)
    assert len(set(first.outer_indices_) & set(first.inner_indices_)) == 0
    assert len(first.inner_indices_) + len(first.outer_indices_) == len(ds.fit_y)
    assert first.estimator_.counts_["refit_rows"] == len(first.inner_indices_)
    changed = ds.fit_y.copy()
    changed[first.outer_indices_] = 1 - changed[first.outer_indices_]
    other = fit(changed)
    assert (
        first.estimator_.counts_["gp_path_digest"]
        == other.estimator_.counts_["gp_path_digest"]
    )
    for k, p in first.candidate_predictions(ds.test_preds).items():
        np.testing.assert_array_equal(other.candidate_predictions(ds.test_preds)[k], p)
    np.testing.assert_allclose(
        first.candidate_predictions(ds.test_preds)["mean"],
        ds.test_preds.mean(1),
        atol=2e-12,
        rtol=0,
    )
    np.testing.assert_array_equal(
        first.predict(ds.test_preds),
        first.candidate_predictions(ds.test_preds)[first.chosen_],
    )

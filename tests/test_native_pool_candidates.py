"""Matching incumbent, held-out selection and full-refit semantics for native pools."""

import numpy as np
import pytest

from fusionbench.eval.suite_point_gp import _bind, predict_tree
from fusionbench.methods.averaging import TopN
from fusionbench.methods.native_pattern import (
    RegularizedNativePatternOKAPI,
    _brier_skill,
)
from fusionbench.methods.native_pool_candidates import (
    PoolCandidateNativePatternOKAPI,
    _pool_tree,
)

pytestmark = pytest.mark.skipif(
    not PoolCandidateNativePatternOKAPI.available(),
    reason="requires native pattern and mean fitters",
)


def _fit(model_type, x, y):
    classes = x.shape[-1]
    model = model_type(split_seed=0, seed=19, gp_validation_top_k=8)
    return model.fit(
        x, y, task="binary" if classes == 2 else "multiclass", n_classes=classes
    )


@pytest.mark.parametrize("classes", [2, 4])
def test_original_fit_and_score_replay_and_native_discovery_pools(classes):
    rng = np.random.default_rng(7783)
    x = rng.dirichlet(np.ones(classes), size=(60, 4))
    y = rng.integers(classes, size=60)
    q = rng.dirichlet(np.ones(classes), size=(11, 4))
    old = _fit(RegularizedNativePatternOKAPI, x, y)
    new = _fit(PoolCandidateNativePatternOKAPI, x, y)
    for key in (
        "discovery_sha256",
        "validation_sha256",
        "gp_path_digest",
        "head_fit_scores",
        "screen",
    ):
        assert old.counts_[key] == new.counts_[key]
    for key, score in old.counts_["validation_scores"].items():
        assert score == new.counts_["validation_scores"][key]
        assert old.trees_[key].structural_key() == new.trees_[key].structural_key()
    assert new.old_chosen_ == old.chosen_
    np.testing.assert_array_equal(new.predict_incumbent(q), old.predict(q))
    d, held = old.discovery_indices_, old.validation_indices_
    assert len(set(d) & set(held)) == 0
    valid = new.counts_["validation_scores"]
    d_prefix = TopN("brier").fit(x[d], y[d], task=new.task, n_classes=classes)
    np.testing.assert_array_equal(
        new.counts_["pool_candidates"]["discovery_top_n_indices"],
        d_prefix.selected_indices_,
    )
    for name, ref in (("top_n", d_prefix.predict(x[held])), ("mean", x[held].mean(1))):
        native = predict_tree(new.discovery_trees_[name], x[held])
        np.testing.assert_allclose(native, ref, rtol=0, atol=2e-15)
        assert valid[name] == pytest.approx(
            1 - np.square(native - np.eye(classes)[y[held]]).sum(1).mean() / 2
        )
    assert new.chosen_ == max(valid, key=valid.get)
    np.testing.assert_array_equal(new.predict(q), predict_tree(new.best_tree, q))
    extra = new.counts_["pool_candidates"]["extra_score_calls"]
    assert extra == 2 * (1 + x.shape[1] * (1 + (new.action_ == "top_n")))
    assert new.counts_["total_score_calls"] == old.counts_["total_score_calls"] + extra


def _fixed_incumbent(monkeypatch, score):
    """Fix the three old scores so each new outcome can be exercised directly."""

    def fixed_fit(self, x, y, *, task, n_classes, on_gp_candidate=None):
        self.task, self.n_classes, self.n_providers_ = task, n_classes, x.shape[1]
        order = np.random.default_rng(
            np.random.SeedSequence([self.split_seed, 51])
        ).permutation(len(y))
        self.discovery_indices_, self.validation_indices_ = np.split(
            order, [7 * len(y) // 10]
        )
        tree = _pool_tree(_bind(x), range(x.shape[1]))
        self.chosen_ = "flat"
        self.trees_ = dict.fromkeys(("flat", "augmented", "gp"), tree)
        self.discovery_trees_ = self.trees_.copy()
        self.counts_ = {
            "validation_scores": dict.fromkeys(self.trees_, score),
            "seconds": {},
            "score_calls": {"family_validation": 3},
        }
        return self

    monkeypatch.setattr(RegularizedNativePatternOKAPI, "fit", fixed_fit)


@pytest.mark.parametrize(("incumbent", "expected"), [(0.75, "native"), (0.74, "mean")])
def test_exact_tie_precedence_and_deployment(monkeypatch, incumbent, expected):
    x = np.full((40, 3, 2), 0.5)
    y = np.arange(len(x)) % 2
    _fixed_incumbent(monkeypatch, incumbent)
    fit = _fit(PoolCandidateNativePatternOKAPI, x, y)
    assert _brier_skill(x[:, 0], y) == 0.75
    assert fit.action_ == expected
    assert fit.counts_["score_calls"]["top_n_refit"] == 0
    assert fit.counts_["pool_candidates"]["refit_top_n_indices"] is None
    np.testing.assert_array_equal(fit.predict(x[:5]), x[:5, 0])


def test_top_n_discovery_ignores_held_labels_and_refits_only_if_winning(monkeypatch):
    _fixed_incumbent(monkeypatch, score=-1.0)
    order = np.random.default_rng(np.random.SeedSequence([0, 51])).permutation(100)
    d, held = np.split(order, [70])
    p = np.empty((100, 3))
    p[d], p[held] = [0.25, 0.75, 0.35], [0.75, 0.05, 0.99]
    x = np.stack([1 - p, p], axis=2)
    y = np.zeros(100, dtype=int)
    y[d[:15]] = 1
    y[held] = 1
    fit = _fit(PoolCandidateNativePatternOKAPI, x, y)
    assert fit.action_ == "top_n"
    info = fit.counts_["pool_candidates"]
    assert info["discovery_top_n_indices"] == [0]
    full = TopN("brier").fit(x, y, task="binary", n_classes=2)
    assert info["refit_top_n_indices"] == full.selected_indices_.tolist()
    assert info["refit_top_n_indices"] != info["discovery_top_n_indices"]
    np.testing.assert_allclose(fit.predict(x), full.predict(x), rtol=0, atol=2e-15)
    # Relabel only this held-out stratum, leaving every discovery observation intact.
    y[held] = 0
    other = _fit(PoolCandidateNativePatternOKAPI, x, y)
    assert other.action_ == "mean"
    assert other.counts_["pool_candidates"]["discovery_top_n_indices"] == [0]
    assert (
        other.discovery_trees_["top_n"].structural_key()
        == fit.discovery_trees_["top_n"].structural_key()
    )
    assert other.counts_["score_calls"]["top_n_refit"] == 0


def test_pools_registered_with_the_evaluated_shortlist_as_opt_in():
    from fusionbench.methods.registry import default_methods

    name = "okapi_native_pattern_l2_pools"
    assert name not in {m.name for m in default_methods(only_available=False)}
    (model,) = default_methods([name], only_available=False)
    assert model.name == name
    assert (model.seed, model.split_seed, model.gp_validation_top_k) == (0, 0, 8)

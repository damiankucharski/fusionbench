"""Public native heads and their opt-in use with evolved feature subtrees."""

import json

import numpy as np
import pytest

from fusionbench.eval.metrics import normalize_proba
from fusionbench.eval.native_switch import serialize
from fusionbench.methods.class_head import OKAPIClassDiagonal, OKAPIClassFull
from fusionbench.methods.native_pattern import NativePatternOKAPI
from fusionbench.methods.registry import FACTORIES, default_methods


def test_class_methods_are_explicit_registry_choices():
    keys = {"okapi_class_diagonal", "okapi_class_full", "stacking_shared_bias", "stacking_diagonal"}
    assert keys.isdisjoint(m.name for m in default_methods(only_available=False))
    assert {m.name for m in default_methods(sorted(keys), only_available=False)} == keys


@pytest.mark.parametrize("method_type", [OKAPIClassDiagonal, OKAPIClassFull])
def test_standalone_native_probabilities_and_serialized_class_order(method_type, monkeypatch):
    pytest.importorskip("okapi.multiclass_affine")
    from sklearn.linear_model import LogisticRegression

    def forbidden(*args, **kwargs):
        pytest.fail("The native adapter called logistic regression")

    monkeypatch.setattr(LogisticRegression, "fit", forbidden)
    rng = np.random.default_rng(58)
    x = rng.dirichlet([0.5, 1, 1.5], size=(56, 3))
    y = np.where(x[:, 0, 2] > 0.5, 0, 2)  # Class 1 is absent from fit labels.
    calls = []

    def score(p, labels):
        calls.append(1)
        assert p.shape == (len(labels), 3)
        return (p[np.arange(len(labels)), labels] >= 0.45).mean()

    assert FACTORIES[method_type.name] is method_type and method_type.available()
    m = method_type(max_evaluations=240, fitness=score)
    m.fit(x[:44], y[:44], task="multiclass", n_classes=3)
    assert m.fit_info_["mode"] == m.mode
    assert len(calls) == m.fit_info_["evaluations"] <= 240
    json.dumps(m.fit_info_)
    np.testing.assert_array_equal(m.classes_, [0, 1, 2])
    saved = json.loads(json.dumps(serialize(m.tree_.root)))
    (h,) = saved["children"]
    assert saved["id"] == "m0.npy" and [child["id"] for child in h["children"]] == ["m1.npy", "m2.npy"]
    assert h["type"] == "MulticlassAffineNode" and h["mode"] == m.mode
    w = np.asarray(h["coefficients"])
    z = x[44:]
    if m.mode == "diagonal":
        logits = np.sum(z * w, axis=1)
    else:
        logits = np.einsum("nmi,mij->nj", z, w)
    logits += h["intercept"]
    p = np.exp(logits - logits.max(axis=1, keepdims=True))
    np.testing.assert_allclose(m.predict(z), normalize_proba(p / p.sum(axis=1, keepdims=True)), atol=1e-14)
    with pytest.raises(ValueError, match="axes"):
        m.predict(z[:, :2])


def test_class_head_rejects_unsupported_banks_and_settings():
    m = OKAPIClassFull()
    with pytest.raises(ValueError, match="Fit"):
        m.predict(np.zeros((3, 1, 3)))
    with pytest.raises(ValueError, match="budget"):
        OKAPIClassFull(max_evaluations=0)
    with pytest.raises(ValueError, match="class_head"):
        NativePatternOKAPI(class_head="scalar")
    with pytest.raises(ValueError, match="explicit class_head"):
        NativePatternOKAPI(class_head_max_evaluations=10)
    pytest.importorskip("okapi.multiclass_affine")
    with pytest.raises(ValueError, match="three classes"):
        m.fit(np.full((3, 2, 2), 0.5), [0, 1, 0], task="binary", n_classes=2)
    with pytest.raises(ValueError, match="class index"):
        m.fit(np.full((3, 2, 3), 1 / 3), [0, 1, 3], task="multiclass", n_classes=3)


def test_evolved_features_feed_the_native_class_head_on_both_splits():
    pytest.importorskip("okapi.multiclass_affine")
    from fusionbench.eval.suite_point_gp import predict_tree

    rng = np.random.default_rng(109)
    x = rng.dirichlet([1, 1, 1], size=(105, 3))
    y = np.minimum(x[:, 0], x[:, 1]).argmax(axis=1)
    m = NativePatternOKAPI(seed=79, l2_strength=0.25, class_head="full", class_head_max_evaluations=36)
    m.fit(x[:90], y[:90], task="multiclass", n_classes=3)
    assert m.counts_["class_head"] == "full" and m.counts_["class_head_max_evaluations"] == 36
    for family in (m.discovery_trees_, m.trees_):
        raw = family["flat"].root.children[-1]
        extended = family["augmented"].root.children[-1]
        assert raw.mode == extended.mode == "full"
        assert raw.coefficients.shape == (3, 3, 3)
        assert len(extended.coefficients) > len(raw.coefficients)
        assert all(child.children for child in extended.children[2:])
    for phase in ("flat_discovery", "augmented_discovery", "flat_refit", "augmented_refit"):
        assert 0 < m.counts_["score_calls"][phase] <= 36
    deployed = m.predict(x[90:])
    np.testing.assert_array_equal(deployed, predict_tree(m.trees_[m.chosen_], x[90:]))
    np.testing.assert_allclose(deployed.sum(axis=1), 1)

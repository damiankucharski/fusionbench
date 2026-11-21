import numpy as np
import pytest
from scipy.special import softmax

from fusionbench.eval.metrics import normalize_proba
from fusionbench.methods.averaging import SimpleAverage, ValWeighted


def test_weighted_compilation_uses_exact_fitted_masses_and_normalization():
    pytest.importorskip("okapi.softmax_pool")
    rng = np.random.default_rng(113)
    w, b = rng.normal(size=(4, 3, 5)), rng.normal(size=(4, 3))
    x = rng.normal(size=(600, 5))
    providers = softmax(np.einsum("nd,kcd->nkc", x, w) + b, axis=2)
    y = providers[:, 0].argmax(1)
    fusion = ValWeighted().fit(providers[:200], y[:200], task="multiclass", n_classes=3)
    direct = fusion.predict(providers[200:])
    model = fusion.compile(w, b, keep_sources=True)
    actual = np.abs(direct - model.predict(x[200:])).sum(1) / 2
    assert np.all(actual <= model.oscillation_bound(x[200:]) + 1e-12)
    assert np.all(actual <= model.error_bound(x[200:]) + 1e-12)
    np.testing.assert_allclose(model.raw_model.mixing, fusion.w)
    np.testing.assert_array_equal(model.predict_labels(x[200:]), model.predict(x[200:]).argmax(1))
    np.testing.assert_allclose(model.predict(x[202]), model.predict(x[202:203])[0])
    with pytest.raises(ValueError):
        ValWeighted().compile(w, b)


def test_probability_floor_is_nonexpansive_and_matches_identical_experts():
    pytest.importorskip("okapi.softmax_pool")
    p = np.array([[0.0, 1.0, 0.0], [1e-15, 1e-10, 1 - 1e-10 - 1e-15], [0.2, 0.3, 0.5]])
    q = p[:, ::-1]

    def tv(a, b):
        return abs(a - b).sum(1) / 2

    assert np.all(tv(normalize_proba(p), normalize_proba(q)) <= tv(p, q) + 1e-15)
    w = np.zeros((2, 3, 1))
    b = np.tile([2000.0, 0.0, -2000.0], (2, 1))
    x = np.ones((1, 1))
    fusion = SimpleAverage()
    model = fusion.compile(w, b)
    np.testing.assert_allclose(
        model.predict(x), fusion.predict(np.array([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]))
    )
    assert model.error_bound(x)[0] == 0


def test_compiled_raw_estimator_labels():
    pytest.importorskip("okapi.softmax_pool")
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(74)
    x = rng.normal(size=(90, 3))
    y = np.where(x[:, 0] > 0.2, "positive", "negative")
    experts = [LogisticRegression(C=c).fit(x, y) for c in (1, 10)]
    ensemble = np.stack([m.predict_proba(x) for m in experts], axis=1)
    fit = SimpleAverage()
    model = fit.compile_estimators(experts)
    tv = np.abs(fit.predict(ensemble) - model.predict(x)).sum(1) / 2
    assert np.all(tv <= model.error_bound(x) + 1e-12)
    np.testing.assert_array_equal(
        model.predict_labels(x), experts[0].classes_[model.predict(x).argmax(1)]
    )

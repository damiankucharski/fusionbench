import inspect
import warnings

import numpy as np
import pytest
from fusionbench.methods.sklearn_logits import sklearn_affine_logits
from scipy.special import softmax
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def observations():
    rng = np.random.default_rng(239)
    x = rng.normal(size=(120, 6)) * [2, 1, 0.5, 1, 1, 3] + [3, 0, 1, 0, 4, 2]
    y = np.array(["blue", "green", "red"])[
        np.argmax(np.column_stack((x[:, 1], x[:, 2] - 1, x[:, 5] / 3 - 0.6)), axis=1)
    ]
    return x, y


def check_probas(models, masks, x, w, b):
    for i, (est, idx) in enumerate(zip(models, masks)):
        expected = est.predict_proba(x[:, idx])
        np.testing.assert_allclose(softmax(x @ w[i].T + b[i], axis=1), expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("with_mean", [True, False])
@pytest.mark.parametrize("with_std", [True, False])
def test_compose_feature_views_and_scaler_on_shared_raw_inputs(with_mean, with_std):
    x, y = observations()
    ix0 = [5, 0, 5]
    ix1 = np.array([False, True, True, False, True, False])
    models = [
        LogisticRegression().fit(x[:, ix0], y),
        make_pipeline(StandardScaler(with_mean=with_mean, with_std=with_std), LogisticRegression()).fit(
            x[:, ix1], y
        ),
    ]
    w, b, classes = sklearn_affine_logits(models, feature_indices=[ix0, ix1])
    assert w.shape == (2, 3, 6) and b.shape == (2, 3)
    np.testing.assert_array_equal(classes, models[0].classes_)
    check_probas(models, [ix0, ix1], x - 0.3, w, b)


def test_binary_single_logit_strings_and_unused_raw_columns():
    x, y = observations()
    labels = np.where(y == "red", "positive", "negative")
    m = LogisticRegression(fit_intercept=False).fit(x[:, [1, 4]], labels)
    w, b, classes = sklearn_affine_logits([m], feature_indices=[[1, 4]], n_features=8)
    x = np.pad(x, [(0, 0), (0, 2)])
    np.testing.assert_array_equal(classes, ["negative", "positive"])
    np.testing.assert_array_equal(w[0, 0], 0.0)
    check_probas([m], [[1, 4]], x, w, b)


def test_aligns_real_class_labels_and_matrix_input_validation():
    x, y = observations()
    a = LogisticRegression().fit(x, y)
    b = LogisticRegression().fit(x, y)
    order = [2, 0, 1]
    b.classes_, b.coef_, b.intercept_ = b.classes_[order], b.coef_[order], b.intercept_[order]
    w, intercept, classes = sklearn_affine_logits([a, b])
    expected = b.predict_proba(x)[:, [list(b.classes_).index(c) for c in classes]]
    np.testing.assert_allclose(softmax(x @ w[1].T + intercept[1], axis=1), expected, atol=1e-12)
    with pytest.raises(NotFittedError):
        sklearn_affine_logits([LogisticRegression()])
    with pytest.raises(ValueError, match="StandardScaler"):
        sklearn_affine_logits([make_pipeline(PolynomialFeatures(), LogisticRegression(max_iter=1000)).fit(x, y)])
    with pytest.raises(TypeError, match="one-vs-rest|One-vs-rest"):
        sklearn_affine_logits([OneVsRestClassifier(LogisticRegression()).fit(x, y)])
    with pytest.raises(ValueError, match="same class"):
        sklearn_affine_logits([a, LogisticRegression().fit(x, np.where(y == "red", "A", "B"))])
    with pytest.raises(ValueError, match="feature dimension"):
        sklearn_affine_logits([a], feature_indices=[[0, 1, 2]])
    with pytest.raises(ValueError, match="shared raw"):
        sklearn_affine_logits([a], n_features=8)


@pytest.mark.skipif(
    "multi_class" not in inspect.signature(LogisticRegression).parameters,
    reason="This sklearn release removed the older binary multinomial parameter",
)
def test_older_explicit_binary_multinomial_and_multiclass_ovr():
    x, y = observations()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        binary = LogisticRegression(multi_class="multinomial").fit(x, y == "red")
        ovr = LogisticRegression(multi_class="ovr").fit(x, y)
    w, b, _ = sklearn_affine_logits([binary])
    np.testing.assert_allclose(w[0, 0], -w[0, 1])
    check_probas([binary], [slice(None)], x, w, b)
    with pytest.raises(ValueError, match="one-vs-rest"):
        sklearn_affine_logits([ovr])

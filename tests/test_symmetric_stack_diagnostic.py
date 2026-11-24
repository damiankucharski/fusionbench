from itertools import permutations
from pathlib import Path
from runpy import run_path

import numpy as np
import pytest
from scipy.optimize import check_grad
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from fusionbench.eval.metrics import normalize_proba
from fusionbench.methods.stacking import StackingLR

_module = run_path(
    str(
        Path(__file__).resolve().parents[1]
        / "docs/ensemble-study/external-coverage/symmetric.py"
    )
)
SymmetricStackingLR = _module["SymmetricStackingLR"]


def bank(seed, n, classes):
    rng = np.random.default_rng(seed)
    p = softmax(rng.normal(size=(n, 4, classes)) + np.linspace(-2, 1, classes), axis=2)
    p[:, 2] = p[:, 1]  # rank-deficient inputs
    p[:, 3] = softmax(np.arange(classes, dtype=float))  # uninformative provider
    y = np.argmax(4.0 * p[:, 0] - 2.0 * p[:, 1] + rng.normal(size=(n, classes)), axis=1)
    return p, y


@pytest.mark.parametrize("classes", [2, 3])
@pytest.mark.parametrize("regularization", [0.35, 1.0])
def test_weighted_complete_augmentation_matches_sklearn(classes, regularization):
    p, y = bank(19, 57, classes)
    test, _ = bank(24, 31, classes)
    permutations_ = tuple(permutations(range(classes)))
    repeated = np.concatenate([p[:, :, perm] for perm in permutations_])
    labels = np.concatenate([np.argsort(perm)[y] for perm in permutations_])
    sample_weight = np.full(len(labels), 1 / len(permutations_))
    public = StackingLR(c_value=regularization)
    # Use the actual public stacker's features, including its flattening order.
    features = public._features(repeated)
    scaler = StandardScaler().fit(features, sample_weight=sample_weight)
    full = LogisticRegression(
        C=regularization, tol=1e-12, max_iter=2000, solver="lbfgs"
    )
    full.fit(scaler.transform(features), labels, sample_weight=sample_weight)
    expected = normalize_proba(
        full.predict_proba(scaler.transform(public._features(test)))
    )
    task = "binary" if classes == 2 else "multiclass"
    fitted = SymmetricStackingLR(c_value=regularization).fit(
        p, y, task=task, n_classes=classes
    )
    np.testing.assert_allclose(
        np.repeat(fitted.scale_, classes), scaler.scale_, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(fitted.predict(test), expected, rtol=0, atol=2e-7)
    assert fitted.gradient_inf_ < 1e-5
    for perm in permutations_:
        np.testing.assert_allclose(
            fitted.predict(test[:, :, perm]),
            fitted.predict(test)[:, perm],
            rtol=0,
            atol=1e-14,
        )
        relabeled = SymmetricStackingLR(c_value=regularization).fit(
            p[:, :, perm], np.argsort(perm)[y], task=task, n_classes=classes
        )
        np.testing.assert_allclose(
            relabeled.weights_, fitted.weights_, rtol=0, atol=3e-8
        )


def test_analytical_gradient_matches_finite_differences():
    p, y = bank(38, 23, 3)
    features, _, _, c = _module["_provider_rows"](p)
    features = StandardScaler().fit_transform(features)

    def objective(w):
        return _module["_loss_and_gradient"](w, features, y, c, 2.0)

    point = np.array([0.12, -0.4, 0.32, 0.8])
    error = check_grad(lambda w: objective(w)[0], lambda w: objective(w)[1], point)
    assert error < 2e-6


def test_constant_providers_do_not_encode_class_frequency():
    x = np.full((12, 2, 3), 1 / 3)
    fitted = SymmetricStackingLR().fit(
        x, np.zeros(12, dtype=int), task="multiclass", n_classes=3
    )
    np.testing.assert_array_equal(fitted.weights_, np.zeros(2))
    np.testing.assert_allclose(fitted.predict(x), x[:, 0])
    with pytest.raises(ValueError, match="axes"):
        fitted.predict(x[:, :1])


@pytest.mark.parametrize("classes", [2, 3])
def test_pooled_unrestricted_matches_full_fit_on_the_same_features(classes):
    p, y = bank(81, 81, classes)
    test, _ = bank(84, 23, classes)
    task = "binary" if classes == 2 else "multiclass"
    unrestricted = _module["PooledScaleStackingLR"]().fit(
        p, y, task=task, n_classes=classes
    )
    symmetric = SymmetricStackingLR().fit(p, y, task=task, n_classes=classes)
    np.testing.assert_array_equal(unrestricted.scale_, symmetric.scale_)
    np.testing.assert_array_equal(unrestricted.scaler_.mean_, symmetric.scaler_.mean_)
    # Independent construction: full column order, pooled moments on raw inputs.
    mean, scale = p.mean(axis=(0, 2)), p.std(axis=(0, 2))

    def design(raw):
        return ((raw - mean[None, :, None]) / scale[None, :, None]).reshape(
            len(raw), -1
        )

    full = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs").fit(design(p), y)
    expected = normalize_proba(full.predict_proba(design(test)))
    np.testing.assert_allclose(unrestricted.predict(test), expected, atol=1e-12, rtol=0)
    assert np.max(abs(unrestricted.predict(test) - symmetric.predict(test))) > 0.01


def test_pooled_full_head_preserves_missing_class_output_column():
    p, y = bank(91, 60, 3)
    y = y % 2
    model = _module["PooledScaleStackingLR"]().fit(p, y, task="multiclass", n_classes=3)
    actual = model.predict(p)
    assert actual.shape == (60, 3)
    np.testing.assert_allclose(actual[:, 2], 1e-12, atol=1e-22, rtol=0)
    np.testing.assert_allclose(actual.sum(1), 1.0)

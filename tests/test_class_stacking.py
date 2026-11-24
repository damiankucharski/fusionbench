import numpy as np
import pytest
from scipy.linalg import helmert
from scipy.special import logsumexp
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from fusionbench.methods.class_stacking import (
    DiagonalStackingLR,
    SharedBiasStackingLR,
    _objective,
)


@pytest.mark.parametrize("classes,active", [(2, 2), (4, 4), (5, 3), (5, 2)])
@pytest.mark.parametrize("diagonal", [False, True])
def test_loss_is_the_exact_penalized_full_objective_and_gradient(
    classes, active, diagonal
):
    rng = np.random.default_rng(11)
    n, m = 31, 3
    z = rng.normal(size=(n, classes, m))
    z -= z.mean(1, keepdims=True)
    observed = np.arange(classes - active, classes)
    y = rng.integers(active, size=n)
    theta = rng.normal(size=m * (active if diagonal else 1) + active - 1)
    basis = helmert(active)
    c_value = 0.73

    def restricted(t):
        return _objective(t, z[:, observed], y, basis, classes, c_value, diagonal)

    value, grad = restricted(theta)
    # Construct a full active-output × input-class matrix independently for
    # every provider. Centering its rows and columns gives the minimum norm
    # model with identical softmax probabilities on simplex-valued inputs.
    w = theta[: len(theta) - active + 1].reshape(-1, m)
    b = theta[-(active - 1) :] @ basis
    coefficient = np.zeros((active, classes, m))
    coefficient[np.arange(active), observed, :] = w
    po = np.eye(active) - np.ones((active, active)) / active
    pi = np.eye(classes) - np.ones((classes, classes)) / classes
    coefficient = np.einsum("ai,ijm,jc->acm", po, coefficient, pi)
    logits = np.einsum("acm,ncm->na", coefficient, z) + b
    full_loss = (logsumexp(logits, axis=1) - logits[np.arange(n), y]).sum()
    squared_norm = (
        np.square(coefficient[1] - coefficient[0]).sum()
        if active == 2
        else np.square(coefficient).sum()
    )
    np.testing.assert_allclose(
        value, full_loss + squared_norm / (2 * c_value), rtol=1e-13
    )
    eps = 1e-6
    central = [
        (restricted(theta + eps * d)[0] - restricted(theta - eps * d)[0]) / (2 * eps)
        for d in np.eye(len(theta))
    ]
    np.testing.assert_allclose(grad, central, rtol=2e-6, atol=3e-7)


@pytest.mark.parametrize("cls", [SharedBiasStackingLR, DiagonalStackingLR])
def test_binary_collapses_to_full_ridge_logistic_on_pooled_provider_scale(cls):
    rng = np.random.default_rng(22)
    x = rng.dirichlet([0.3, 2], (140, 3))
    y = rng.integers(2, size=len(x))
    test = rng.dirichlet([0.4, 1.4], (29, 3))
    model = cls(c_value=0.8).fit(x, y, task="binary", n_classes=2)
    scaler = StandardScaler().fit(x.transpose(0, 2, 1).reshape(-1, 3))

    def design(p):
        return (
            scaler.transform(p.transpose(0, 2, 1).reshape(-1, 3))
            .reshape(-1, 2, 3)
            .transpose(0, 2, 1)
            .reshape(len(p), -1)
        )

    full = LogisticRegression(C=0.8, max_iter=2000, tol=1e-12).fit(design(x), y)
    np.testing.assert_allclose(
        model.predict(test), full.predict_proba(design(test)), atol=1e-7, rtol=0
    )


def test_nested_fits_and_permutation_of_labels_and_providers():
    rng = np.random.default_rng(99)
    n, c, m = 250, 4, 3
    x = rng.dirichlet(np.ones(c), (n, m))
    y = rng.integers(c, size=n)
    perm = rng.permutation(c)
    relabeled = np.argsort(perm)[y]
    shared = SharedBiasStackingLR().fit(x, y, task="multiclass", n_classes=c)
    diagonal = DiagonalStackingLR().fit(x, y, task="multiclass", n_classes=c)
    full = LogisticRegression(C=1.0, max_iter=2000, tol=1e-12)
    z = (
        shared.scaler_.transform(x.transpose(0, 2, 1).reshape(-1, m))
        .reshape(n, c, m)
        .transpose(0, 2, 1)
        .reshape(n, -1)
    )
    full.fit(z, y)
    ln = full.predict_log_proba(z)
    loss = -ln[np.arange(n), y].sum() + 0.5 * np.square(full.coef_).sum()
    assert loss < diagonal.objective_ < shared.objective_
    for cls, original in [
        (SharedBiasStackingLR, shared),
        (DiagonalStackingLR, diagonal),
    ]:
        p = x[:, ::-1, perm]
        shifted = cls().fit(p, relabeled, task="multiclass", n_classes=c)
        np.testing.assert_allclose(
            shifted.predict(p), original.predict(x)[:, perm], atol=2e-7
        )


@pytest.mark.parametrize("cls", [SharedBiasStackingLR, DiagonalStackingLR])
def test_uniform_inputs_recover_class_priors_and_pad_missing_labels(cls):
    p = np.full((10, 2, 4), 0.25)
    y = np.array([2] * 7 + [0] * 3)
    model = cls().fit(p, y, task="multiclass", n_classes=4)
    np.testing.assert_allclose(
        model.predict(p)[0], [0.3, 1e-12, 0.7, 1e-12], atol=2e-12
    )
    model.fit(p, np.ones(10), task="multiclass", n_classes=4)
    np.testing.assert_allclose(
        model.predict(p)[0], [1e-12, 1.0, 1e-12, 1e-12], atol=4e-12
    )

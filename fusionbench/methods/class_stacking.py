"""Logistic stackers with pooled provider scales and free class intercepts."""

from numbers import Integral, Real

import numpy as np
from scipy.linalg import helmert
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax
from sklearn.preprocessing import StandardScaler

from fusionbench.eval.metrics import normalize_proba
from fusionbench.methods.base import Method


def _provider_values(preds):
    values = np.asarray(preds, dtype=np.float64)
    if values.ndim != 3 or min(values.shape) < 1 or values.shape[2] < 2:
        raise ValueError(
            "Expected probabilities with nonempty row, provider and class axes"
        )
    if (
        not np.isfinite(values).all()
        or np.any(values < 0)
        or np.any(values > 1)
        or not np.allclose(values.sum(2), 1, rtol=0, atol=1e-6)
    ):
        raise ValueError(
            "Provider probabilities must be finite, lie in [0, 1] and sum to one"
        )
    return values.transpose(0, 2, 1)


def _scaled(values, scaler):
    z = scaler.transform(values.reshape(-1, values.shape[2])).reshape(values.shape)
    # Remove provider-wise constants before selecting the observed class channels.
    # This also enforces the simplex constraint on float32 probability records.
    return z - z.mean(axis=1, keepdims=True)


def _objective(params, z, labels, basis, n_classes, c_value, diagonal):
    """Summed log loss and gradient including the least equivalent full-block penalty."""
    n, active, providers = z.shape
    splits = providers * (active if diagonal else 1)
    weights = params[:splits].reshape(-1, providers)
    intercept = params[splits:] @ basis
    logits = np.einsum("ncm,cm->nc", z, weights) + intercept
    logp = logits - logsumexp(logits, axis=1, keepdims=True)
    loss = -logp[np.arange(n), labels].sum()
    residual = np.exp(logp)
    residual[np.arange(n), labels] -= 1
    slope_grad = np.einsum("nc,ncm->cm", residual, z)

    # sklearn's two-class parameter is a single log-odds row; it has twice
    # the squared norm of the two centered multinomial output rows.
    strength = (2.0 if active == 2 else 1.0) / c_value
    if diagonal:
        penalty_grad = strength * (
            (1 - 1 / active - 1 / n_classes) * weights
            + weights.sum(0) / (active * n_classes)
        )
    else:
        slope_grad = slope_grad.sum(axis=0, keepdims=True)
        penalty_grad = strength * (active - 1) * weights
    loss += 0.5 * np.sum(weights * penalty_grad)
    gradient = np.r_[(slope_grad + penalty_grad).ravel(), basis @ residual.sum(0)]
    return float(loss), gradient


class SharedBiasStackingLR(Method):
    """Fit a shared own-class slope per provider and an unpenalized bias per class.

    Each provider has one StandardScaler mean and scale shared across all class
    coordinates. All class intercepts sum to zero. If a class is missing from
    training, return the fitted distribution on the observed classes and apply
    the benchmark's standard probability floor.
    """

    name = "stacking_shared_bias"
    _diagonal = False

    def __init__(self, c_value=1.0, max_iter=2000):
        if (
            not isinstance(c_value, Real)
            or isinstance(c_value, (bool, np.bool_))
            or not np.isfinite(c_value)
            or c_value <= 0
        ):
            raise ValueError("c_value must be a finite positive number")
        if (
            isinstance(max_iter, bool)
            or not isinstance(max_iter, Integral)
            or max_iter < 1
        ):
            raise ValueError("max_iter must be a positive integer")
        self.c_value, self.max_iter = float(c_value), int(max_iter)

    def fit(self, fit_preds, fit_y, *, task, n_classes):
        values = _provider_values(fit_preds)
        n, classes, providers = values.shape
        if classes != n_classes or task != ("binary" if classes == 2 else "multiclass"):
            raise ValueError("Task and n_classes must match the input class axis")
        labels = np.asarray(fit_y)
        if (
            labels.shape not in ((n,), (n, 1))
            or labels.dtype.kind not in "biuf"
            or not np.isfinite(labels).all()
            or np.any(labels < 0)
            or np.any(labels >= classes)
            or np.any(labels != np.floor(labels))
        ):
            raise ValueError("Supply one integer class index per training row")
        observed, labels, counts = np.unique(
            labels.ravel().astype(np.int64), return_inverse=True, return_counts=True
        )
        active = len(observed)
        scaler = StandardScaler().fit(values.reshape(-1, providers))
        z = _scaled(values, scaler)[:, observed]
        # With exactly two input classes the two own-class slopes on a provider
        # only identify their mean. Use that identifiable shared coordinate.
        diagonal = self._diagonal and classes > 2
        slopes = active if diagonal else 1
        weights = np.zeros((slopes, providers))
        intercept = np.zeros(active)
        nit, objective, gradient = 0, 0.0, 0.0
        if active > 1:
            basis = helmert(active)
            initial = np.r_[weights.ravel(), basis @ np.log(counts)]
            result = minimize(
                _objective,
                initial,
                args=(z, labels, basis, classes, self.c_value, diagonal),
                jac=True,
                method="L-BFGS-B",
                options={
                    "maxiter": self.max_iter,
                    "gtol": 1e-9,
                    "ftol": 1e-14,
                    "maxls": 50,
                },
            )
            objective, grad = _objective(
                result.x, z, labels, basis, classes, self.c_value, diagonal
            )
            gradient = float(np.linalg.norm(grad, ord=np.inf))
            if not np.isfinite(objective) or not np.isfinite(gradient):
                raise RuntimeError(
                    "Class stacking fit returned a non-finite objective or gradient"
                )
            if not result.success and gradient > 1e-8 * n:
                raise RuntimeError(f"Class stacking fit failed: {result.message}")
            weights = result.x[: weights.size].reshape(weights.shape)
            intercept = result.x[weights.size :] @ basis
            nit = int(result.nit)
        super().fit(fit_preds, fit_y, task=task, n_classes=n_classes)
        self.scaler_ = scaler
        self.classes_ = observed
        self.n_providers_ = providers
        self.weights_ = weights
        self.intercept_ = intercept
        self.n_iter_, self.objective_, self.gradient_inf_ = nit, objective, gradient
        return self

    def predict(self, preds):
        if not hasattr(self, "weights_"):
            raise ValueError("Fit the stacker before querying predictions")
        values = _provider_values(preds)
        if values.shape[1:] != (self.n_classes, self.n_providers_):
            raise ValueError("Expected the fitted provider and class axes")
        z = _scaled(values, self.scaler_)[:, self.classes_]
        logits = np.einsum("ncm,cm->nc", z, self.weights_) + self.intercept_
        result = np.zeros(values.shape[:2])
        result[:, self.classes_] = softmax(logits, axis=1)
        return normalize_proba(result)


class DiagonalStackingLR(SharedBiasStackingLR):
    """Fit a separate own-class slope for each provider and class, plus free biases.

    For C observed classes and a single provider with slopes w, let
    P = I - 11'/C. The smallest full multinomial coefficient block giving the
    same probabilities on the provider simplex is P diag(w) P. Its squared
    norm is (1-2/C) sum(w²) + (sum(w)/C)². The regularizer is half that norm
    divided by c_value. Shared slopes reduce the norm to (C-1) w². With A
    observed outputs among C input channels, the analogous rectangular block
    has norm squared (1-1/A-1/C) sum(w²) + sum(w)²/(A C). For two observed
    outputs, double this norm to match sklearn's binary log-odds convention.
    """

    name = "stacking_diagonal"
    _diagonal = True

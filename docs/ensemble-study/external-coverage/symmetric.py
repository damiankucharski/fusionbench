"""Class-permutation-equivariant version of the public raw-probability stacker."""

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from fusionbench.eval.metrics import normalize_proba
from fusionbench.methods.base import Method


def _provider_rows(preds):
    p = np.asarray(preds, dtype=np.float64)
    if p.ndim != 3 or min(p.shape) < 1 or p.shape[2] < 2:
        raise ValueError(
            "Supply [rows, providers, classes] probabilities with at least two classes"
        )
    if (
        not np.isfinite(p).all()
        or np.any(p < 0)
        or np.any(p > 1)
        or not np.allclose(p.sum(axis=2), 1, rtol=0, atol=1e-6)
    ):
        raise ValueError(
            "Each provider must supply finite probabilities summing to one"
        )
    n, m, c = p.shape
    return p.transpose(0, 2, 1).reshape(n * c, m), n, m, c


def _scaled_training(fit_preds, fit_y, task, n_classes, c_value):
    features, n, m, c = _provider_rows(fit_preds)
    if n_classes != c or task != ("binary" if c == 2 else "multiclass"):
        raise ValueError("Task and n_classes must match the input class axis")
    original = np.asarray(fit_y)
    if original.shape not in ((n,), (n, 1)):
        raise ValueError("Expected one class label per row")
    original = original.ravel()
    if (
        original.dtype.kind not in "biuf"
        or not np.isfinite(original).all()
        or np.any(original < 0)
        or np.any(original >= c)
    ):
        raise ValueError("Labels must be class indices")
    y = original.astype(np.int64)
    if not np.array_equal(original, y):
        raise ValueError("Labels must be integer-valued class indices")
    if not np.isfinite(c_value) or c_value <= 0:
        raise ValueError("c_value must be finite and positive")
    # In the full augmented design each original row has total weight one.
    constant = np.all(features == features[0], axis=0)
    if np.any(constant):
        # Avoid negative roundoff in sklearn's variance calculation for a
        # perfectly constant nonzero column (e.g. uniform class probabilities).
        to_fit = features.copy()
        to_fit[:, constant] = 0.0
    else:
        to_fit = features
    scaler = StandardScaler().fit(to_fit, sample_weight=np.full(len(features), 1 / c))
    scaler.mean_[constant] = features[0, constant]
    return scaler.transform(features).reshape(n, c, m), y, scaler


def _scaled_query(preds, scaler, n_classes):
    features, n, m, c = _provider_rows(preds)
    if (m, c) != (scaler.n_features_in_, n_classes):
        raise ValueError("Expected the fitted provider and class axes")
    return scaler.transform(features).reshape(n, c, m)


def _loss_and_gradient(weights, features, y, n_classes, penalty):
    """Summed multinomial log loss and gradient in standardized coordinates."""
    logits = (features @ weights).reshape(len(y), n_classes)
    rows = np.arange(len(y))
    log_probs = logits - logsumexp(logits, axis=1, keepdims=True)
    loss = -log_probs[rows, y].sum() + 0.5 * penalty * (weights @ weights)
    error = np.exp(log_probs)
    error[rows, y] -= 1.0
    return float(loss), features.T @ error.ravel() + penalty * weights


class SymmetricStackingLR(Method):
    """Regularized logistic regression with one learned weight per provider.

    Fitting the public standardized stacker to all simultaneous permutations of
    class coordinates and labels (each weighted 1/C!) gives this objective.
    Its class coefficients in each standardized provider block have the form
    w*(I - 11'/C), whose squared Frobenius norm is (C-1)*w². The shared class
    intercept may be set to zero. For binary sklearn LR the single log-odds
    row is (-w,w), whose squared norm is 2*w². The pooled StandardScaler fits
    each provider separately and uses the same mean and scale for all classes.
    """

    name = "symmetric_lr"

    def __init__(self, c_value=1.0, max_iter=2000):
        self.c_value = c_value
        self.max_iter = max_iter

    def fit(self, fit_preds, fit_y, *, task, n_classes):
        values, y, scaler = _scaled_training(
            fit_preds, fit_y, task, n_classes, self.c_value
        )
        n, c, m = values.shape
        scaled = values.reshape(n * c, m)
        penalty = (2 if c == 2 else c - 1) / self.c_value
        result = minimize(
            _loss_and_gradient,
            np.zeros(m),
            args=(scaled, y, c, penalty),
            method="L-BFGS-B",
            jac=True,
            options={
                "maxiter": self.max_iter,
                "ftol": 1e-14,
                "gtol": 1e-9,
                "maxls": 50,
            },
        )
        if (
            not result.success
            or not np.isfinite(result.fun)
            or not np.isfinite(result.x).all()
        ):
            raise RuntimeError(f"Symmetric logistic fit failed: {result.message}")
        super().fit(fit_preds, fit_y, task=task, n_classes=n_classes)
        self.scaler_ = scaler
        self.weights_ = result.x
        self.scale_ = scaler.scale_
        self.n_providers_ = m
        self.n_iter_ = result.nit
        self.objective_ = float(result.fun)
        self.gradient_inf_ = float(np.linalg.norm(result.jac, ord=np.inf))
        return self

    def predict(self, preds):
        if not hasattr(self, "weights_"):
            raise ValueError("Fit the stacker before querying predictions")
        z = _scaled_query(preds, self.scaler_, self.n_classes)
        return normalize_proba(softmax(z @ self.weights_, axis=1))


class PooledScaleStackingLR(Method):
    """Public StackP's full logistic head on the symmetric comparator's features."""

    name = "pooled_lr"

    def __init__(self, c_value=1.0, max_iter=2000):
        self.c_value = c_value
        self.max_iter = max_iter

    def fit(self, fit_preds, fit_y, *, task, n_classes):
        z, y, scaler = _scaled_training(fit_preds, fit_y, task, n_classes, self.c_value)
        design = z.transpose(0, 2, 1).reshape(len(y), -1)
        # Match the public stacker's solver, intercept, default tolerance and penalty.
        model = LogisticRegression(
            C=self.c_value, max_iter=self.max_iter, solver="lbfgs"
        ).fit(design, y)
        super().fit(fit_preds, fit_y, task=task, n_classes=n_classes)
        self.scaler_ = scaler
        self.scale_ = scaler.scale_
        self.model_ = model
        self.classes_ = model.classes_
        self.n_iter_ = int(model.n_iter_.max())
        return self

    def predict(self, preds):
        if not hasattr(self, "model_"):
            raise ValueError("Fit the stacker before querying predictions")
        z = _scaled_query(preds, self.scaler_, self.n_classes)
        probability = self.model_.predict_proba(
            z.transpose(0, 2, 1).reshape(len(z), -1)
        )
        full = np.zeros((len(z), self.n_classes))
        full[:, self.classes_] = probability
        return normalize_proba(full)

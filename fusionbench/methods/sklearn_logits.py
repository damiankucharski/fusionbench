"""Export scikit-learn logistic experts as affine softmax logits on shared features."""

import operator

import numpy as np
from scipy.sparse import issparse
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted


def _local_logits(estimator):
    scaler = None
    if isinstance(estimator, Pipeline):
        steps = [step for _, step in estimator.steps]
        if len(steps) != 2 or not isinstance(steps[0], StandardScaler):
            raise ValueError("Only a StandardScaler followed by LogisticRegression is supported")
        scaler, estimator = steps
        check_is_fitted(scaler, attributes=["n_features_in_"])
    if isinstance(estimator, OneVsRestClassifier):
        raise TypeError("One-vs-rest probability normalization is not a joint softmax")
    if not isinstance(estimator, LogisticRegression):
        raise TypeError(
            "Expected a fitted LogisticRegression or StandardScaler/LogisticRegression pipeline"
        )
    check_is_fitted(estimator, attributes=["coef_", "intercept_", "classes_", "n_features_in_"])
    classes = estimator.classes_
    # In versions supporting explicit 'multinomial', binary predict_proba used
    # softmax([-z,z]). The standard binary path uses the single logistic z.
    mode = estimator.get_params(deep=False).get("multi_class")
    if len(classes) > 2 and (mode in ("ovr", "warn") or estimator.solver == "liblinear"):
        raise ValueError(
            "Multiclass one-vs-rest probabilities cannot be exported as affine softmax logits"
        )
    coef = estimator.coef_
    w = np.asarray(coef.toarray() if issparse(coef) else coef, dtype=float)
    b = np.asarray(estimator.intercept_, dtype=float)
    expected_rows = 1 if len(classes) == 2 else len(classes)
    if w.shape != (expected_rows, estimator.n_features_in_) or b.shape != (expected_rows,):
        raise ValueError("The logistic model has incompatible fitted class or feature dimensions")
    if scaler is not None:
        if scaler.n_features_in_ != w.shape[1]:
            raise ValueError("The fitted scaler and logistic model use different feature dimensions")
        if scaler.with_std:
            w = w / scaler.scale_
        if scaler.with_mean:
            b = b - w @ scaler.mean_
    if len(classes) == 2:
        w = np.concatenate((-w if mode == "multinomial" else np.zeros_like(w), w))
        b = np.r_[-b if mode == "multinomial" else np.zeros_like(b), b]
    if not np.isfinite(w).all() or not np.isfinite(b).all():
        raise ValueError("The fitted logits must have finite weights and intercepts")
    return w, b, classes


def _feature_view(indices, local_width):
    if indices is None:
        return np.arange(local_width), local_width
    idx = np.asarray(indices)
    if idx.ndim != 1 or idx.dtype.kind not in "biu":
        raise ValueError(
            "Each feature view must be a one-dimensional integer index array or boolean mask"
        )
    specified_width = None
    if idx.dtype.kind == "b":
        specified_width = len(idx)
        idx = np.flatnonzero(idx)
    if len(idx) != local_width or np.any(idx < 0):
        raise ValueError(
            "Each feature view must match the fitted feature dimension and use nonnegative indices"
        )
    return idx, specified_width


def sklearn_affine_logits(estimators, *, feature_indices=None, n_features=None):
    """Return (K×C×D weights, K×C intercepts, class labels) for shared raw inputs.

    Each feature_indices entry specifies the raw columns, in fitted order, of
    that expert. An entry of None uses the full raw input. Repeated columns add
    their coefficients. Provide n_features to retain unused trailing columns.
    Class order follows the first expert and every expert must contain that set.
    """
    estimators = list(estimators)
    if not estimators:
        raise ValueError("Supply at least one fitted estimator")
    views = [None] * len(estimators) if feature_indices is None else list(feature_indices)
    if len(views) != len(estimators):
        raise ValueError("Supply one feature view per estimator")
    local = [_local_logits(estimator) for estimator in estimators]
    views = [_feature_view(view, w.shape[1]) for view, (w, _, _) in zip(views, local)]
    if n_features is None:
        width = max(explicit if explicit is not None else int(idx.max()) + 1 for idx, explicit in views)
    else:
        if isinstance(n_features, (bool, np.bool_)):
            raise ValueError("n_features must be a positive integer")
        width = operator.index(n_features)
    if width <= 0 or any(
        (explicit is not None and explicit != width) or int(idx.max()) >= width
        for idx, explicit in views
    ):
        raise ValueError("The feature views are incompatible with the shared raw feature dimension")

    classes = np.asarray(local[0][2]).copy()
    weights = np.zeros((len(local), len(classes), width))
    intercepts = np.empty(weights.shape[:2])
    for k, ((w, b, labels), (idx, _)) in enumerate(zip(local, views)):
        positions = {label: i for i, label in enumerate(labels)}
        if len(positions) != len(classes) or any(label not in positions for label in classes):
            raise ValueError("All fitted estimators must contain the same class labels")
        order = [positions[label] for label in classes]
        np.add.at(weights[k].T, idx, w[order].T)
        intercepts[k] = b[order]
    return weights, intercepts, classes

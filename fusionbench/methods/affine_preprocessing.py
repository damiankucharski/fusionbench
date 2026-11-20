"""Compose fitted standardization, PCA, and affine readouts in raw coordinates."""

import numpy as np
from sklearn.utils.validation import check_is_fitted


def _finite_array(value, name):
    array = np.asarray(value)
    if array.dtype.kind not in "iuf" or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite real numbers")
    return array


def fold_standardized_pca_affines(scaler, pca, weights, bias):
    """Return raw weights and biases for affine predictions after StandardScaler and PCA.

    The last axis of weights indexes retained principal components. Leading axes
    index readouts, and bias can be scalar or match their shape. For a matrix of
    weights, x @ raw_weights.T + raw_bias evaluates all readouts. Like sklearn's
    PCA.transform, whitening clips each component's standard deviation to the
    epsilon of the fitted variance dtype. Algebraic folding changes the order
    of floating-point operations, especially for nearly singular components.
    """
    check_is_fitted(scaler, attributes=["n_features_in_"])
    check_is_fitted(pca, attributes=["components_", "mean_"])
    components = _finite_array(pca.components_, "PCA components")
    mean = _finite_array(pca.mean_, "PCA mean")
    n_features = scaler.n_features_in_
    if (
        components.ndim != 2
        or not len(components)
        or components.shape[1] != n_features
        or mean.shape != (n_features,)
    ):
        raise ValueError("scaler and PCA must have matching fitted input dimensions")

    weights = _finite_array(weights, "weights").astype(np.float64, copy=False)
    bias = _finite_array(bias, "bias")
    if weights.ndim == 0 or weights.shape[-1] != len(components):
        raise ValueError(
            "the last axis of weights must match the retained PCA dimension"
        )
    if bias.shape not in ((), weights.shape[:-1]):
        raise ValueError("bias must be a scalar or have one entry per readout")

    if pca.whiten:
        variance = _finite_array(pca.explained_variance_, "PCA explained variance")
        if (
            variance.shape != (len(components),)
            or variance.dtype.kind != "f"
            or (variance < 0).any()
        ):
            raise ValueError(
                "expected one nonnegative floating-point variance per PCA component"
            )
        weights = weights / np.maximum(np.sqrt(variance), np.finfo(variance.dtype).eps)

    projected = weights @ components
    raw_bias = bias - weights @ (mean @ components.T)
    if scaler.with_std:
        scale = _finite_array(scaler.scale_, "scaler scale")
        if scale.shape != (n_features,) or (scale <= 0).any():
            raise ValueError("expected one positive fitted scale per raw feature")
        projected = projected / scale
    if scaler.with_mean:
        center = _finite_array(scaler.mean_, "scaler mean")
        if center.shape != (n_features,):
            raise ValueError("expected one fitted mean per raw feature")
        raw_bias = raw_bias - projected @ center
    return projected, raw_bias

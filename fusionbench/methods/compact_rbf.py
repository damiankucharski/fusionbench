"""Fold a standardized Nyström logistic predictor into a Gaussian radial network."""

from dataclasses import dataclass

import numpy as np
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted


def _finite_real(value, name):
    array = np.asarray(value)
    if array.dtype.kind not in "iuf" or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite real values")
    return array.astype(np.float64, copy=False)


@dataclass(frozen=True)
class CompactRBF:
    """One weight per center in standardized coordinates and a binary logistic output."""

    centers: np.ndarray
    weights: np.ndarray
    bias: float
    gamma: float
    mean: np.ndarray
    scale: np.ndarray

    def predict_positive(self, raw_X):
        """Evaluate the fitted readout's second class on a dense raw feature matrix."""
        raw = _finite_real(raw_X, "raw_X")
        if raw.ndim != 2 or raw.shape[1] != self.centers.shape[1]:
            raise ValueError("raw_X must be a matrix with the fitted feature count")
        standardized = (raw - self.mean) / self.scale
        responses = rbf_kernel(standardized, self.centers, gamma=self.gamma)
        logit = responses @ self.weights + self.bias
        return np.exp(-np.logaddexp(0.0, -logit))


def compile_standardized_rbf(scaler, basis, readout):
    """Compile fitted StandardScaler, RBF Nystroem and binary LogisticRegression.

    The compact predictor consumes dense raw inputs. It stores normalized landmark
    coordinates, standardization, and one readout coefficient per landmark. The
    Nyström normalization matrix and the original estimators can then be discarded.
    """
    if not isinstance(scaler, StandardScaler) or not isinstance(basis, Nystroem):
        raise TypeError("expected a StandardScaler and Nystroem basis")
    if not isinstance(readout, LogisticRegression):
        raise TypeError("expected a LogisticRegression readout")
    check_is_fitted(scaler, attributes=["n_features_in_"])
    check_is_fitted(basis, attributes=["components_", "normalization_"])
    check_is_fitted(readout, attributes=["coef_", "intercept_", "classes_"])
    if basis.kernel != "rbf":
        raise ValueError("the fitted basis must use the RBF kernel")
    if np.shape(readout.classes_) != (2,):
        raise ValueError("the readout must have two classes")

    n_features = scaler.n_features_in_
    centers = basis.components_
    if hasattr(centers, "toarray"):
        centers = centers.toarray()
    centers = _finite_real(centers, "centers")
    if centers.ndim != 2 or centers.shape[1] != n_features or not len(centers):
        raise ValueError("centers and scaler must have the same fitted feature count")
    n_centers = len(centers)
    normalization = _finite_real(basis.normalization_, "Nyström normalization")
    coef = _finite_real(readout.coef_, "readout coefficients")
    intercept = _finite_real(readout.intercept_, "readout intercept")
    if (
        normalization.shape != (n_centers, n_centers)
        or coef.shape != (1, n_centers)
        or intercept.shape != (1,)
    ):
        raise ValueError("basis and readout dimensions do not match")

    mean = (
        _finite_real(scaler.mean_, "scaler mean")
        if scaler.with_mean
        else np.zeros(n_features)
    )
    scale = (
        _finite_real(scaler.scale_, "scaler scale")
        if scaler.with_std
        else np.ones(n_features)
    )
    if (
        mean.shape != (n_features,)
        or scale.shape != (n_features,)
        or (scale <= 0).any()
    ):
        raise ValueError("invalid fitted scaler dimensions or feature scale")
    gamma = basis.gamma
    if gamma is None:
        gamma = (basis.kernel_params or {}).get("gamma")
    gamma = 1.0 / n_features if gamma is None else float(gamma)
    if not np.isfinite(gamma) or gamma < 0:
        raise ValueError("expected a finite nonnegative Gaussian gamma")

    # Older sklearn versions support an explicitly multinomial binary softmax
    # over [-decision, decision], which doubles the positive-class log odds.
    factor = 2.0 if getattr(readout, "multi_class", None) == "multinomial" else 1.0
    weights = normalization.T @ (factor * coef[0])
    return CompactRBF(
        centers.copy(),
        weights,
        factor * float(intercept[0]),
        gamma,
        mean.copy(),
        scale.copy(),
    )

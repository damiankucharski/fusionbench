"""Calibrated base-model prediction generators.

Binary foundation (calibrated by construction)
----------------------------------------------
Draw the shared true label ``y in {0,1}``. For a model with *signal strength*
``s >= 0`` define a latent evidence

    t = (2y - 1) * s + noise,   noise ~ N(0, 1)

Under this well-specified Gaussian model (equal class prior) the exact Bayes
posterior is ``P(y=1 | t) = sigmoid(2 s t)``. A model that *outputs that
posterior* is therefore perfectly calibrated. Its accuracy at threshold 0.5 is
``Phi(s)``, so ``s = Phi^{-1}(accuracy)`` gives direct control.

Correlation between models is introduced by sharing part of the noise while
keeping each model's marginal ``noise ~ N(0,1)``, which preserves single-model
calibration.

Multiclass uses a logit-normal construction (approximately calibrated; verified
empirically by the reliability/calibration checks).
"""

from __future__ import annotations

from statistics import NormalDist

import numpy as np

_NORM = NormalDist()


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def acc_to_strength(accuracy: float) -> float:
    """Signal strength ``s`` giving a calibrated binary model of this accuracy."""
    accuracy = float(np.clip(accuracy, 0.5 + 1e-6, 1 - 1e-9))
    return _NORM.inv_cdf(accuracy)


def to_binary_proba(p1: np.ndarray) -> np.ndarray:
    """Map P(class=1) vector ``[N]`` to a ``[N, 2]`` proba matrix."""
    p1 = np.clip(p1, 0.0, 1.0)
    return np.stack([1.0 - p1, p1], axis=1)


def binary_calibrated(
    y: np.ndarray,
    accuracy: float,
    rng: np.random.Generator,
    *,
    shared_noise: np.ndarray | None = None,
    shared_weight: float = 0.0,
    active: np.ndarray | None = None,
) -> np.ndarray:
    """Return calibrated ``P(y=1)`` of shape ``[N]``.

    ``shared_noise`` (a pre-drawn ``N(0,1)`` vector) mixed in with weight
    ``shared_weight in [0,1]`` correlates this model's errors with others while
    keeping its marginal noise standard normal.

    Where ``active`` is False the model *defers* (outputs ~0.5), modelling a
    specialist that is competent only on its region of expertise.
    """
    s = acc_to_strength(accuracy)
    n = y.shape[0]
    indep = rng.standard_normal(n)
    if shared_noise is not None and shared_weight > 0.0:
        w = float(np.clip(shared_weight, 0.0, 1.0))
        noise = np.sqrt(1.0 - w) * indep + np.sqrt(w) * shared_noise
    else:
        noise = indep
    t = (2.0 * y - 1.0) * s + noise
    p1 = sigmoid(2.0 * s * t)
    if active is not None:
        defer = ~active.astype(bool)
        # Tiny jitter around 0.5 keeps "deferral" close to the decision boundary
        # so confidence-based routers correctly ignore it.
        p1 = p1.copy()
        p1[defer] = 0.5 + 0.001 * rng.standard_normal(int(defer.sum()))
    return np.clip(p1, 1e-6, 1 - 1e-6)


def sharpen(p1: np.ndarray, k: float) -> np.ndarray:
    """Re-scale confidence in logit space. ``k>1`` overconfident, ``k<1`` under."""
    return sigmoid(k * logit(p1))


def random_binary(y: np.ndarray, rng: np.random.Generator, spread: float = 0.05) -> np.ndarray:
    """A near-useless model: probabilities clustered around 0.5, ~chance accuracy."""
    return np.clip(0.5 + spread * rng.standard_normal(y.shape[0]), 1e-6, 1 - 1e-6)


# --------------------------------------------------------------------------- #
# Multiclass (logit-normal; approximately calibrated)
# --------------------------------------------------------------------------- #
def mc_strength(accuracy: float, n_classes: int) -> float:
    """Heuristic logit boost giving ~``accuracy`` top-1 for ``n_classes``.

    Calibrated exactly only in the binary limit; multiclass calibration is
    verified empirically. The mapping is monotone so scenarios can tune it.
    """
    base = acc_to_strength(min(accuracy, 0.999))
    return base * (1.0 + 0.35 * np.log(max(n_classes, 2)))


def multiclass_calibrated(
    y: np.ndarray,
    accuracy: float,
    n_classes: int,
    rng: np.random.Generator,
    *,
    shared_noise: np.ndarray | None = None,
    shared_weight: float = 0.0,
    active: np.ndarray | None = None,
    class_strength: np.ndarray | None = None,
) -> np.ndarray:
    """Return a ``[N, n_classes]`` softmax proba with controllable top-1 accuracy.

    ``class_strength`` (length ``n_classes``) lets a model be competent only on
    certain classes (class-conditional competence). ``active`` toggles per-sample
    deferral (uniform output).
    """
    n = y.shape[0]
    s = mc_strength(accuracy, n_classes)
    indep = rng.standard_normal((n, n_classes))
    if shared_noise is not None and shared_weight > 0.0:
        w = float(np.clip(shared_weight, 0.0, 1.0))
        noise = np.sqrt(1.0 - w) * indep + np.sqrt(w) * shared_noise
    else:
        noise = indep
    boost = np.zeros((n, n_classes))
    if class_strength is None:
        boost[np.arange(n), y] = s
    else:
        boost[np.arange(n), y] = s * class_strength[y]
    z = boost + noise
    if active is not None:
        defer = ~active.astype(bool)
        z[defer] = 0.01 * rng.standard_normal((int(defer.sum()), n_classes))
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------- #
# Analytic Bayes oracles (derived from the DGP; method-independent ceilings)
# --------------------------------------------------------------------------- #
def binary_independent_oracle(preds: np.ndarray, prior_logodds: float = 0.0) -> np.ndarray:
    """Bayes posterior for conditionally-independent calibrated binary models.

    For calibrated models ``logit(p_m)`` equals the per-model log-likelihood
    ratio, so the optimal combination is the **sum of logits** (not the mean of
    probabilities). ``preds`` is ``[N, K, 2]``.
    """
    p1 = preds[:, :, 1]
    lo = prior_logodds + logit(p1).sum(axis=1)
    return to_binary_proba(sigmoid(lo))


def binary_gaussian_oracle(
    t: np.ndarray, s: np.ndarray, sigma_inv: np.ndarray, prior_logodds: float = 0.0
) -> np.ndarray:
    """Bayes posterior for jointly-Gaussian latent evidence ``t`` ``[N, K]``.

    Under ``t | y ~ N((2y-1) s, Sigma)`` the optimal log-odds is
    ``2 s^T Sigma^{-1} t`` (Gaussian linear discriminant), correctly
    down-weighting correlated (redundant) evidence.
    """
    v = 2.0 * sigma_inv @ np.asarray(s, dtype=float)  # [K]
    lo = prior_logodds + t @ v
    return to_binary_proba(sigmoid(lo))


def multiclass_independent_oracle(
    preds: np.ndarray,
    strengths: np.ndarray | None = None,
    log_prior: np.ndarray | None = None,
    *,
    min_probability: float = 1e-12,
) -> np.ndarray:
    """Bayes posterior for conditionally-independent logit-normal models.

    Optimal combination is ``softmax_c( log_prior_c + sum_m s_m log p_{m,c} )``
    — the (strength-weighted) product of probabilities. ``preds`` is ``[N, K, C]``.
    The default floor preserves the earlier behavior of the generic combiner.
    """
    logp = np.log(np.clip(preds, min_probability, 1.0))  # [N, K, C]
    if strengths is None:
        agg = logp.sum(axis=1)
    else:
        agg = np.tensordot(np.asarray(strengths, dtype=float), logp, axes=(0, 1))
    if log_prior is not None:
        agg = agg + log_prior
    agg = agg - agg.max(axis=1, keepdims=True)
    e = np.exp(agg)
    return e / e.sum(axis=1, keepdims=True)


def multiclass_gaussian_oracle(
    preds: np.ndarray, accuracy: float, *, shared_weight: float = 0.0
) -> np.ndarray:
    """Posterior for equally strong logit-normal providers with shared Gaussian noise.

    Given class c, each provider logit vector has mean s times the c-th
    unit vector. The observable log ratios discard only a class-independent
    row offset. Multiplying their sum by s / (1 + (K-1) rho) is Gaussian
    discriminant analysis with the generator's equicorrelation covariance.
    At rho=1 all provider outputs coincide and the expression reduces to
    the single-provider posterior.
    """
    preds = np.asarray(preds, dtype=np.float64)
    _, k, classes = preds.shape
    rho = float(np.clip(shared_weight, 0.0, 1.0))
    strength = mc_strength(accuracy, classes) / (1 + (k - 1) * rho)
    weights = np.full(k, strength)
    return multiclass_independent_oracle(
        preds, strengths=weights, min_probability=np.nextafter(0.0, 1.0)
    )


def recover_t(p1: np.ndarray, accuracy: float) -> np.ndarray:
    """Recover latent evidence ``t`` from a calibrated binary output ``p1``.

    Inverse of :func:`binary_calibrated`: ``logit(p1) = 2 s t``.
    """
    s = acc_to_strength(accuracy)
    return logit(p1) / (2.0 * s)

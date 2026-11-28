"""Marginalize one provider in a fitted fusion model using fixed source reports."""

from numbers import Integral, Real

import numpy as np

from fusionbench.eval.suite_point_gp import _full


def _probabilities(values, ndim, name):
    p = np.asarray(values, dtype=float)
    if p.ndim != ndim or min(p.shape, default=0) < 1 or p.shape[-1] < 2:
        raise ValueError(
            f"{name} must have {ndim} nonempty axes and at least two classes"
        )
    if not np.all((p >= 0) & (p <= 1)) or not np.allclose(
        p.sum(axis=-1), 1, atol=1e-6, rtol=0
    ):
        raise ValueError(
            f"{name} must contain finite class probabilities summing to one"
        )
    return p


def _pool(x, provider):
    pool = _probabilities(_full(x), 3, "Provider predictions")
    if (
        isinstance(provider, bool)
        or not isinstance(provider, Integral)
        or not 0 <= provider < pool.shape[1]
    ):
        raise ValueError("provider must be a valid zero-based provider index")
    return pool


def source_donors(x, provider, count=8):
    """Select equiprobable mid-quantile ranks from one provider's source reports.

    Sort whole reports lexicographically by their original class columns,
    starting with class zero. Select floor((j + 0.5) * n / count) for each j.
    Repeated reports retain their source frequency. When count exceeds n,
    donor rows necessarily repeat. Returns [count, class] in the original
    class order. Supply training data, and reuse this result at prediction.
    """
    if isinstance(count, bool) or not isinstance(count, Integral) or count < 1:
        raise ValueError("count must be a positive integer")
    reports = _pool(x, provider)[:, provider, :]
    # np.lexsort treats its last key as the primary key.
    order = np.lexsort(reports.T[::-1])
    positions = ((2 * np.arange(count) + 1) * len(reports)) // (2 * count)
    return reports[order[positions]].copy()


def marginal_prediction(predict, x, provider, donors):
    """Average predictions when one provider's report is replaced by each donor.

    predict receives a full [row, provider, class] tensor and returns full
    [row, class] probabilities. Each observed source donor vector is supplied
    at every query row in one call. Other provider columns remain paired
    within the query row. The returned array has shape [row, class].
    """
    if not callable(predict):
        raise TypeError("predict must be a callable returning class probabilities")
    pool = _pool(x, provider)
    reports = _probabilities(donors, 2, "Donors")
    if pool.shape[2] != reports.shape[1]:
        raise ValueError("Donor class axis must match the query pool")
    total = np.zeros((len(pool), pool.shape[2]))
    for report in reports:
        replaced = pool.copy()
        replaced[:, provider, :] = report
        forecast = _probabilities(predict(replaced), 2, "Fused predictions")
        if forecast.shape != total.shape:
            raise ValueError("Expected one fused class distribution per query row")
        total += forecast
    return total / len(reports)


def limit_reliance(original, marginal, strength):
    """Blend aligned predictions, with zero retaining the original fitted forecast."""
    if (
        isinstance(strength, bool)
        or not isinstance(strength, Real)
        or not 0 <= strength <= 1
    ):
        raise ValueError("strength must be a finite number between zero and one")
    original = _probabilities(original, 2, "Original predictions")
    marginal = _probabilities(marginal, 2, "Marginal predictions")
    if original.shape != marginal.shape:
        raise ValueError("Original and marginal prediction shapes must agree")
    return (1 - strength) * original + strength * marginal

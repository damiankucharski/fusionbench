"""Paired label-prior shifts with fixed class-conditional committee predictions."""

from __future__ import annotations

from numbers import Integral

import numpy as np
from scipy.special import expit, log_ndtr

from fusionbench import __version__
from fusionbench.scenarios import calibrated as cal
from fusionbench.scenarios import get_scenario
from fusionbench.types import Dataset

_SUPPORTED = ("committee", "multiclass_committee", "saturated")


def _integer(value, name, minimum):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Integral)
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _prior(values, classes, name):
    prior = np.asarray(values)
    if prior.shape != (classes,) or prior.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain {classes} class probabilities")
    prior = prior.astype(np.float64)
    if (
        not np.isfinite(prior).all()
        or (prior < 0).any()
        or not np.isclose(prior.sum(), 1.0, rtol=0, atol=1e-7)
    ):
        raise ValueError(
            f"{name} must contain finite, nonnegative probabilities summing to one"
        )
    return prior / prior.sum()


def _binary_posterior(positive, accuracy, prior):
    """Account for the probability mass at both clipping limits in binary_calibrated."""
    if np.any(prior == 0):
        return np.broadcast_to(prior, (len(positive), 2)).copy()
    low, high = 1e-6, 1 - 1e-6
    llr = np.log(positive) - np.log1p(-positive)
    strength = cal.acc_to_strength(accuracy)
    for boundary in (low, high):
        observed = positive == boundary
        if observed.any():
            threshold = (np.log(boundary) - np.log1p(-boundary)) / (2 * strength)
            if boundary == low:
                llr[observed] = log_ndtr(threshold - strength) - log_ndtr(
                    threshold + strength
                )
            else:
                llr[observed] = log_ndtr(strength - threshold) - log_ndtr(
                    -strength - threshold
                )
    return cal.to_binary_proba(
        expit(llr.sum(axis=1) + np.log(prior[1]) - np.log(prior[0]))
    )


def _draw(scenario_id, scenario, prior, n, seed, fold, split):
    labels_rng = np.random.default_rng([seed, fold, split, 0])
    noise_rng = np.random.default_rng([seed, fold, split, 1])
    # Shared uniforms couple different priors at the same row.  Generating labels
    # separately leaves the conditional noise untouched, including at zero mass.
    cdf = np.cumsum(prior)
    cdf[np.flatnonzero(prior)[-1] :] = 1.0
    labels = np.searchsorted(cdf, labels_rng.random(n), side="right")
    if scenario_id == "committee":
        positive = np.column_stack(
            [
                cal.binary_calibrated(labels, scenario.accuracy, noise_rng)
                for _ in range(scenario.k)
            ]
        )
        return (
            np.stack((1 - positive, positive), axis=-1),
            labels,
            _binary_posterior(positive, scenario.accuracy, prior),
        )

    c = scenario.n_classes
    rho = scenario.corr if scenario_id == "saturated" else 0.0
    shared = noise_rng.standard_normal((n, c)) if scenario_id == "saturated" else None
    preds = np.stack(
        [
            cal.multiclass_calibrated(
                labels,
                scenario.accuracy,
                c,
                noise_rng,
                shared_noise=shared,
                shared_weight=rho,
            )
            for _ in range(scenario.k)
        ],
        axis=1,
    )
    weight = cal.mc_strength(scenario.accuracy, c) / (
        1 + (scenario.k - 1) * np.clip(rho, 0, 1)
    )
    log_prior = np.full(c, -np.inf)
    np.log(prior, out=log_prior, where=prior > 0)
    posterior = cal.multiclass_independent_oracle(
        preds,
        strengths=np.full(scenario.k, weight),
        log_prior=log_prior,
        min_probability=np.nextafter(0.0, 1.0),
    )
    return preds, labels, posterior


def generate_prior_shift(
    scenario_id: str,
    *,
    fit_prior,
    test_prior,
    n_fit: int = 512,
    n_test: int = 2048,
    seed: int = 0,
    fold: int = 0,
    **scenario_params,
) -> Dataset:
    """Generate independent fit/test splits with specified class probabilities.

    Supported IDs are committee, multiclass_committee and saturated. Priors are
    vectors in class-index order. Keep seed, fold, sizes and scenario parameters
    fixed to pair different priors; change fold to draw independent replicates.
    Each split's oracle conditions only on that split's observed provider scores.
    """
    if scenario_id not in _SUPPORTED:
        raise ValueError(f"Prior shift supports {', '.join(_SUPPORTED)}")
    seed, fold = _integer(seed, "seed", 0), _integer(fold, "fold", 0)
    n_fit, n_test = _integer(n_fit, "n_fit", 1), _integer(n_test, "n_test", 1)
    scenario = get_scenario(scenario_id, **scenario_params)
    _integer(scenario.k, "k", 1)
    classes = _integer(scenario.n_classes, "n_classes", 2)
    if not np.isfinite(scenario.accuracy) or (
        scenario_id == "saturated" and not np.isfinite(scenario.corr)
    ):
        raise ValueError("accuracy and correlation must be finite")
    fit_prior = _prior(fit_prior, classes, "fit_prior")
    test_prior = _prior(test_prior, classes, "test_prior")
    fit_x, fit_y, fit_q = _draw(scenario_id, scenario, fit_prior, n_fit, seed, fold, 0)
    test_x, test_y, test_q = _draw(
        scenario_id, scenario, test_prior, n_test, seed, fold, 1
    )
    return Dataset(
        name=f"{scenario.name}_prior_shift",
        task=scenario.task,
        n_classes=classes,
        model_ids=scenario.model_ids(),
        fit_preds=fit_x,
        fit_y=fit_y,
        test_preds=test_x,
        test_y=test_y,
        fit_oracle=fit_q,
        test_oracle=test_q,
        oracle_description="Posterior from observed committee probabilities and the generating split prior",
        meta={
            **scenario._registry_metadata,
            "scenario": type(scenario).__name__,
            "protocol": "prior_shift_v1",
            "fusionbench_version": __version__,
            "seed": seed,
            "fold": fold,
            "n_fit": n_fit,
            "n_test": n_test,
            "primary_metric": "brier",
            "fit_prior": fit_prior.tolist(),
            "test_prior": test_prior.tolist(),
            "fit_class_counts": np.bincount(fit_y, minlength=classes).tolist(),
            "test_class_counts": np.bincount(test_y, minlength=classes).tolist(),
        },
    )

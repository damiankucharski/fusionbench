"""Paired sampling audits of two fixed binary predictions on the same finite rows."""

from functools import lru_cache
from numbers import Integral

import numpy as np
from scipy.stats import beta

from fusionbench.eval.paired_brier import (
    _probabilities,
    deterministic_gap_requests,
    sharp_pair_bounds,
)

PROPOSALS = ("uniform", "gap")


@lru_cache(maxsize=64)
def _binomial_bounds(n):
    """95% equal-tailed Clopper–Pearson endpoints for every possible success count."""
    success = np.arange(n + 1)
    alpha = (1 - 0.95) / 2
    lo = np.empty(n + 1)
    hi = np.empty(n + 1)
    lo[0] = 0
    hi[-1] = 1
    lo[1:] = beta.ppf(alpha, success[1:], n - success[1:] + 1)
    hi[:-1] = beta.ppf(1 - alpha, success[:-1] + 1, n - success[:-1])
    return lo, hi


def _unit_draws(seed, repetitions, maximum):
    if isinstance(seed, np.random.SeedSequence):
        # Reusing a caller's seed sequence produces the same results.
        seed = np.random.SeedSequence(
            seed.entropy, spawn_key=seed.spawn_key, pool_size=seed.pool_size
        )
    else:
        seed = np.random.SeedSequence(seed)
    return np.stack(
        [
            np.random.default_rng(restart).random(maximum)
            for restart in seed.spawn(repetitions)
        ]
    )


def audit_fixed_pair(
    f, g, world_labels, *, seed, draw_budgets=(8, 32, 128), repetitions=64
):
    """Audit mean((f−Y)² − (g−Y)²) for each fixed, separately labeled world.

    Proposal axis 0 is centered uniform and axis 1 is gap proportional.
    The estimates have axes [proposal,world,repetition,budget]; the exact
    95% binomial intervals pertain to the gap proposal and use the last three
    axes. Unique paid row counts omit the shared world axis. Rows with identical
    prediction vectors can carry distinct labels and retain separate indices.

    Draws are with replacement. In each repetition both proposals map the same
    uniforms to rows; budgets take prefixes, and every world shares these rows.
    For all-equal predictions the returned index array has final dimension zero.
    """
    p, q = _probabilities(f, g)
    labels = np.asarray(world_labels)
    if (
        labels.ndim != 2
        or not labels.shape[0]
        or labels.shape[1] != len(p)
        or not np.isin(labels, [0, 1]).all()
    ):
        raise ValueError("Expected a nonempty [world,row] array of binary labels")
    requested = tuple(draw_budgets)
    if (
        not requested
        or any(
            isinstance(k, bool) or not isinstance(k, Integral) or k <= 0
            for k in requested
        )
        or len(set(requested)) != len(requested)
    ):
        raise ValueError("Draw budgets must be distinct positive integers")
    if (
        isinstance(repetitions, bool)
        or not isinstance(repetitions, Integral)
        or repetitions < 1
    ):
        raise ValueError("Repetitions must be a positive integer")
    ks = np.asarray(requested, dtype=int)
    signs = 1.0 - 2.0 * labels
    delta = p - q
    magnitude = np.abs(delta)
    center = float(np.mean(delta * (p + q - 1.0)))
    mean_gap = float(magnitude.mean())
    residual = delta * signs
    # Both means use the same fixed pool. The gap proposal sees a two-point
    # residual ±mean_gap, with probabilities proportional to the signed masses.
    positives = np.maximum(residual, 0).mean(axis=1)
    negatives = np.maximum(-residual, 0).mean(axis=1)
    variance = np.stack((residual.var(axis=1), 4 * positives * negatives))
    worlds = len(labels)
    shape = (2, worlds, repetitions, len(ks))
    out = {
        "draw_budgets": ks,
        "center": center,
        "mean_gap": mean_gap,
        "true_gap": np.mean((p - labels) ** 2 - (q - labels) ** 2, axis=1),
        "exact_variance": variance[:, :, None] / ks,
        "estimate": np.zeros(shape),
        "interval_lower": np.zeros(shape[1:]),
        "interval_upper": np.zeros(shape[1:]),
        "n_unique": np.zeros((2, repetitions, len(ks)), dtype=int),
        "det_budgets": np.r_[0, ks],
        "det_n_unique": np.zeros(len(ks) + 1, dtype=int),
        "det_lower": np.zeros((worlds, len(ks) + 1)),
        "det_upper": np.zeros((worlds, len(ks) + 1)),
    }
    if mean_gap == 0:
        out["draw_indices"] = np.empty((2, repetitions, 0), dtype=int)
        return out

    # Excluding zero-mass rows keeps the gap schedule valid even when a uniform
    # equals zero or the cumulative probabilities have repeated endpoints.
    maximum = int(ks.max())
    uniforms = _unit_draws(seed, repetitions, maximum)
    active = np.flatnonzero(magnitude)
    cdf = np.cumsum(magnitude[active] / magnitude[active].sum())
    cdf /= cdf[-1]
    rows = np.stack(
        (
            (uniforms * len(p)).astype(int),
            active[np.searchsorted(cdf, uniforms, side="right")],
        )
    )
    out["draw_indices"] = rows
    out["estimate"][0] = (
        center + np.cumsum(residual[:, rows[0]], axis=-1)[..., ks - 1] / ks
    )
    successes = np.cumsum(residual[:, rows[1]] > 0, axis=-1)[..., ks - 1]
    out["estimate"][1] = center + mean_gap * (2.0 * successes / ks - 1)
    for j, k in enumerate(ks):
        # Repeated queries of one index use its fixed label, but keep multiplicity
        # in the estimate. Each proposal has its own number of paid indices.
        ordered = np.sort(rows[..., :k], axis=-1)
        out["n_unique"][..., j] = 1 + np.count_nonzero(
            np.diff(ordered, axis=-1), axis=-1
        )
        lo, hi = _binomial_bounds(int(k))
        out["interval_lower"][..., j] = center + mean_gap * (
            2 * lo[successes[..., j]] - 1
        )
        out["interval_upper"][..., j] = center + mean_gap * (
            2 * hi[successes[..., j]] - 1
        )

    order = deterministic_gap_requests(p, q, maximum)
    for j, k in enumerate(out["det_budgets"]):
        seen = order[:k]
        out["det_n_unique"][j] = len(seen)
        limits = [sharp_pair_bounds(p, q, seen, y[seen]) for y in labels]
        out["det_lower"][:, j], out["det_upper"][:, j] = np.asarray(limits).T
    return out

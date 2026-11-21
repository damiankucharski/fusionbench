from itertools import product

import numpy as np
import pytest

from fusionbench.eval.frozen_pair_sampling import audit_fixed_pair
from fusionbench.eval.paired_brier import (
    PairAudit,
    deterministic_gap_requests,
    sharp_pair_bounds,
)


def example():
    # Rows 0 and 1 are observationally identical with different fixed labels.
    # Row 2 has zero gap. Uniform sampling may pay for it, gap sampling cannot.
    f = np.array([0.8, 0.8, 0.5, 0.1, 0.7])
    g = np.array([0.2, 0.2, 0.5, 0.4, 0.75])
    labels = np.array(
        [
            [0, 1, 1, 0, 0],
            [1, 0, 1, 0, 0],
            [0, 0, 0, 1, 1],
            [1, 1, 1, 0, 0],
            [1, 0, 0, 1, 0],
        ],
        dtype=np.uint8,
    )
    return f, g, labels


def test_vectorized_draws_match_scalar_public_estimators_and_exact_intervals():
    f, g, labels = example()
    result = audit_fixed_pair(
        f, g, labels, seed=112, draw_budgets=(1, 7, 31), repetitions=6
    )
    delta = f - g
    for i, proposal in enumerate(("uniform", "gap")):
        for rep, schedule in enumerate(result["draw_indices"][i]):
            for j, k in enumerate(result["draw_budgets"]):
                ids = schedule[:k]
                signed_weights = (
                    delta[ids] if i == 0 else result["mean_gap"] * np.sign(delta[ids])
                )
                audit = PairAudit(
                    ids, signed_weights, result["center"], result["mean_gap"], proposal
                )
                assert result["n_unique"][i, rep, j] == len(np.unique(ids))
                for w, y in enumerate(labels):
                    draws = y[ids]
                    np.testing.assert_allclose(
                        result["estimate"][i, w, rep, j],
                        audit.estimate(draws),
                        rtol=0,
                        atol=4e-16,
                    )
                    if proposal == "gap":
                        np.testing.assert_array_equal(
                            [
                                result["interval_lower"][w, rep, j],
                                result["interval_upper"][w, rep, j],
                            ],
                            audit.exact_interval(draws),
                        )
    assert np.any(result["draw_indices"][0] == 2)
    assert np.all(result["draw_indices"][1] != 2)
    assert np.any(result["n_unique"][0] != result["n_unique"][1])
    # Swapping the labels of the two feature-tied rows can change an estimate,
    # although their fully labeled pool risks are equal.
    assert result["true_gap"][0] == pytest.approx(result["true_gap"][1])
    assert np.any(result["estimate"][:, 0] != result["estimate"][:, 1])


def test_plans_have_shared_worlds_and_stable_repetition_and_budget_prefixes():
    f, g, labels = example()
    seed = np.random.SeedSequence([920, 6])
    a = audit_fixed_pair(f, g, labels, seed=seed, draw_budgets=(4, 9), repetitions=3)
    b = audit_fixed_pair(
        f, g, labels[::-1], seed=seed, draw_budgets=(9, 15), repetitions=5
    )
    np.testing.assert_array_equal(a["draw_indices"], b["draw_indices"][:, :3, :9])
    np.testing.assert_array_equal(
        a["estimate"][:, :, :, 1], b["estimate"][:, ::-1, :3, 0]
    )
    np.testing.assert_array_equal(a["n_unique"][:, :, 1], b["n_unique"][:, :3, 0])
    np.testing.assert_array_equal(
        a["interval_lower"][:, :, 1], b["interval_lower"][::-1, :3, 0]
    )
    assert seed.n_children_spawned == 0

    # Positive support and inverse-CDF maps reconstruct the paired schedules
    # without inspecting any labels.
    delta = np.abs(f - g)
    cdf = np.cumsum(delta / delta.sum())
    cdf /= cdf[-1]
    for r, stream in enumerate(np.random.SeedSequence([920, 6]).spawn(3)):
        uniforms = np.random.default_rng(stream).random(9)
        np.testing.assert_array_equal(
            a["draw_indices"][0, r], np.floor(uniforms * len(f)).astype(int)
        )
        np.testing.assert_array_equal(
            a["draw_indices"][1, r], cdf.searchsorted(uniforms, side="right")
        )


def test_finite_variances_and_deterministic_bounds_match_direct_enumeration():
    f, g, labels = example()
    result = audit_fixed_pair(
        f, g, labels, seed=52, draw_budgets=(1, 3, 9), repetitions=2
    )
    delta = f - g
    weights = np.abs(delta) / np.abs(delta).sum()
    for w, y in enumerate(labels):
        direct = np.mean((f - y) ** 2 - (g - y) ** 2)
        assert result["true_gap"][w] == direct
        for i, k in enumerate(result["draw_budgets"]):
            magnitude = np.abs(delta).mean()
            centered = delta * (1.0 - 2.0 * y)
            gap_values = magnitude * np.sign(delta) * (1.0 - 2.0 * y)
            expected_gap = (
                np.sum(weights * (gap_values - (weights @ gap_values)) ** 2) / k
            )
            variances = result["exact_variance"][:, w, i]
            np.testing.assert_allclose(
                variances, [np.var(centered) / k, expected_gap], rtol=1e-14, atol=1e-16
            )
            assert variances[0] - variances[1] == pytest.approx(
                np.var(np.abs(delta)) / k, abs=1e-15
            )
        for j, k in enumerate(result["det_budgets"]):
            ix = deterministic_gap_requests(f, g, k)
            assert result["det_n_unique"][j] == len(ix)
            observed = [result["det_lower"][w, j], result["det_upper"][w, j]]
            np.testing.assert_array_equal(observed, sharp_pair_bounds(f, g, ix, y[ix]))
            values = []
            for bits in product((0, 1), repeat=len(f)):
                z = np.array(bits)
                if np.array_equal(z[ix], y[ix]):
                    values.append(np.mean((f - z) ** 2 - (g - z) ** 2))
            np.testing.assert_allclose(
                observed, [min(values), max(values)], rtol=0, atol=2e-16
            )


def test_zero_gap_and_constant_sign_use_actual_label_costs():
    f, g, labels = example()
    zero = audit_fixed_pair(f, f, labels, seed=2, draw_budgets=(1, 9), repetitions=4)
    assert zero["draw_indices"].shape == (2, 4, 0)
    for field in (
        "true_gap",
        "exact_variance",
        "estimate",
        "interval_lower",
        "interval_upper",
        "n_unique",
        "det_n_unique",
        "det_lower",
        "det_upper",
    ):
        assert np.count_nonzero(zero[field]) == 0
    # A single nonzero-gap row may be drawn 100 times; only its one label
    # is paid for. Each binomial trial still uses its sign.
    f = np.array([1.0, 0.5, 0.5])
    g = np.array([0.0, 0.5, 0.5])
    y = np.array([[0, 1, 1], [1, 0, 1]])
    one = audit_fixed_pair(f, g, y, seed=7, draw_budgets=(100,), repetitions=2)
    np.testing.assert_array_equal(one["n_unique"][1], np.ones((2, 1)))
    np.testing.assert_array_equal(one["n_unique"][0], np.full((2, 1), 3))
    assert np.all(one["exact_variance"][1] == 0)
    assert np.all(one["interval_upper"][0] == pytest.approx(one["true_gap"][0]))
    assert np.all(one["interval_lower"][1] == pytest.approx(one["true_gap"][1]))

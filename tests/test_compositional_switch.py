from fractions import Fraction
from itertools import combinations, product

import numpy as np
import pytest

from fusionbench.scenarios.compositional_switch import CompositionalSwitch
from fusionbench.scenarios.registry import SCENARIOS


def test_paired_sampling_stream_and_anonymous_roles():
    seed = 7053
    scenario = CompositionalSwitch(role_seed=seed)
    assert scenario.model_ids() == [f"p{i}" for i in range(6)]
    assert scenario.roles == tuple(
        np.random.default_rng([seed, 229]).permutation(6)[:4]
    )
    n = 265
    g = lambda: np.random.default_rng([seed, 2])
    expected = g()
    mask = expected.random(n) < 0.5
    low_high = 1 + 62 * expected.integers(0, 2, (n, 6))
    full = expected.integers(1, 64, (n, 6))
    u = expected.random(n)
    sample = scenario.draw(n, g())
    np.testing.assert_array_equal(sample.uniform, u)
    np.testing.assert_array_equal(sample.corner_rows, mask)
    np.testing.assert_array_equal(
        sample.scores, np.where(mask[:, None], low_high, full) / 64
    )
    np.testing.assert_array_equal(sample.preds[:, :, 0], 1 - sample.scores)
    np.testing.assert_array_equal(scenario.draw(n, g(), 1).scores, low_high / 64)
    np.testing.assert_array_equal(scenario.draw(n, g(), 0).scores, full / 64)
    np.testing.assert_array_equal(scenario.draw(n, g(), 1).uniform, u)
    a, b, c, d = sample.scores[:, sample.roles].T
    # The Boolean threshold-subset rule gives the same scalar lattice output.
    ticks = np.arange(1, 64) / 64
    support = ((a[:, None] >= ticks) & (b[:, None] >= ticks)) | (
        (c[:, None] >= ticks) & (d[:, None] >= ticks)
    )
    composition = support.sum(1) / 64
    odds = np.exp(0.2) * (a / (1 - a)) ** 0.7 * (b / (1 - b)) ** 0.7
    additive = odds / (1 + odds)
    np.testing.assert_array_equal(sample.probability("composition"), composition)
    np.testing.assert_allclose(sample.probability("logit"), additive, rtol=1e-14)
    np.testing.assert_array_equal(
        sample.labels("composition"), (u < composition).astype(int)
    )
    np.testing.assert_array_equal(sample.labels("logit"), (u < additive).astype(int))


@pytest.mark.parametrize("law", ["logit", "composition"])
def test_standard_scenario_fit_test_interface(law):
    scenario = SCENARIOS["compositional_switch"](
        law=law, role_seed=219, corner_probability=1
    )
    dataset = scenario.build(31, seed=533)
    assert dataset.model_ids == scenario.model_ids()
    for x, q in [
        (dataset.fit_preds, dataset.fit_oracle),
        (dataset.test_preds, dataset.test_oracle),
    ]:
        assert x.shape == (31, 6, 2) and q.shape == (31, 2)
        assert set(np.unique(x)) == {1 / 64, 63 / 64}
        np.testing.assert_array_equal(q, scenario.bayes_oracle(x, {}))


def test_corner_certificate_for_arbitrary_support_order_and_repeated_operands():
    # Every nonempty multiset is equivalent here to a subset for MIN/MAX.
    # CLOSE/FAR on two endpoint values choose MIN, MAX, or the first operand
    # when both distances tie. This covers every fixed order and cardinality.
    bits = np.asarray(list(product((0, 1), repeat=6)), dtype=int)
    target = (bits[:, 0] & bits[:, 1]) | (bits[:, 2] & bits[:, 3])
    errs = {"min": [], "max": [], "first": []}
    for j in range(6):
        errs["first"].append(np.count_nonzero(bits[:, j] != target))
    for size in range(1, 7):
        for cols in combinations(range(6), size):
            x = bits[:, cols]
            errs["min"].append(np.count_nonzero(x.min(1) != target))
            errs["max"].append(np.count_nonzero(x.max(1) != target))
    assert {k: Fraction(int(min(v)), 64) for k, v in errs.items()} == {
        "min": Fraction(3, 16),
        "max": Fraction(5, 16),
        "first": Fraction(5, 16),
    }
    # Check the two selector directions directly. On this two-value alphabet,
    # one threshold strictly below/equal/above the midpoint exhausts the
    # regimes. Every possible first operand is included at the exact tie.
    half_grid = (0, 0.5, 1)
    selector_errors = {"close": [], "far": []}
    for size in range(1, 7):
        for subset in combinations(range(6), size):
            for first in subset:
                columns = [first, *(i for i in subset if i != first)]
                x = bits[:, columns]
                for t in half_grid:
                    distances = abs(x - t)
                    for kind, chosen in (
                        ("close", distances.argmin(1)),
                        ("far", distances.argmax(1)),
                    ):
                        selected = x[np.arange(64), chosen]
                        selector_errors[kind].append(
                            np.count_nonzero(selected != target)
                        )
    assert {k: Fraction(int(min(v)), 64) for k, v in selector_errors.items()} == {
        "close": Fraction(3, 16),
        "far": Fraction(3, 16),
    }
    # Any nonnegative weighted mean, including means with repeated inputs,
    # has w in the simplex on six variables. At w*=(1/4,...,1/4,0,0), the
    # four supported gradient components must be equal, and each remaining
    # component must be at least as large. These KKT conditions suffice for
    # the convex quadratic. Compute the inner products in integer arithmetic.
    residual_times_four = bits[:, :4].sum(1) - 4 * target
    gradients = bits.T @ residual_times_four
    assert np.all(gradients[:4] == gradients[0])
    assert np.all(gradients[4:] > gradients[0])
    binary_excess = Fraction(int(residual_times_four @ residual_times_four), 64 * 4**2)
    assert binary_excess == Fraction(1, 8)
    corner_excess = binary_excess * Fraction(62, 64) ** 2
    assert corner_excess == Fraction(961, 8192)
    assert corner_excess / 2 == Fraction(961, 16384)
    # The witness attains zero excess under the composition law on all corners.
    scenario = CompositionalSwitch(role_seed=71)
    scores = (1 + 62 * bits) / 64
    a, b, c, d = scores[:, scenario.roles].T
    witness = np.maximum(np.minimum(a, b), np.minimum(c, d))
    p = np.stack((1 - scores, scores), axis=-1)
    np.testing.assert_array_equal(scenario.bayes_oracle(p, {})[:, 1], witness)


@pytest.mark.parametrize("p", [-0.1, 1.1, np.nan])
def test_invalid_corner_probability(p):
    with pytest.raises(ValueError):
        CompositionalSwitch(corner_probability=p)
    with pytest.raises(ValueError):
        CompositionalSwitch().draw(10, np.random.default_rng(1), p)

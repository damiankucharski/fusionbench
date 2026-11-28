import numpy as np
import pytest

from fusionbench.scenarios.stratified_pool import StratifiedPool


@pytest.mark.parametrize("n", [1, 7, 32, 128, 513])
def test_balance_across_amplitudes_and_realized_labels(n):
    scenario = StratifiedPool()
    x, y, meta = scenario.sample(n, np.random.default_rng(42))
    count = np.bincount(2 * meta["high_amplitude"].astype(int) + y, minlength=4)
    assert count.max() - count.min() <= 1
    np.testing.assert_allclose(
        np.abs(x[:, 0, 1] - 0.5), np.where(meta["high_amplitude"], 0.35, 0.15)
    )
    for split in ("fit", "test"):
        same, labels, groups = scenario.sample_split(
            n, np.random.default_rng(42), split
        )
        np.testing.assert_array_equal(same, x)
        np.testing.assert_array_equal(labels, y)
        np.testing.assert_array_equal(groups["high_amplitude"], meta["high_amplitude"])


@pytest.mark.parametrize("d", [0, 0.25, 0.75, 1])
def test_mean_population_excess_and_stratum_mass(d):
    x, m, q = StratifiedPool(d).support()
    assert np.unique(x[:, :, 1], axis=0).shape == (512, 8)
    np.testing.assert_allclose([q[:256].sum() / 512, q[256:].sum() / 512], [0.25, 0.25])
    np.testing.assert_allclose(m.sum(), 1)
    excess = 2 * np.dot(m, (x[:, :, 1].mean(1) - q) ** 2)
    expected = 2 * np.mean(np.square([0.15, 0.35])) * (7 / 8) * d**2
    np.testing.assert_allclose(excess, expected, atol=1e-15)


def test_sampler_conditional_distribution_matches_soft_label_oracle():
    law = StratifiedPool(0.75)
    x, y, _ = law.sample(40_000, np.random.default_rng(471))
    support, m, q = law.support()
    agree = x[:, 0, 1] > 0.5
    measured = np.mean(agree == y)
    exact = m @ np.where(support[:, 0, 1] > 0.5, q, 1 - q)
    assert abs(measured - exact) < 0.01


@pytest.mark.parametrize("bad", [-0.1, np.nan, 1.01, True, "0.25"])
def test_specialization_is_a_probability(bad):
    with pytest.raises(ValueError):
        StratifiedPool(bad)

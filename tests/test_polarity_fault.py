import numpy as np
from scipy.special import expit
from scipy.stats import multivariate_normal, norm

from fusionbench.eval.bounded_evidence import capped
from fusionbench.scenarios.polarity import PolarityFault
from fusionbench.scenarios.registry import SCENARIOS


def test_formula_against_explicit_gaussian_mixture_density():
    sc = SCENARIOS["polarity_fault"](fault_rate=0.13, correlation=0.4)
    preds, _, latent = sc._generate(450, np.random.default_rng(3))
    x = latent["raw"]
    s = sc.strengths
    cov = 0.4 * np.ones((6, 6)) + 0.6 * np.eye(6)
    density = []
    for sign in (-1, 1):
        stable = multivariate_normal.pdf(x[:, 1:], mean=sign * s[1:], cov=cov)
        unreliable = 0.87 * norm.pdf(x[:, 0], sign * s[0]) + 0.13 * norm.pdf(x[:, 0], -sign * s[0])
        density.append(stable * unreliable)
    want = density[1] / (density[0] + density[1])
    np.testing.assert_allclose(sc.bayes_oracle(preds, None)[:, 1], want, atol=1e-12)


def test_zero_correlation_and_zero_fault_reduce_to_sum_of_logits():
    sc = PolarityFault(0.0, 0.0)
    p, _, latent = sc._generate(100, np.random.default_rng(4))
    np.testing.assert_allclose(
        sc.bayes_oracle(p, None)[:, 1], expit((2 * sc.strengths * latent["raw"]).sum(1))
    )


def test_clip_is_two_relu_units_and_keeps_other_raw_sensors():
    x = np.random.default_rng(5).normal(size=(128, 7))
    c = capped(x, 2, 0.75)
    expected = np.maximum(x[:, 2] + 0.75, 0) - np.maximum(x[:, 2] - 0.75, 0) - 0.75
    np.testing.assert_allclose(c[:, 2], expected, atol=1e-14)
    np.testing.assert_array_equal(c[:, [0, 1, 3, 4, 5, 6]], x[:, [0, 1, 3, 4, 5, 6]])

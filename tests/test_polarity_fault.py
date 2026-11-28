import numpy as np
from scipy.special import expit
from scipy.stats import multivariate_normal, norm

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
        unreliable = 0.87 * norm.pdf(x[:, 0], sign * s[0]) + 0.13 * norm.pdf(
            x[:, 0], -sign * s[0]
        )
        density.append(stable * unreliable)
    want = density[1] / (density[0] + density[1])
    np.testing.assert_allclose(sc.bayes_oracle(preds, None)[:, 1], want, atol=1e-12)


def test_zero_correlation_and_zero_fault_reduce_to_sum_of_logits():
    sc = PolarityFault(0.0, 0.0)
    p, _, latent = sc._generate(100, np.random.default_rng(4))
    np.testing.assert_allclose(
        sc.bayes_oracle(p, None)[:, 1], expit((2 * sc.strengths * latent["raw"]).sum(1))
    )

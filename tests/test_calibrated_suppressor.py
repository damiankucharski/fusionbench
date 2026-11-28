import numpy as np
import pytest
from scipy.special import expit
from scipy.stats import multivariate_normal

from fusionbench.scenarios.registry import SCENARIOS


@pytest.mark.parametrize("corr", [0.0, 0.5, 0.85])
def test_posterior_matches_gaussian_density_and_weak_direction(corr):
    scenario = SCENARIOS["calibrated_suppressor"](corr=corr)
    p, _, latent = scenario._generate(1024, np.random.default_rng(42))
    z = np.log(p[:, :, 1]) - np.log(p[:, :, 0])
    mu = np.array([2.0, 0.5])
    density_ratio = multivariate_normal.logpdf(
        z, mu, scenario.covariance
    ) - multivariate_normal.logpdf(z, -mu, scenario.covariance)
    np.testing.assert_allclose(
        scenario.bayes_oracle(p, latent)[:, 1], expit(density_ratio)
    )
    z = np.array([[0.0, -1.0], [0.0, 1.0]])
    probe = np.stack((expit(-z), expit(z)), axis=2)
    q = scenario.bayes_oracle(probe, {})[:, 1]
    assert np.sign(q[1] - q[0]) == np.sign(0.5 - corr)


def test_singular_correlation_rejected():
    with pytest.raises(ValueError, match="corr"):
        SCENARIOS["calibrated_suppressor"](corr=1)

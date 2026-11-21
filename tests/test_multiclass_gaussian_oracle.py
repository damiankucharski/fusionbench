"""Check the public posterior against densities in observable log-ratio space."""

import numpy as np
import pytest
from scipy.special import softmax
from scipy.stats import multivariate_normal

from fusionbench.scenarios.calibrated import mc_strength, multiclass_gaussian_oracle
from fusionbench.scenarios.calibration import MulticlassCommittee
from fusionbench.scenarios.committee import Saturated


def density_reference(p, accuracy, rho):
    n, k, c = p.shape
    if rho == 1:
        # All providers are copies. Work only in the nonsingular observed subspace.
        p, k, rho = p[:, :1], 1, 0
    differences = np.eye(c)[:-1] - np.eye(c)[-1]
    cross_provider = (1 - rho) * np.eye(k) + rho * np.ones((k, k))
    cov = np.kron(cross_provider, differences @ differences.T)
    observations = np.log(p[:, :, :-1] / p[:, :, -1:]).reshape(n, -1)
    strength = mc_strength(accuracy, c)
    likelihood = [
        multivariate_normal.logpdf(
            observations, mean=np.tile(strength * differences[:, cls], k), cov=cov
        )
        for cls in range(c)
    ]
    return softmax(np.column_stack(likelihood), axis=1)


@pytest.mark.parametrize(
    "scenario,rho",
    [
        (MulticlassCommittee(), 0),
        (MulticlassCommittee(k=1, n_classes=3, accuracy=0.85), 0),
        (Saturated(), 0.95),
        (Saturated(k=3, n_classes=3, accuracy=0.68, corr=0.4), 0.4),
        (Saturated(k=3, n_classes=3, accuracy=0.8, corr=1), 1),
        (Saturated(k=3, n_classes=3, accuracy=0.8, corr=2), 1),
        (Saturated(k=3, n_classes=3, accuracy=0.8, corr=-1), 0),
    ],
)
def test_public_oracle_matches_density_in_observed_coordinates(scenario, rho):
    p, _, latent = scenario._generate(23, np.random.default_rng(701))
    q = scenario.bayes_oracle(p, latent)
    np.testing.assert_allclose(
        q, density_reference(p, scenario.accuracy, rho), atol=2e-13, rtol=1e-11
    )


def test_small_positive_probabilities_retain_their_evidence():
    p = np.array([[[1e-30, 0.4, 0.6], [0.2, 0.5, 0.3]]])
    q = multiclass_gaussian_oracle(p, 0.58)
    log_ratios = np.log(p[0, :, 0] / p[0, :, 1])
    assert np.isclose(np.log(q[0, 0] / q[0, 1]), mc_strength(0.58, 3) * sum(log_ratios))

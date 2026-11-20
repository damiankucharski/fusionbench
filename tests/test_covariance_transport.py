import numpy as np
from scipy.special import expit, logit

from fusionbench.eval.covariance_transport import (
    FAULT,
    Population,
    draw,
    posterior,
    projected_correlation,
)
from fusionbench.scenarios.polarity import PolarityFault


def test_posterior_agrees_with_polarity_oracle_after_changing_prior_and_offsets():
    scenario = PolarityFault()
    s = scenario.strengths
    target = Population(0.3, 0.72, 0.6, -0.1)
    x, _, _ = draw(19312, 3, 1024, target, s)
    centered = x.copy()
    centered[:, 1:] -= 0.6 * s[1:]
    centered[:, 0] += 0.1 * (1 - 2 * FAULT) * s[0]
    scores = np.stack([expit(-2 * s * centered), expit(2 * s * centered)], axis=-1)
    equal = scenario.posterior(scores, fault_rate=FAULT, correlation=0.3)
    np.testing.assert_allclose(posterior(x, s, target), expit(logit(0.72) + logit(equal)), atol=1e-12)


def test_contrasts_cancel_shifts_along_signal_and_common_noise():
    generator = np.random.default_rng(728)
    s = PolarityFault().strengths[1:]
    x = generator.normal(scale=0.6, size=(128, len(s)))
    shifted = x + generator.normal(size=(128, 1)) * s + generator.normal(size=(128, 1))
    assert abs(projected_correlation(x, s) - projected_correlation(shifted, s)) < 1e-12

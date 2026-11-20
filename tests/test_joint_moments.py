import numpy as np
from scipy.special import expit, logit
from fusionbench.eval.joint_moments import OFFSETS, SCALES, STRENGTHS, draw, moments_to_rule


def test_population_recovers_gaussian_oracle_with_unknown_correlated_pair():
    d = 2 * STRENGTHS * SCALES
    p = 0.75
    residual = np.eye(6)
    residual[0, 1] = residual[1, 0] = 0.75
    residual *= np.outer(SCALES, SCALES)
    mean = OFFSETS + (p - 0.5) * d
    cov = residual + p * (1 - p) * np.outer(d, d)
    third = p * (1 - p) * (1 - 2 * p) * np.einsum("i,j,k->ijk", d, d, d)
    sd = np.sqrt(np.diag(cov))
    rule = moments_to_rule(cov / np.outer(sd, sd), third / np.einsum("i,j,k->ijk", sd, sd, sd), mean, sd)
    assert abs(rule.prior - p) < 1e-12
    x = draw(991, 0, 100, "pair")[0]
    q = expit(logit(p) + (x - OFFSETS) @ np.linalg.solve(residual, d))
    np.testing.assert_allclose(rule.predict(x), q, atol=1e-12)


def test_exactly_matched_unlabeled_inputs_for_invisible_shift():
    x, _, q, _ = draw(8, 1, 21, "pair")
    shifted, _, flipped, _ = draw(8, 1, 21, "invisible")
    np.testing.assert_array_equal(x, shifted)
    np.testing.assert_array_equal(0.25 + 0.5 * q, flipped)

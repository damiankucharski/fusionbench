import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from fusionbench.eval.weak_mixtures import Batch, draw, m_step, parameters, standardize


def test_fast_log_density_and_posterior_match_direct_gaussians():
    truth = parameters("weak")
    x = draw(2134, 0, 92, truth)
    batch = Batch.make(x)
    post, ll = truth.evaluate(batch)
    joint = np.stack(
        [
            multivariate_normal.logpdf(x, mean, truth.cov) + np.log(w)
            for mean, w in zip(truth.means, (1 - truth.prior, truth.prior))
        ]
    )
    density = logsumexp(joint, axis=0)
    np.testing.assert_allclose(ll, density.mean(), atol=1e-12)
    np.testing.assert_allclose(post, np.exp(joint[1] - density), atol=1e-12)
    _, after = m_step(batch, post).evaluate(batch)
    assert after >= ll
    scales = x.std(0)
    newq, newdensity = standardize(truth, x.mean(0), scales).evaluate(
        Batch.make((x - x.mean(0)) / scales)
    )
    np.testing.assert_allclose(newq, post, atol=1e-12)
    np.testing.assert_allclose(newdensity, ll + np.log(scales).sum(), atol=1e-12)


@pytest.mark.parametrize("seed", range(2300, 2312))
def test_equal_components_keep_null_prior_through_repeated_em(seed):
    truth = parameters("zero_signal")
    x = draw(seed, 0, 1024, truth)
    m, s = x.mean(0), x.std(0)
    data = Batch.make((x - m) / s)
    state = standardize(truth, m, s)
    for _ in range(192):
        r, _ = state.evaluate(data)
        state = m_step(data, r)
    np.testing.assert_allclose(state.prior, truth.prior, atol=1e-12)

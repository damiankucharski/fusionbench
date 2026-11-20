import numpy as np

from fusionbench.eval.context_stress import features, modified, transfer_surface
from fusionbench.eval.seed_selection import observations


def test_vote_moments_are_insensitive_to_temperature():
    base, _, _ = observations(4, 17, 300)
    before, after = features(base), features(modified(base, temperature=.1))
    np.testing.assert_array_equal(before[:, :21], after[:, :21])
    assert np.mean(abs(before[:, 21] - after[:, 21])) > .3


def test_fit_independent_extra_fault_weights_and_risks():
    reference = np.array([[0., 0., 0.], [1., 0., 0.]])
    alternatives = np.stack([reference + [0., 1., 0.], reference + [0., 0., 1.]])
    risks = np.array([[.1, .2], [.3, .4]])
    perturbed_risks = np.stack([risks + .2, risks + .3])
    result, channel, p, q, sse = transfer_surface(
        np.array([.2, 0., .4]), reference, alternatives, risks, perturbed_risks)
    assert channel == 2
    np.testing.assert_allclose([p, q, sse], [.2, .4, 0.], atol=1e-14)
    np.testing.assert_allclose(result, .8 * risks[0] + .2 * risks[1] + .4 * .3)

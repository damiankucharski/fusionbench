import numpy as np
import pytest

from fusionbench.eval.independent_adoption import draw, log_evidence


def test_fixed_stake_gate_and_full_support():
    gains = np.array([.25, -.5, .125])
    raw = np.mean([np.prod(1 + stake * gains) for stake in (.25, .5, 1.)])
    assert np.exp(log_evidence(gains)) == pytest.approx(raw)
    assert log_evidence(np.zeros(16)) == pytest.approx(0.)
    assert log_evidence([-1]) < 0
    with pytest.raises(ValueError):
        log_evidence([1.01])


def test_shift_changes_conditional_law_and_preserves_channel_marginals():
    a, _, maximum = draw(32, 5, 0, 1.)
    b, _, linear = draw(32, 5, 0, 0.)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(maximum, a[:,:,1].max(axis=1))
    assert not np.allclose(maximum, linear)

import numpy as np

from fusionbench.scenarios.native_switch import (
    NativeSwitch,
    draw_native_switch,
    farther,
)
from fusionbench.scenarios.registry import SCENARIOS


def test_known_switch_and_stationary_controls():
    fit = draw_native_switch(10, 0, 512)
    val = draw_native_switch(10, 1, 512)
    assert (fit.parent, fit.child) == (val.parent, val.child)
    assert fit.parent != fit.child and not np.array_equal(fit.preds, val.preds)
    assert [*SCENARIOS["native_switch"]().model_ids()] == [f"p{i}" for i in range(6)]
    for mixing in [0, 0.5, 1]:
        q = fit.probability(mixing)
        np.testing.assert_array_equal(fit.labels(mixing), fit.uniform < q)
        np.testing.assert_array_equal(
            NativeSwitch(mixing=mixing, role_seed=10).bayes_oracle(fit.preds, {})[:, 1],
            q,
        )
    np.testing.assert_allclose(
        fit.probability(0.5), (fit.probability(0) + fit.probability(1)) / 2
    )
    assert farther(0.25, 0.75, 0.5) == 0.25 and farther(0.75, 0.25, 0.5) == 0.75


def test_close_control_reuses_anon_features_and_native_tie_priority():
    from fusionbench.scenarios.native_switch import (
        NativeCloseSwitch,
        conditional_probability,
    )

    far = NativeSwitch(role_seed=905)
    close = NativeCloseSwitch(role_seed=905)
    sample = close.draw(40, np.random.default_rng(22))
    partner = far.draw(40, np.random.default_rng(22))
    np.testing.assert_array_equal(sample.preds, partner.preds)
    np.testing.assert_array_equal(sample.uniform, partner.uniform)
    a, b = sample.preds[:, sample.parent, 1], sample.preds[:, sample.child, 1]
    expected = np.where(abs(a - 37 / 64) <= abs(b - 37 / 64), a, b)
    np.testing.assert_array_equal(sample.probability(0), expected)
    np.testing.assert_array_equal(close.bayes_oracle(sample.preds, {})[:, 1], expected)
    np.testing.assert_array_equal(sample.probability(1), partner.probability(1))
    assert (
        conditional_probability(36 / 64, 38 / 64, 0, close=True, threshold=37 / 64)
        == 36 / 64
    )
    assert SCENARIOS["native_close_switch"] is NativeCloseSwitch

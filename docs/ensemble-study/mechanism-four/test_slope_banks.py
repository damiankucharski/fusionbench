"""Check the sign and magnitude of the paired slope intervention on both splits."""

import numpy as np
from banks_study import _factorial
from scipy.special import logit
from slope_banks import SLOPES, _vary_slope

from fusionbench.scenarios.grouped_rare_event import GroupedRareEvent


def test_slope_changes_only_provider_signal_loading_on_both_splits():
    seed = 81048
    bank = next(b for b in _factorial(5, (seed,)) if b.context["family"] == "h1c1s1")
    first, second, third = _vary_slope(bank)
    for split, (attr, n) in enumerate((("x", 128), ("audit_x", 2048))):
        _, _, latent = GroupedRareEvent().sample(
            n, np.random.default_rng([seed, split])
        )
        signal = np.broadcast_to(latent["signal"][:, None], (n, 8)).copy()
        signal[latent["specialist_region"], 3] *= 1 - 0.85
        old = logit(getattr(first, attr)[:, :, 1])
        for other in (second, third):
            change = other.context["slope"] - SLOPES[0]
            actual = logit(getattr(other, attr)[:, :, 1]) - old
            np.testing.assert_allclose(actual, change * signal, atol=2e-11)
    assert [b.context["slope"] for b in (first, second, third)] == list(SLOPES)
    assert len({b.context["audit_x_sha256"] for b in (first, second, third)}) == 3
    assert first.context["audit_x_sha256"] == bank.context["audit_x_sha256"]
    assert first.context["fit_x_sha256"] == bank.context["fit_x_sha256"]

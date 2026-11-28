"""Bank integrity and the full-class Brier bound for identical training information."""

import numpy as np
import pandas as pd
import pytest
from banks_auto import _factorial, _witness, banks, score
from witness_auto import summarize

from fusionbench.scenarios.grouped_rare_event import GroupedRareEvent


def test_factorial_roles_move_together_and_only_the_planted_provider_changes():
    by_name = {b.context["family"]: b for b in _factorial(1, [72719])}
    assert len(by_name) == 8
    original = by_name["h0c0s0"]
    order_seed = original.context["provider_order_seed"]
    assert order_seed == np.random.SeedSequence([72719, 317]).generate_state(1)[0]
    role = GroupedRareEvent(provider_order_seed=order_seed).provider_order.index(3)
    assert tuple(
        GroupedRareEvent(provider_order_seed=order_seed).provider_order
    ) != tuple(range(8))
    for b in by_name.values():
        assert b.context["provider_signal_slope"] == 1.8
        assert b.context["provider_order_seed"] == order_seed
        assert b.context["fit_y_sha256"] == original.context["fit_y_sha256"]
        assert b.context["audit_y_sha256"] == original.context["audit_y_sha256"]
        assert (
            b.context["audit_regime_sha256"] == original.context["audit_regime_sha256"]
        )
        assert not np.intersect1d(b.groups, b.audit_groups).size

    changed = by_name["h0c0s1"]
    same = np.arange(8) != role
    np.testing.assert_array_equal(changed.audit_x[:, same], original.audit_x[:, same])
    np.testing.assert_array_equal(
        changed.audit_x[~original.audit_regime],
        original.audit_x[~original.audit_regime],
    )
    assert np.any(
        changed.audit_x[original.audit_regime, role]
        != original.audit_x[original.audit_regime, role]
    )


def test_witness_has_identical_observations_at_choice_time_and_exact_risk_bound():
    a, b = list(_witness(3, (32171,)))
    for name in ("x", "y", "audit_x"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
    for name in (
        "fit_x_sha256",
        "fit_y_sha256",
        "audit_x_sha256",
        "audit_uniform_sha256",
    ):
        assert a.context[name] == b.context[name]
    assert not np.array_equal(a.audit_y, b.audit_y)
    np.testing.assert_array_equal(a.q, a.audit_x[:, 0, 1])
    np.testing.assert_array_equal(b.q, b.audit_x[:, :, 1].mean(1))

    bound = a.context["common_rule_excess_bound"]
    assert 0 < bound == b.context["common_rule_excess_bound"]
    midpoint = (a.q + b.q) / 2
    alternative = np.clip(midpoint + 0.21, 0, 1)
    for x in (midpoint, alternative, a.q, b.q):
        prob = np.stack((1 - x, x), axis=1)
        av = (
            score(a, prob)["expected_brier"]
            - a.context["bayes_brier"]
            + score(b, prob)["expected_brier"]
            - b.context["bayes_brier"]
        ) / 2
        assert av == pytest.approx(bound + 2 * np.mean((x - midpoint) ** 2), abs=5e-15)


@pytest.mark.parametrize("bad", [True, None, -1, 0, 4, 2.0])
def test_invalid_round_number(bad):
    with pytest.raises(ValueError, match="round_no"):
        next(banks(bad))


def test_witness_audit_rejects_a_changed_forecast_and_reports_expected_excess():
    rows = []
    for a in _witness(3, (32171,)):
        p = a.audit_x[:, 0]
        rows.append(
            {
                **a.context,
                **score(a, p),
                "method": "first",
                "action": "first",
                "fit_hash": "shared_fit",
                "test_hash": "shared_test",
            }
        )
    table = pd.DataFrame(rows)
    paired, aggregate = summarize(table)
    assert aggregate.loc["first", "paired_draws"] == 1
    assert paired.iloc[0]["best_world_expected_excess"] == pytest.approx(0)
    bound = paired.iloc[0]["common_rule_excess_bound"]
    assert aggregate.loc["first", "mean_expected_excess"] == pytest.approx(2 * bound)
    table.loc[1, "test_hash"] = "changed"
    with pytest.raises(ValueError, match="share"):
        summarize(table)

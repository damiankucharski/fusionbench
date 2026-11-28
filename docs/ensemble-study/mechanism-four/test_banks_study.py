"""Exercise paired controls and conditional risk without running any fusion training."""

from types import SimpleNamespace

import numpy as np
import pytest
from banks_study import GROUPED_CASES, _clustered, _factorial, _standard, banks, score


def test_all_eight_cells_share_truths_but_only_specialist_changes_in_its_region():
    by_family = {b.context["family"]: b for b in _factorial(1, [81032])}
    assert len(by_family) == 8
    for stem in ("h0c0", "h0c1", "h1c0", "h1c1"):
        off, on = by_family[stem + "s0"], by_family[stem + "s1"]
        np.testing.assert_array_equal(
            off.audit_x[~on.audit_regime], on.audit_x[~on.audit_regime]
        )
        np.testing.assert_array_equal(
            off.audit_x[:, [0, 1, 2, 4, 5, 6, 7]], on.audit_x[:, [0, 1, 2, 4, 5, 6, 7]]
        )
        assert np.any(off.audit_x[on.audit_regime, 3] != on.audit_x[on.audit_regime, 3])
        assert on.context["fit_y_sha256"] == by_family["h0c0s0"].context["fit_y_sha256"]
        np.testing.assert_array_equal(on.audit_y, by_family["h0c0s0"].audit_y)
        np.testing.assert_array_equal(on.q, by_family["h0c0s0"].q)
        assert len(np.unique(on.groups)) == 16
        assert len(np.unique(on.audit_groups)) == 256
        assert not np.intersect1d(on.groups, on.audit_groups).size


def test_exact_conditional_and_observed_score_have_different_information():
    p = np.array([[0.7, 0.3], [0.7, 0.3], [0.3, 0.7]])
    q = np.array([0.2, 0.8, 0.6])
    bank = SimpleNamespace(
        audit_x=p[:, None, :], x=p[:, None, :], audit_y=np.array([1, 0, 1]), q=q
    )
    t = np.eye(2)
    observed = np.square(p - t[bank.audit_y]).sum(1).mean()
    expected = np.mean(
        (1 - q) * np.square(p - t[0]).sum(1) + q * np.square(p - t[1]).sum(1)
    )
    scores = score(bank, p)
    assert scores["brier"] == pytest.approx(observed)
    assert scores["expected_brier"] == pytest.approx(expected)
    bank.audit_y = 1 - bank.audit_y
    changed = score(bank, p)
    assert changed["brier"] != pytest.approx(observed)
    assert changed["expected_brier"] == pytest.approx(expected)


def test_scene_controls_share_audit_and_reference_posterior():
    a, b = list(_clustered(1, (81035,)))
    np.testing.assert_array_equal(a.q, b.q)
    np.testing.assert_array_equal(a.audit_x, b.audit_x)
    assert len(np.unique(a.groups)) == 16
    assert len(np.unique(b.groups)) == 128
    assert len(a.y) == len(b.y) == 128


def test_standard_panel_retains_all_families_and_oracle_access_is_optional():
    all_banks = list(_standard(1, [81036]))
    assert len(all_banks) == 30
    for b in all_banks:
        m = score(b, b.audit_x.mean(1))
        assert m["brier"] >= 0
        assert np.isfinite(m["expected_brier"]) == (
            b.context["family"] in GROUPED_CASES
        )
        assert np.isfinite(m["ap"]) == (b.context["n_classes"] == 2)


@pytest.mark.parametrize("bad", [True, 0, 5, 1.0, -1])
def test_round_boundaries(bad):
    with pytest.raises(ValueError):
        next(banks(bad))

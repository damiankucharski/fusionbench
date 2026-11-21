"""Exercise a complete disjoint synthetic bank and explicit ranking counterexamples."""

import json

import numpy as np
import pandas as pd
import pytest

from fusionbench.eval.frozen_pair_ranking import (
    Setting,
    integrated_risks,
    run,
    select_frozen,
    validation_losses,
)


def test_integrated_label_change_slope_and_crossing():
    y = np.array([0.0, 1.0, 1.0, 0.0])
    a = np.array([[0.1, 0.9, 0.9, 0.1], [0.2, 0.8, 0.8, 0.2]])
    b = np.array([[0.03, 0.97, 0.97, 0.03], [0.94, 0.06, 0.06, 0.94]])
    lams = np.linspace(0, 1, 5)
    _, _, _, _, overall = integrated_risks(a, b, y, 0.5, lams)
    expected = np.stack(
        [
            0.5 * ((a - y) ** 2).mean(1)
            + 0.5
            * ((1 - w) * ((b - y) ** 2).mean(1) + w * ((b - (1 - y)) ** 2).mean(1))
            for w in lams
        ],
        axis=1,
    )
    np.testing.assert_allclose(expected, overall, atol=2e-15)
    diff = overall[0] - overall[1]
    assert diff[0] < 0 < diff[-1]
    np.testing.assert_allclose(
        diff, diff[0] + lams * np.mean((b[0] - b[1]) * (2 * y - 1))
    )


def test_missing_validation_context_uses_original_order():
    pytest.importorskip("okapi")
    x = np.tile([0.4, 0.6], (10, 7, 1))
    y = np.zeros(10)
    group = np.zeros(10, dtype=bool)
    selected, bit = select_frozen(
        x, group, validation_losses(np.full((3, 10), 0.5), y, group), x, 0.1, 0.5
    )
    assert (
        selected["validation_idx"]
        == selected["transport_idx"]
        == selected["known_p_idx"]
        == 0
    )
    assert selected["feature_index"] == 0 and selected["missing_context"]
    assert selected["estimated_p"] == 0.1 and not bit.any()


def test_smoke_with_real_search(tmp_path):
    pytest.importorskip("okapi")
    cfg = Setting(
        data_seeds=(81,),
        gp_seeds=(301, 307),
        generations=1,
        repeats=2,
        fit_n=32,
        val_n=25,
        audit_n=17,
        monitor_n=13,
        test_n=49,
        budgets=(2, 5),
        schedules=3,
    )
    run(tmp_path, workers=1, setting=cfg)
    c = pd.read_csv(tmp_path / "candidates.csv.gz")
    assert (
        c.fit_calls.min() > 0 and c.program.map(lambda s: len(json.loads(s))).min() > 0
    )
    risks = pd.read_csv(tmp_path / "risks.csv.gz")
    assert set(risks.method) == {"gp_301", "gp_307", "mean", "top_n", "stack_logit"}
    selections = pd.read_csv(tmp_path / "selections.csv.gz")
    archive = np.load(tmp_path / "audit-draws.npz")
    assert archive["estimate"].shape == (1, 2, 2, 2, 5, 3, 2)
    for (r, p), rows in selections.groupby(["repeat", "target_p"]):
        pi = cfg.target_p.index(p)
        assert rows.transport_seed.nunique() == rows.validation_seed.nunique() == 1
        np.testing.assert_allclose(
            rows.d, rows.d0 + rows["lambda"] * rows.slope, atol=2e-15
        )
        np.testing.assert_allclose(
            rows.d, rows.transport_brier - rows.validation_brier, atol=2e-15
        )
        labels = archive["world_labels"][0, r, pi]
        f, g = archive["f"][0, r, pi], archive["g"][0, r, pi]
        np.testing.assert_allclose(
            archive["true_gap"][0, r, pi],
            ((f - labels) ** 2 - (g - labels) ** 2).mean(axis=1),
            atol=2e-15,
        )
        assert np.array_equal(
            labels[0] != labels[-1], archive["audit_context"][0, r, pi].astype(bool)
        )
    pools = pd.read_csv(tmp_path / "pools.csv.gz")
    assert set(pools.role) == {
        "fit",
        "eval_clean",
        "eval_changed",
        "validation",
        "monitor",
        "audit",
    }
    assert pools.query("role=='monitor'").labels_sha256.isna().all()
    assert (pools.query("role=='audit'").n == 17).all()

"""Check shared external and synthetic coverage controls on small fixed arrays."""

import importlib
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

from fusionbench import Dataset


@pytest.fixture
def common(monkeypatch):
    folder = Path(__file__).resolve().parents[1] / "docs" / "ensemble-study" / "external-coverage"
    monkeypatch.syspath_prepend(str(folder))
    return importlib.import_module("common")


def test_profile_matches_individual_providers_and_unordered_pairs(common):
    y = np.array([0, 0, 1, 2])
    # Both first rows have only wrong predictions; only the first is unanimous.
    top = np.array([[2, 2, 2], [1, 2, 1], [1, 1, 1], [2, 1, 0]])
    x = np.eye(3)[top]
    x[2, 2] = [0.5, 0.5, 0]  # tied top label chooses class zero
    top[2, 2] = 0
    out = common.profile_fit(x, y)
    marginal = ((x - np.eye(3)[y, None]) ** 2).sum(-1).mean(axis=0)
    expected = np.mean([top[:, i] != top[:, j] for i, j in combinations(range(3), 2)])
    assert out["mean_provider_brier"] == pytest.approx(marginal.mean())
    assert out["best_provider_brier"] == pytest.approx(marginal.min())
    assert out["mean_pairwise_top1_disagreement"] == pytest.approx(expected)
    assert out["unanimous_wrong_fraction"] == 0.25
    assert out["zero_fraction"] == np.mean(x == 0)
    assert out["mean_max_probability"] == x.max(axis=2).mean()
    assert common.profile_fit(x[:, :1], y)["mean_pairwise_top1_disagreement"] is None


def test_temperature_is_fitted_only_to_full_brier_and_ties_prefer_identity(common):
    x = np.tile([[[0.80, 0.20], [0.70, 0.30]]], (10, 1, 1))
    y = np.array([0] * 9 + [1])
    assert common.fit_mean_temperature(x, y) == 0.5
    np.testing.assert_allclose(common.mean_temperature(x, 0.5), [[0.9, 0.1]] * 10)
    assert common.fit_mean_temperature(np.full((10, 3, 5), 0.2), y) == 1.0
    multi = np.array([[[0.5, 0.2, 0.3], [0.3, 0.2, 0.5]]])
    squared = multi.mean(axis=1) ** 2
    np.testing.assert_allclose(common.mean_temperature(multi, 0.5), squared / squared.sum(axis=1, keepdims=True))


def test_evaluate_methods_freezes_fit_only_calibration_and_scores_heldout(common, monkeypatch):
    x = np.tile([[[0.80, 0.20], [0.70, 0.30]]], (10, 1, 1))
    y = np.array([0] * 9 + [1])
    z = x[:2].copy()
    heldout = np.array([1, 1])
    ds = Dataset("toy", "binary", 2, ["A", "B"], x, y, z, heldout)

    class RecordingNative:
        def __init__(self, *, seed, l2_strength):
            assert seed == 42 and l2_strength == 0.25

        def fit(self, preds, labels, *, task, n_classes):
            assert preds is ds.fit_preds and labels is ds.fit_y
            assert task == "binary" and n_classes == 2
            self.chosen_ = "flat"

        def predict(self, preds):
            assert preds is ds.test_preds
            return preds.mean(axis=1)

    monkeypatch.setattr(common, "NativePatternOKAPI", RecordingNative)
    by_name = {row["method"]: row for row in common.evaluate_methods(ds, seed=42)}
    assert set(by_name) == {"mean", "native", "top_n", "prob_lr", "logit", "symmetric_lr", "pooled_lr", "temperature", "sharpener"}
    assert by_name["temperature"]["temperature"] == 0.5
    assert by_name["sharpener"]["temperature"] == 0.5
    assert by_name["temperature"]["brier"] == pytest.approx(1.62)
    assert by_name["temperature"]["brier"] == by_name["sharpener"]["brier"]
    assert by_name["mean"]["brier"] == pytest.approx(1.125)
    assert by_name["mean"]["accuracy"] == 0
    assert by_name["native"]["native_choice"] == "flat"
    assert by_name["top_n"]["top_n"] == 1
    assert all(0 <= row["brier"] <= 2 and row["fit_seconds"] >= 0 for row in by_name.values())

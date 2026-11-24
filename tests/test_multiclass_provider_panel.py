"""Contracts for the label-first, paired continuous/finite-vote provider panel."""

import json
from importlib.resources import files

import numpy as np
import pytest

from fusionbench import generate_dataset, get_scenario, list_scenarios, load_dataset
from fusionbench.cli import main
from fusionbench.scenarios.registry import STANDARD_V1_SCENARIOS
from fusionbench.transfer import load_preset

PANEL = "multiclass_provider_panel"


def draw(seed=641, rows=80, **params):
    return get_scenario(PANEL, **params).sample(rows, np.random.default_rng(seed))


@pytest.mark.parametrize("classes", [3, 10])
def test_paired_voting_preserves_both_splits_and_other_providers(classes):
    params = {"seed": 227, "n_per_split": 79, "n_classes": classes}
    voted = generate_dataset(PANEL, **params)
    continuous = generate_dataset(PANEL, **params, votes=(0, 0))
    one_voted = generate_dataset(PANEL, **params, votes=(0, 7))
    assert voted.model_ids == ["provider_0", "provider_1", "provider_2"]
    assert voted.fit_oracle is None and voted.test_oracle is None
    for split in ("fit", "test"):
        x = getattr(voted, split + "_preds")
        soft = getattr(continuous, split + "_preds")
        one = getattr(one_voted, split + "_preds")
        np.testing.assert_array_equal(
            getattr(voted, split + "_y"), getattr(continuous, split + "_y")
        )
        np.testing.assert_array_equal(x[:, 0], soft[:, 0])
        np.testing.assert_array_equal(soft[:, 1], one[:, 1])
        np.testing.assert_array_equal(x[:, 2], one[:, 2])
        assert x.shape == (79, 3, classes)
        assert np.all(soft > 0) and np.any(x[:, 1:] == 0)
        np.testing.assert_allclose(x.sum(axis=2), 1, atol=1e-14)
        for j, votes in [(1, 31), (2, 7)]:
            counts = x[:, j] * votes
            np.testing.assert_allclose(counts, np.rint(counts), rtol=0, atol=1e-13)
            np.testing.assert_array_equal(np.rint(counts).sum(axis=1), votes)


@pytest.mark.parametrize("classes", [3, 10])
def test_directed_confusion_changes_only_designated_class_odds_on_hard_rows(classes):
    params = {"n_classes": classes, "hard_fraction": 0.45}
    x, y, meta = draw(**params, votes=(0, 0))
    altered, other_y, other_meta = draw(**params, votes=(0, 0), confusion_bias=3.0)
    hard = meta["hard"]
    assert hard.any() and not hard.all()
    np.testing.assert_array_equal(y, other_y)
    np.testing.assert_array_equal(hard, other_meta["hard"])
    expected = x.copy()
    for j, temperature in enumerate((1.5, 1.0, 0.75)):
        wrong = (y[hard] + 1 + j % (classes - 1)) % classes
        expected[np.flatnonzero(hard), j, wrong] *= np.exp(3.0 / temperature)
    expected /= expected.sum(axis=2, keepdims=True)
    np.testing.assert_allclose(altered, expected, rtol=1e-13, atol=1e-15)
    voted, _, _ = draw(**params)
    explicit_zero, _, _ = draw(**params, confusion_bias=0.0)
    bias, _, _ = draw(**params, confusion_bias=3.0)
    np.testing.assert_array_equal(voted, explicit_zero)
    np.testing.assert_array_equal(voted[~hard], bias[~hard])


def test_difficulty_and_strength_rescale_correct_class_odds():
    params = {"hard_fraction": 0.4, "hard_scale": 0.55, "votes": (0, 0)}
    low, y, info = draw(**params, separation=2.5)
    high, y_high, info_high = draw(**params, separation=6.0)
    np.testing.assert_array_equal(y, y_high)
    np.testing.assert_array_equal(info["hard"], info_high["hard"])
    expected = low.copy()
    for j, (strength, temperature) in enumerate(
        zip((1.0, 0.9, 0.8), (1.5, 1.0, 0.75), strict=True)
    ):
        expected[np.arange(len(y)), j, y] *= np.exp(
            (6.0 - 2.5) * strength * np.where(info["hard"], 0.55, 1.0) / temperature
        )
    expected /= expected.sum(axis=2, keepdims=True)
    np.testing.assert_allclose(high, expected, rtol=1e-13, atol=1e-15)


def test_shared_error_extremes_and_anonymous_provider_ids():
    params = {
        "temperatures": (1.0, 1.0, 1.0),
        "strengths": (1.0, 1.0, 1.0),
        "votes": (0, 0),
    }
    shared, y, _ = draw(**params, shared_error=1.0)
    private, yp, _ = draw(**params, shared_error=0.0)
    np.testing.assert_array_equal(y, yp)
    for j in (1, 2):
        np.testing.assert_array_equal(shared[:, 0], shared[:, j])
        assert not np.array_equal(private[:, 0], private[:, j])


def test_registry_cli_and_frozen_v1_inventories(tmp_path):
    assert PANEL in list_scenarios() and PANEL not in STANDARD_V1_SCENARIOS
    source = files("fusionbench").joinpath("data", "reference-v1", "manifest.json")
    archived = json.loads(source.read_bytes())["banks"]
    assert len(archived) == 19 and set(archived) == set(STANDARD_V1_SCENARIOS)
    assert tuple(load_preset("standard")["scenarios"]) == tuple(STANDARD_V1_SCENARIOS)
    main(
        [
            "generate",
            "--scenarios",
            PANEL,
            "--n",
            "16",
            "--seed",
            "51",
            "--out",
            str(tmp_path),
            "--param",
            "n_classes=3",
            "--param",
            "votes=[0,7]",
            "--param",
            "confusion_bias=3",
        ]
    )
    (path,) = tmp_path.iterdir()
    ds = load_dataset(path)
    assert ds.meta["scenario_id"] == PANEL and ds.meta["parameters"]["votes"] == [0, 7]
    assert ds.test_preds.shape == (16, 3, 3) and ds.test_oracle is None


@pytest.mark.parametrize(
    "opts",
    [
        {"n_classes": 2},
        {"n_classes": 3.5},
        {"separation": -1},
        {"separation": True},
        {"hard_fraction": 1.1},
        {"hard_scale": -0.1},
        {"shared_error": np.nan},
        {"strengths": (1, 2)},
        {"strengths": (-1, 1, 1)},
        {"temperatures": (1.0, 0.0, 1.0)},
        {"votes": (0, 7.2)},
        {"votes": (0, -7)},
        {"votes": (True, 7)},
        {"confusion_bias": np.inf},
    ],
)
def test_invalid_controls_raise_before_sampling(opts):
    with pytest.raises((TypeError, ValueError)):
        get_scenario(PANEL, **opts)

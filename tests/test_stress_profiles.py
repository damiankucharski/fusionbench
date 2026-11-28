import json

import numpy as np
import pandas as pd
import pytest

from fusionbench import get_scenario
from fusionbench import profiles as profiles_api
from fusionbench.cli import main
from fusionbench.methods.averaging import SimpleAverage
from fusionbench.scenarios.registry import STANDARD_SCENARIOS


def test_all_settings_generate_valid_oracle_and_observed_probabilities():
    variants = profiles_api.list_profiles()
    assert set(variants) == set(STANDARD_SCENARIOS) and len(variants) == 30
    for name, arms in variants.items():
        assert len(arms) == 3 and arms["reference"] == {}
        for kwargs in arms.values():
            scenario = get_scenario(name, **kwargs)
            for split, n in [("fit", 32), ("test", 35)]:
                x, y, q, masks = profiles_api._draw(scenario, name, 913, split, n)
                assert x.shape == (
                    n,
                    len(scenario.model_ids()),
                    scenario.n_classes,
                ) and y.shape == (n,)
                np.testing.assert_allclose(x.sum(-1), 1, atol=2e-7)
                if q is not None:
                    assert np.isfinite(q).all()
                    np.testing.assert_allclose(q.sum(-1), 1, atol=2e-7)
                assert all(k.shape == (n,) and k.dtype == bool for k in masks.values())
    variants["weakest_link"]["rarer_matched_depth"]["m"] = 0
    assert profiles_api.list_profiles("weakest_link")["rarer_matched_depth"]["m"] == 4


@pytest.mark.parametrize(
    "name,arm",
    [
        ("committee", "fewer_providers"),
        ("multiclass_committee", "fewer_providers"),
        ("deadweight", "many_irrelevant"),
        ("needle", "more_irrelevant"),
    ],
)
def test_size_profiles_preserve_both_sets_of_labels_and_common_provider_prefix(
    name, arm
):
    for split, n in [("fit", 67), ("test", 89)]:
        drawn = []
        for profile in ["reference", arm]:
            sc = get_scenario(name, **profiles_api.list_profiles(name)[profile])
            drawn.append(profiles_api._draw(sc, name, 41, split, n))
        (x, y, *_), (p, target, *_) = drawn
        width = min(x.shape[1], p.shape[1])
        np.testing.assert_array_equal(y, target)
        np.testing.assert_array_equal(x[:, :width], p[:, :width])


def test_exact_duplicates_and_brier_preserving_confusion():
    sc = get_scenario(
        "saturated", **profiles_api.list_profiles("saturated")["identical_providers"]
    )
    x, _, _, _ = profiles_api._draw(sc, "saturated", 42, "test", 190)
    for provider in range(1, x.shape[1]):
        np.testing.assert_array_equal(x[:, provider], x[:, 0])
    frames = []
    for kwargs in profiles_api.list_profiles("similar_classes_38").values():
        sc = get_scenario("similar_classes_38", **kwargs)
        x, y, _, masks = profiles_api._draw(sc, "similar_classes_38", 42, "test", 190)
        frames.append((x, y, masks["affected_3_8"]))
    x0, y0, affected = frames[0]
    loss0 = np.square(x0 - np.eye(10)[y0][:, None]).sum(-1)
    for x, y, _ in frames[1:]:
        np.testing.assert_array_equal(y, y0)
        np.testing.assert_array_equal(x[~affected], x0[~affected])
        np.testing.assert_allclose(
            np.square(x - np.eye(10)[y][:, None]).sum(-1), loss0, atol=2e-14
        )


def test_macro_equal_weights_families_and_explicit_paired_intervals():
    rows = []
    for scenario, arms in [
        ("a", [("reference", 0.2), ("shift", 0.4)]),
        ("b", [("reference", 0.8)]),
    ]:
        for profile, val in arms:
            for seed, diff in enumerate([-0.10, -0.20]):
                for method, gap in [
                    ("simple_average", 0),
                    ("top_n", diff),
                ]:
                    rows.append(
                        {
                            "scenario": scenario,
                            "profile": profile,
                            "data_seed": seed,
                            "method": method,
                            "test_brier": val + gap,
                        }
                    )
    raw = pd.DataFrame(rows)
    tables = profiles_api.summarize_profiles(raw)
    macro = tables["macro"].set_index(["panel", "method"])
    assert macro.loc[("all_profiles", "simple_average"), "mean_brier"] == pytest.approx(
        0.55
    )
    assert macro.loc[("reference", "simple_average"), "mean_brier"] == pytest.approx(
        0.5
    )
    contrast = tables["paired"].set_index("panel").loc["all_profiles"]
    assert contrast.difference == pytest.approx(-0.15)
    assert contrast.lower_95 < -0.15 < contrast.upper_95
    with pytest.raises(ValueError, match="Each bank"):
        profiles_api.summarize_profiles(raw.iloc[:-1])


def test_cli_banks_controls_groups_and_manifest(tmp_path, monkeypatch):
    # Use an inexpensive stand-in to exercise CLI orchestration and mandatory references.
    seen = []

    class Fake(SimpleAverage):
        def fit(self, x, y, **options):
            seen.append(self.seed)
            return super().fit(x, y, **options)

    monkeypatch.setitem(profiles_api.FACTORIES, "okapi", Fake)
    main(
        [
            "profiles",
            "--scenario",
            "polarity_fault",
            "--methods",
            "okapi",
            "--seeds",
            "7",
            "--search-seed",
            "71",
            "--fit-rows",
            "45",
            "--test-rows",
            "65",
            "--out",
            str(tmp_path),
        ]
    )
    scores = pd.read_csv(tmp_path / "cells.csv")
    assert len(scores) == 12 and set(scores.method) == {
        "okapi",
        *profiles_api.DEFAULT_METHODS,
    }
    assert (scores.data_seed == 7).all() and seen == [71] * 3
    assert (scores[scores.method == "top_n"].top_n_selected >= 1).all()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert (
        manifest["rng_streams"] == {"fit": 0, "test": 1}
        and len(manifest["profiles"]) == 3
    )
    groups = pd.read_csv(tmp_path / "groups.csv")
    none = groups[(groups.profile == "no_flips") & (groups.group == "flipped")]
    assert (
        len(none) == 4 and (none.test_rows == 0).all() and none.test_brier.isna().all()
    )
    positive = (
        groups[groups.profile == "frequent_flips"].groupby("method").test_rows.sum()
    )
    assert (positive == 65).all()

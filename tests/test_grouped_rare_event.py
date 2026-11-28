"""Bench-facing contracts and controls of the three versioned grouped-scene scenarios."""

import numpy as np
import pytest
from scipy.special import expit, logit

from fusionbench import generate_dataset, get_scenario, load_reference_dataset
from fusionbench.profiles import _draw, list_profiles
from fusionbench.scenarios.grouped_rare_event import GroupedRareEvent
from fusionbench.transfer import load_preset

ID = "grouped_rare_event"


def draw(rows=43, **knobs):
    return get_scenario(ID, **knobs).sample(rows, np.random.default_rng(183))


@pytest.mark.parametrize(
    "scenario_id", [ID, "shared_error_bloc", "rare_scene_specialist"]
)
def test_version_six_appends_three_slots_and_bundles_their_banks(scenario_id):
    v5 = load_preset("standard-v5")
    v6 = load_preset("standard-v6")
    assert len(v5["scenarios"]) == 27
    new = {
        key: {"seed_slot": idx}
        for idx, key in enumerate(
            (ID, "shared_error_bloc", "rare_scene_specialist"), start=27
        )
    }
    assert v6 == {**v5, "scenarios": {**v5["scenarios"], **new}}
    assert load_preset("standard") == v6
    live = generate_dataset(scenario_id, seed=1729, n_per_split=128)
    frozen = load_reference_dataset(scenario_id)
    assert frozen.meta["reference_bank"] == "reference-v6"
    for split in ("fit", "test"):
        np.testing.assert_array_equal(
            getattr(frozen, f"{split}_y"), getattr(live, f"{split}_y")
        )
        np.testing.assert_allclose(
            getattr(frozen, f"{split}_preds"),
            getattr(live, f"{split}_preds"),
            atol=3e-8,
        )
        assert (
            frozen.meta["evaluation_groups"][split]
            == live.meta["evaluation_groups"][split]
        )
        assert getattr(frozen, f"{split}_oracle") is None


@pytest.mark.parametrize("n", [1, 3, 8, 34])
def test_partial_final_scene_and_heldout_metadata(n):
    x, y, latent = draw(n)
    np.testing.assert_array_equal(latent["scene_id"], np.arange(n) // 8)
    np.testing.assert_array_equal(
        latent["specialist_region"],
        latent["specialist_region"][::8][latent["scene_id"]],
    )
    np.testing.assert_array_equal(latent["posterior"], expit(1.8 * latent["signal"]))
    assert x.shape == (n, 8, 2) and y.shape == (n,)
    np.testing.assert_allclose(x.sum(2), 1)


def test_specialist_switch_is_local_and_preserves_underlying_events():
    kwargs = {"specialist_rate": 0.5, "heterogeneity": 0.3, "common_error_share": 0.9}
    base, y, latent = draw(1001, specialist_strength=0, **kwargs)
    oracle, labels, fields = draw(1001, specialist_strength=1, **kwargs)
    mask = latent["specialist_region"]
    assert mask.any() and (~mask).any()
    np.testing.assert_array_equal(base[:, :3], oracle[:, :3])
    np.testing.assert_array_equal(base[:, 4:], oracle[:, 4:])
    np.testing.assert_array_equal(base[~mask], oracle[~mask])
    np.testing.assert_array_equal(oracle[mask, 3, 1], latent["posterior"][mask])
    np.testing.assert_array_equal(y, labels)
    for key in fields:
        np.testing.assert_array_equal(fields[key], latent[key])
    _, _, _, groups = _draw(get_scenario(ID), ID, 25, "test", 128)
    np.testing.assert_array_equal(
        groups["specialist_region"], ~groups["outside_specialist_region"]
    )
    assert len(list_profiles(ID)) == 3


def test_common_error_heterogeneity_and_scene_persistence_endpoints():
    # With a completely shared innovation, provider differences are known scalings.
    kwargs = {"common_error_share": 1, "scene_error_share": 1, "specialist_strength": 0}
    base, labels, latent = draw(67, heterogeneity=0, **kwargs)
    np.testing.assert_array_equal(base, np.broadcast_to(base[:, :1], base.shape))
    scaled, labels2, latent2 = draw(67, heterogeneity=1, **kwargs)
    np.testing.assert_array_equal(labels2, labels)
    np.testing.assert_array_equal(latent2["signal"], latent["signal"])
    resid = logit(scaled[:, :, 1]) - 1.3 * latent["signal"][:, None]
    np.testing.assert_allclose(resid[:, -1], np.exp(2) * resid[:, 0], atol=1e-11)
    np.testing.assert_allclose(resid, resid[::8][latent["scene_id"]], atol=1e-11)
    scaled_rms = np.sqrt(np.mean(resid**2, axis=1))
    np.testing.assert_allclose(
        scaled_rms, np.abs(logit(base[:, 0, 1]) - 1.3 * latent["signal"]), atol=1e-11
    )


def test_offset_changes_prevalence_monotonically_on_paired_draws():
    _, frequent, _ = draw(6400, event_offset=0)
    _, rare, _ = draw(6400, event_offset=2)
    assert np.all(rare <= frequent)
    assert 0.07 < rare.mean() < 0.13
    assert 0.44 < frequent.mean() < 0.56


@pytest.mark.parametrize(
    "options",
    [
        {"bloc_size": True},
        {"bloc_size": 9},
        {"scene_size": True},
        {"scene_size": 0},
        {"scene_size": 1.2},
        {"event_offset": np.nan},
        {"heterogeneity": -1},
        {"common_error_share": 1.1},
        {"scene_error_share": np.inf},
        {"specialist_rate": False},
        {"specialist_strength": np.nan},
    ],
)
def test_invalid_controls(options):
    with pytest.raises(ValueError, match=next(iter(options))):
        GroupedRareEvent(**options)


def test_bloc_members_exchange_quality_for_redundancy_without_moving_outsiders():
    n = 16000
    baseline = get_scenario("shared_error_bloc", common_error_share=0, scene_size=1)
    bloc = get_scenario("shared_error_bloc", scene_size=1)
    x0, y0, latent = baseline.sample(n, np.random.default_rng(77))
    x, y, _ = bloc.sample(n, np.random.default_rng(77))
    np.testing.assert_array_equal(y, y0)
    np.testing.assert_array_equal(x[:, 6:], x0[:, 6:])
    errors = logit(x[:, :, 1]) - 1.3 * latent["signal"][:, None]
    corr = np.corrcoef(errors, rowvar=False)
    np.testing.assert_allclose(corr[:6, :6][np.triu_indices(6, 1)], 0.85, atol=0.025)
    assert np.abs(corr[:6, 6:]).max() < 0.035
    assert np.ptp(errors.std(axis=0)) < 0.06
    empty = get_scenario("shared_error_bloc", bloc_size=0, scene_size=1)
    x_empty, _, _ = empty.sample(n, np.random.default_rng(77))
    np.testing.assert_array_equal(x0, x_empty)


def test_specialist_base_has_equal_providers_outside_region_and_exact_expert_inside():
    specialist = get_scenario("rare_scene_specialist")
    x, y, latent = specialist.sample(2048, np.random.default_rng(111))
    neutral = get_scenario("rare_scene_specialist", specialist_strength=0)
    xn, yn, _ = neutral.sample(2048, np.random.default_rng(111))
    mask = latent["specialist_region"]
    np.testing.assert_array_equal(y, yn)
    np.testing.assert_array_equal(x[~mask], xn[~mask])
    np.testing.assert_array_equal(x[mask, 3, 1], latent["posterior"][mask])
    assert not np.array_equal(x[mask], xn[mask])
    assert (
        specialist.knobs["heterogeneity"] == specialist.knobs["common_error_share"] == 0
    )

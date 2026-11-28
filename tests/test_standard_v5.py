"""Check versioned suite, offline bank, and the two fit/test sampling interfaces."""

import numpy as np
import pytest

from fusionbench import generate_dataset, get_scenario, load_reference_dataset
from fusionbench.eval.transfer_grid import STREAMS, iter_test_conditions
from fusionbench.scenarios.registry import STANDARD_V4_SCENARIOS, STANDARD_V5_SCENARIOS
from fusionbench.transfer import load_preset

ID = "rare_shortcut_noise"


def test_standard_v5_appends_two_seed_slots_to_immutable_v4():
    old = load_preset("standard-v4")
    new = load_preset("standard-v5")
    assert list(STANDARD_V4_SCENARIOS) == list(old["scenarios"])
    assert list(STANDARD_V5_SCENARIOS) == [
        *STANDARD_V4_SCENARIOS,
        ID,
        "stratified_pool",
    ]
    assert list(new["scenarios"]) == list(STANDARD_V5_SCENARIOS)
    assert new == {
        **old,
        "scenarios": {
            **old["scenarios"],
            ID: {"seed_slot": 25},
            "stratified_pool": {"seed_slot": 26},
        },
    }


@pytest.mark.parametrize(
    "scenario_id, group", [(ID, "region"), ("stratified_pool", "high_amplitude")]
)
def test_offline_v5_bank_matches_public_split_generator(scenario_id, group):
    saved = load_reference_dataset(scenario_id)
    current = generate_dataset(scenario_id, n_per_split=128, seed=1729)
    assert saved.meta["reference_bank"] == "reference-v5"
    assert saved.meta["evaluation_groups"] == current.meta["evaluation_groups"]
    assert set(saved.meta["evaluation_groups"]["test"]) == {group}
    assert saved.model_ids == current.model_ids
    for split in ("fit", "test"):
        for name in ("y", "preds", "oracle"):
            actual = getattr(saved, f"{split}_{name}")
            expectation = getattr(current, f"{split}_{name}").astype(actual.dtype)
            np.testing.assert_array_equal(actual, expectation)


def test_transfer_grid_pairs_source_and_shifted_rows_with_labels():
    seed, n = 41, 1024
    sc = get_scenario(ID)
    clean, source_y, _ = sc.sample_split(
        n, np.random.default_rng([seed, STREAMS["clean_test"]]), "fit"
    )
    shifted, target_y, _ = sc.sample_split(
        n, np.random.default_rng([seed, STREAMS["canonical_shift"]]), "test"
    )
    threshold = np.random.default_rng([seed, STREAMS["row_threshold"]]).random(n)
    rows = list(
        iter_test_conditions(
            ID, seed, n, modes=("canonical_shift",), severities=(0.5, 1)
        )
    )
    assert [(mode, s) for mode, s, _, _ in rows] == [
        ("clean", 0),
        ("canonical_shift", 0.5),
        ("canonical_shift", 1),
    ]
    for _, s, x, y in rows:
        mask = threshold < s
        np.testing.assert_array_equal(x, np.where(mask[:, None, None], shifted, clean))
        np.testing.assert_array_equal(y, np.where(mask, target_y, source_y))


def test_profiles_expose_balanced_fit_and_test_amplitude_groups():
    from fusionbench.profiles import _draw

    sc = get_scenario("stratified_pool")
    for split in ("fit", "test"):
        _, labels, _, groups = _draw(sc, "stratified_pool", 71, split, 128)
        for group in ("high_amplitude", "outside_high_amplitude"):
            np.testing.assert_array_equal(np.bincount(labels[groups[group]]), [32, 32])

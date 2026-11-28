"""Paired controls for redistributing a provider's wrong-class probabilities."""

import numpy as np
import pytest

from fusionbench import (
    generate_dataset,
    get_scenario,
    list_scenarios,
    load_dataset,
    save_dataset,
)
from fusionbench.scenarios.multiclass_provider_panel import MulticlassProviderPanel
from fusionbench.scenarios.registry import STANDARD_V1_SCENARIOS

ID = "shared_confusion_panel"


def matrices():
    shared = np.zeros((10, 10))
    shared[3, 8] = shared[8, 3] = 1
    private = np.zeros((3, 10, 10))
    for provider in range(3):
        private[provider, [3, 8], provider] = 1
    return shared, private


def test_paired_defaults_and_external_rng_across_splits():
    assert ID in list_scenarios() and ID not in STANDARD_V1_SCENARIOS
    shared, private = matrices()
    options = [
        {},
        {
            "shared_matrix": shared,
            "provider_matrices": private,
            "deviation_mix": 1,
            "strength": 0,
        },
        {"strength": 1},
    ]
    base = get_scenario("multiclass_provider_panel")
    for kwargs in options:
        panel = get_scenario(ID, **kwargs)
        rng, paired = np.random.default_rng(79), np.random.default_rng(79)
        for n in (67, 52):
            x, y, latent = base.sample(n, rng)
            altered, injected_y, injected_latent = panel.sample(n, paired)
            np.testing.assert_array_equal(x, altered)
            np.testing.assert_array_equal(y, injected_y)
            np.testing.assert_array_equal(latent["hard"], injected_latent["hard"])
        assert rng.random() == paired.random()


@pytest.mark.parametrize("votes", [(31, 7), (0, 0)])
def test_swap_preserves_true_probability_and_each_wrong_value_and_loss(votes):
    shared, private = matrices()
    kwargs = {"n_per_split": 211, "seed": 53, "votes": votes, "confusion_bias": 1.7}
    original = generate_dataset("multiclass_provider_panel", **kwargs)
    for mix in (0.0, 0.5, 1.0):
        result = generate_dataset(
            ID,
            shared_matrix=shared,
            provider_matrices=private,
            deviation_mix=mix,
            strength=1,
            **kwargs,
        )
        assert (
            original.model_ids == result.model_ids
            and result.fit_oracle is None
            and result.test_oracle is None
        )
        for split in ("fit", "test"):
            before = getattr(original, split + "_preds")
            after = getattr(result, split + "_preds")
            y = getattr(result, split + "_y")
            np.testing.assert_array_equal(y, getattr(original, split + "_y"))
            rows = np.arange(len(y))
            np.testing.assert_array_equal(after[rows, :, y], before[rows, :, y])
            is_wrong = np.broadcast_to(np.arange(10) != y[:, None, None], before.shape)
            wrong = before[is_wrong].reshape(len(y), 3, 9)
            wrong_after = after[is_wrong].reshape(len(y), 3, 9)
            np.testing.assert_array_equal(
                np.sort(wrong, axis=2), np.sort(wrong_after, axis=2)
            )
            labels = np.eye(10)[y, None]
            np.testing.assert_allclose(
                ((before - labels) ** 2).sum(axis=2),
                ((after - labels) ** 2).sum(axis=2),
                atol=2e-15,
                rtol=0,
            )
            affected = (y == 3) | (y == 8)
            np.testing.assert_array_equal(after[~affected], before[~affected])
            assert np.any(after[affected] != before[affected])
            for j in range(3):
                shared_max = after[rows[affected], j, 11 - y[affected]] == wrong[
                    affected, j
                ].max(axis=1)
                private_max = after[affected, j, j] == wrong[affected, j].max(axis=1)
                assert np.all(shared_max | private_max)
                if mix == 0:
                    assert shared_max.all()
                if mix == 1:
                    assert private_max.all()


def test_nested_strength_and_relative_destination_weights():
    shared = np.zeros((3, 3))
    shared[0] = (0, 1, 3)
    kwargs = {"n_classes": 3, "votes": (0, 0), "shared_matrix": shared}

    def sample(**params):
        return get_scenario(ID, **kwargs, **params).sample(
            3500, np.random.default_rng(83)
        )[:2]

    original, y = sample(strength=0)
    middle, _ = sample(strength=0.4)
    full, _ = sample(strength=1)
    fallback, _ = sample(strength=1, deviation_mix=1)
    np.testing.assert_array_equal(fallback, full)
    changed = (middle != original).any(axis=2)
    assert changed.any() and (np.any(full != original, axis=2) & ~changed).any()
    np.testing.assert_array_equal(middle[changed], full[changed])
    assert np.mean(full[y == 0, :, 2] > full[y == 0, :, 1]) == pytest.approx(
        0.75, abs=0.035
    )


def test_tied_largest_wrong_values_use_both_sources(monkeypatch):
    bins = np.array([0.5, 0.25, 0.25, 0])

    def fixed_bank(self, n, latent_seed, vote_seeds):
        return (
            np.broadcast_to(bins, (n, 3, len(bins))).copy(),
            np.zeros(n, dtype=int),
            {},
        )

    monkeypatch.setattr(MulticlassProviderPanel, "_sample_with_seeds", fixed_bank)
    weights = np.zeros((4, 4))
    weights[0, 3] = 1
    panel = get_scenario(ID, n_classes=4, shared_matrix=weights, strength=1)
    x, _, _ = panel.sample(1000, np.random.default_rng(43))
    np.testing.assert_array_equal(x[:, :, 3], 0.25)
    assert np.mean(x[:, :, 1] == 0.0) == pytest.approx(0.5, abs=0.035)


def test_public_config_recorded_and_roundtrips(tmp_path):
    shared, private = matrices()
    kwargs = {
        "shared_matrix": shared,
        "provider_matrices": private,
        "deviation_mix": 0.25,
        "strength": 0.8,
    }
    bank = generate_dataset(ID, **kwargs, seed=14, n_per_split=50)
    result = load_dataset(save_dataset(bank, tmp_path))
    assert (
        result.meta["scenario_id"] == ID
        and result.meta["parameters"]["shared_matrix"] == shared.tolist()
    )
    assert result.meta["parameters"]["provider_matrices"] == private.tolist()
    replay = generate_dataset(ID, **result.meta["parameters"], seed=14, n_per_split=50)
    np.testing.assert_array_equal(replay.test_preds, bank.test_preds)
    assert (
        replay.name == bank.name != generate_dataset(ID, seed=14, n_per_split=50).name
    )


@pytest.mark.parametrize(
    "opts",
    [
        {"strength": 1.01},
        {"deviation_mix": -0.1},
        {"strength": True},
        {"shared_matrix": np.ones((2, 2))},
        {"shared_matrix": np.eye(3)},
        {"shared_matrix": [[0, -1, 0], [0, 0, 0], [0, 0, 0]]},
        {"shared_matrix": [[0, np.inf, 0], [0, 0, 0], [0, 0, 0]]},
        {"provider_matrices": np.zeros((2, 3, 3))},
        {"provider_matrices": [np.eye(3)] * 3},
    ],
)
def test_invalid_parameters(opts):
    with pytest.raises((TypeError, ValueError)):
        get_scenario(ID, n_classes=3, **opts)


def test_similar_classes_is_a_standard_regression_case_on_both_splits():
    from fusionbench.cli import _scenarios
    from fusionbench.scenarios.registry import default_suite
    from fusionbench.transfer import load_preset

    scenario_id = "similar_classes_38"
    expected = [*STANDARD_V1_SCENARIOS, scenario_id, "rare_specialist", "redundant_pool",
                "environment_shortcut", "selective_fallback", "cloned_shortcut", "rare_shortcut_noise", "stratified_pool"]
    assert [s._registry_metadata["scenario_id"] for s in default_suite()] == expected
    assert [s._registry_metadata["scenario_id"] for s in _scenarios(None)] == expected
    assert list(load_preset("standard")["scenarios"]) == expected
    assert list(load_preset("standard-v2")["scenarios"]) == expected[:20]
    assert list(load_preset("standard-v3")["scenarios"]) == expected[:24]
    assert list(load_preset("standard-v4")["scenarios"]) == expected[:25]
    assert load_preset("standard") == load_preset("standard-v5")
    data = get_scenario(scenario_id).build(n_per_split=1024, seed=37)
    base = get_scenario("multiclass_provider_panel").build(n_per_split=1024, seed=37)
    for split in ("fit", "test"):
        p = getattr(data, f"{split}_preds")
        y = getattr(data, f"{split}_y")
        np.testing.assert_array_equal(y, getattr(base, f"{split}_y"))
        other = ~np.isin(y, [3, 8])
        np.testing.assert_array_equal(p[other], getattr(base, f"{split}_preds")[other])
        for label, partner in ((3, 8), (8, 3)):
            rows = p[y == label].copy()
            assert len(rows) > 0
            rows[:, :, label] = -1
            np.testing.assert_array_equal(rows[:, :, partner], rows.max(axis=2))

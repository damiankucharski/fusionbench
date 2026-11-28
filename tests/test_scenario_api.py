"""Public scenario discovery, generation, reproducibility and extension contracts."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from fusionbench import (
    Dataset,
    Scenario,
    __version__,
    generate_dataset,
    get_scenario,
    list_scenarios,
    load_dataset,
    register_scenario,
    save_dataset,
)
from fusionbench.scenarios.blocks import draw_block
from fusionbench.scenarios.committee import Committee
from fusionbench.scenarios.registry import SCENARIOS


@pytest.mark.parametrize("scenario_id", list_scenarios())
def test_public_datasets_match_historical_generator_and_oracle(scenario_id):
    rows, seed = 37, 1793
    historical = SCENARIOS[scenario_id]()
    rng = np.random.default_rng(seed)
    dataset = generate_dataset(scenario_id, n_per_split=rows, seed=seed)
    assert isinstance(dataset, Dataset)
    assert dataset.model_ids == historical.model_ids()
    assert dataset.name == historical.name
    assert dataset.meta["scenario_id"] == scenario_id
    assert dataset.meta["seed"] == seed and dataset.meta["n_per_split"] == rows
    assert dataset.meta["fusionbench_version"] == __version__
    assert dataset.meta["parameters"] == {}
    for split in ("fit", "test"):
        if scenario_id in ("environment_shortcut", "cloned_shortcut", "rare_shortcut_noise"):
            x, y, latent = historical.sample_split(rows, rng, split=split)
        else:
            x, y, latent = historical._generate(rows, rng)
        np.testing.assert_array_equal(x, getattr(dataset, f"{split}_preds"))
        np.testing.assert_array_equal(y, getattr(dataset, f"{split}_y"))
        expected_oracle = historical.bayes_oracle(x, latent)
        if expected_oracle is None:
            assert getattr(dataset, f"{split}_oracle") is None
        else:
            np.testing.assert_array_equal(
                expected_oracle, getattr(dataset, f"{split}_oracle")
            )


def test_configured_build_and_factory_share_rng_and_provenance():
    opts = {"k": np.int64(3), "n_classes": 4}
    scenario = get_scenario("multiclass_committee", **opts)
    rows = 42
    one = scenario.build(n_per_split=rows, seed=712)
    with ThreadPoolExecutor(max_workers=2) as pool:
        other_seed = pool.submit(scenario.build, rows, 723)
        same = pool.submit(
            generate_dataset, "multiclass_committee", n_per_split=rows, seed=712, **opts
        )
    two = same.result()
    assert one.meta == two.meta
    assert one.meta["parameters"] == {"k": 3, "n_classes": 4}
    assert one.meta["effective_parameters"]["accuracy"] == 0.6
    assert one.fit_preds.shape == (rows, 3, 4)
    np.testing.assert_array_equal(one.fit_preds, two.fit_preds)
    np.testing.assert_array_equal(one.test_oracle, two.test_oracle)
    assert not np.array_equal(one.fit_preds, one.test_preds)
    assert not np.array_equal(one.fit_preds, other_seed.result().fit_preds)
    with pytest.raises(ValueError, match="Unknown constructor knob.*accracy"):
        get_scenario("needle", accracy=0.8)
    with pytest.raises(ValueError, match="Unknown scenario"):
        get_scenario("missing_id")


def test_custom_sample_hook_and_numpy_parameters_roundtrip(tmp_path):
    class CustomBinary(Scenario):
        n_classes = 2

        def __init__(self, **knobs):
            self.weights = np.asarray(knobs["weights"])
            super().__init__(name="custom_binary", **knobs)

        def model_ids(self):
            return [f"channel_{i}" for i in range(len(self.weights))]

        def sample(self, n, rng):
            p = rng.random((n, len(self.weights))) * self.weights
            y = (p[:, 0] > 0.5).astype(int)
            return np.stack((1.0 - p, p), axis=-1), y, {}

    name = "api_test_custom"
    register_scenario(name, CustomBinary)
    try:
        assert list_scenarios()[-1] == name
        with pytest.raises(ValueError, match="already registered"):
            register_scenario(name, CustomBinary)
        ds = generate_dataset(
            name, n_per_split=23, seed=32, weights=np.array([0.65, 0.8])
        )
        assert ds.fit_oracle is None and ds.test_oracle is None
        assert ds.meta["parameters"] == ds.meta["knobs"] == {"weights": [0.65, 0.8]}
        folder = save_dataset(ds, tmp_path)
        loaded = load_dataset(folder)
        assert loaded.meta == ds.meta
        for split in ("fit", "test"):
            np.testing.assert_array_equal(
                getattr(loaded, f"{split}_y"), getattr(ds, f"{split}_y")
            )
            np.testing.assert_array_equal(
                getattr(loaded, f"{split}_preds"),
                getattr(ds, f"{split}_preds").astype(np.float32),
            )
        replay = generate_dataset(
            name,
            n_per_split=loaded.meta["n_per_split"],
            seed=loaded.meta["seed"],
            **loaded.meta["parameters"],
        )
        np.testing.assert_array_equal(replay.test_preds, ds.test_preds)
    finally:
        del SCENARIOS[name]


def test_legacy_and_public_hooks_share_overrides_and_super_dispatch():
    class Legacy(Committee):
        def _generate(self, n, rng):
            x, y, info = super()._generate(n, rng)
            return x[:, ::-1], y, info

    class Public(Legacy):
        def sample(self, n, rng):
            x, y, info = super().sample(n, rng)
            return x, 1 - y, info

    class NewLegacy(Public):
        def _generate(self, n, rng):
            x, y, info = super()._generate(n, rng)
            return x[:, :, ::-1], y, info

    n, seed = 28, 43
    original_x, original_y, _ = Committee()._generate(
        n, np.random.default_rng([seed, 7])
    )
    for generator, expected_x, expected_y in (
        (Legacy(), original_x[:, ::-1], original_y),
        (Public(), original_x[:, ::-1], 1 - original_y),
        (NewLegacy(), original_x[:, ::-1, ::-1], 1 - original_y),
    ):
        for x, y, _ in (
            generator.sample(n, np.random.default_rng([seed, 7])),
            generator._generate(n, np.random.default_rng([seed, 7])),
            draw_block(generator, seed, 7, n),
        ):
            np.testing.assert_array_equal(x, expected_x)
            np.testing.assert_array_equal(y, expected_y)


def test_registration_validation_and_invalid_generation_inputs():
    with pytest.raises(ValueError, match="already registered"):
        register_scenario("committee", Committee)
    with pytest.raises(TypeError, match="subclass"):
        register_scenario("fixture", dict)
    with pytest.raises(TypeError, match="concrete"):
        register_scenario("fixture", Scenario)
    with pytest.raises(ValueError, match="nonempty"):
        register_scenario("", Committee)
    with pytest.raises(ValueError, match="positive integer"):
        generate_dataset("committee", n_per_split=0)
    with pytest.raises(ValueError, match="nonnegative"):
        generate_dataset("committee", n_per_split=5, seed=-1)

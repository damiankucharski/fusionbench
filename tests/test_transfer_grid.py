import numpy as np
import pytest
from scipy.special import softmax

from fusionbench.eval.full_suite_point import SHIFTS, scenario_block
from fusionbench.eval.transfer_grid import (
    MODES,
    SEVERITIES,
    brier,
    development,
    iter_input_stresses,
    iter_test_conditions,
    noisy_labels,
    provider_swap_plan,
)
from fusionbench.scenarios.base import Scenario
from fusionbench.scenarios.blocks import default_target_shift
from fusionbench.scenarios.calibration import MulticlassCommittee
from fusionbench.scenarios.native_switch import NativeSwitch
from fusionbench.scenarios.registry import SCENARIOS


def _conditions(name, seed, n, **options):
    return {
        (mode, s): (x, y)
        for mode, s, x, y in iter_test_conditions(name, seed, n, **options)
    }


@pytest.mark.parametrize("name", SCENARIOS)
def test_entire_registry_has_shared_clean_source_and_normalized_test_conditions(name):
    n, seed = 24, 391
    scenario = SCENARIOS[name]()
    source, source_y = development(name, seed, n)
    clean, clean_y, _ = scenario_block(scenario, seed, 0, n)
    np.testing.assert_array_equal(source, clean)
    np.testing.assert_array_equal(source_y, clean_y)
    x, y, _ = scenario_block(scenario, seed, 1, n)
    grid = _conditions(name, seed, n)
    np.testing.assert_array_equal(grid["clean", 0][0], x)
    np.testing.assert_array_equal(grid["clean", 0][1], y)
    assert len(grid) == (10 if default_target_shift(name) is not None else 7)
    for (mode, s), (test_x, test_y) in grid.items():
        assert mode in (*MODES, "clean") and s in (*SEVERITIES, 0.0)
        assert test_x.shape == x.shape
        assert np.isfinite(test_x).all() and np.min(test_x) >= 0
        np.testing.assert_allclose(test_x.sum(-1), 1, atol=1e-12)
        if mode != "canonical_shift":
            np.testing.assert_array_equal(test_y, y)


def test_nested_multiclass_annotation_errors_are_uniform_wrong_classes_and_repeatable():
    seed, n = 120, 1800
    x, y = development("multiclass_committee", seed, n)
    small_x, small = development("multiclass_committee", seed, n, 0.15)
    large_x, large = development("multiclass_committee", seed, n, 0.30)
    np.testing.assert_array_equal(small_x, x)
    np.testing.assert_array_equal(large_x, x)
    assert np.all(large[small != y] == small[small != y])
    assert 0.10 < np.mean(small != y) < 0.20
    assert 0.24 < np.mean(large != y) < 0.36
    np.testing.assert_array_equal(large, noisy_labels(y, 5, seed, 0.30))
    wrong = noisy_labels(y, 5, seed, 1.0)
    assert (wrong != y).all()
    for c in range(5):
        assert set(wrong[y == c]) == set(range(5)) - {c}


def test_derangement_donor_vectors_nested_masks_and_repeatability():
    seed, n = 410, 120
    grid = _conditions("multiclass_committee", seed, n)
    x, y = grid["clean", 0]
    j, donors, threshold = provider_swap_plan(n, x.shape[1], seed)
    assert np.all(donors != np.arange(n))
    np.testing.assert_array_equal(np.sort(donors), np.arange(n))
    previous = np.zeros(n, dtype=bool)
    for s in SEVERITIES:
        swapped, labels = grid["provider_swap", s]
        changed = np.any(x != swapped, axis=(1, 2))
        np.testing.assert_array_equal(changed, threshold < s)
        assert np.all(changed[previous])
        previous = changed
        expected = x.copy()
        expected[changed, j] = x[donors[changed], j]
        np.testing.assert_array_equal(swapped, expected)
        np.testing.assert_array_equal(labels, y)
    assert previous.all()


@pytest.mark.parametrize("name", ["committee", "multiclass_committee"])
def test_jitter_uses_the_same_gaussian_draw_at_fixed_scale(name):
    seed, n = 921, 96
    grid = _conditions(name, seed, n)
    x, y = grid["clean", 0]
    logs = np.log(x)
    logs -= logs.mean(axis=-1, keepdims=True)
    noise = np.random.default_rng([seed, 13]).standard_normal(x.shape)
    # Filtering and reversing the modes or severities must preserve exact draws.
    reordered = _conditions(
        name,
        seed,
        n,
        modes=("log_jitter", "provider_swap"),
        severities=(1.0, 0.0, 0.25),
    )
    assert len(reordered) == 5
    for s in SEVERITIES:
        jitter, labels = grid["log_jitter", s]
        np.testing.assert_allclose(
            jitter, softmax(logs + 1.5 * s * noise, axis=-1), atol=1e-14
        )
        np.testing.assert_array_equal(y, labels)
        if s in (0.25, 1.0):
            for mode in ("log_jitter", "provider_swap"):
                np.testing.assert_array_equal(reordered[mode, s][0], grid[mode, s][0])
    # The source-only operator has no knowledge of the scenario or its labels.
    [(mode, s, from_arrays)] = iter_input_stresses(
        x, seed, severities=(0.5,), modes=("log_jitter",)
    )
    assert (mode, s) == ("log_jitter", 0.5)
    np.testing.assert_array_equal(from_arrays, grid["log_jitter", 0.5][0])


@pytest.mark.parametrize("name", SHIFTS)
def test_canonical_endpoint_and_all_intermediate_prediction_label_pairs(name):
    n, seed = 45, 314
    clean, labels, _ = scenario_block(SCENARIOS[name](), seed, 1, n)
    endpoint, ey, _ = scenario_block(SCENARIOS[name](**SHIFTS[name]), seed, 2, n)
    grid = _conditions(name, seed, n, modes=("canonical_shift",))
    threshold = np.random.default_rng([seed, 11]).random(n)
    for s in SEVERITIES:
        mask = threshold < s
        x, y = grid["canonical_shift", s]
        np.testing.assert_array_equal(x[mask], endpoint[mask])
        np.testing.assert_array_equal(y[mask], ey[mask])
        np.testing.assert_array_equal(x[~mask], clean[~mask])
        np.testing.assert_array_equal(y[~mask], labels[~mask])
    np.testing.assert_array_equal(grid["canonical_shift", 1][0], endpoint)
    np.testing.assert_array_equal(grid["canonical_shift", 1][1], ey)


def test_custom_generator_source_knobs_and_shift_carryover(monkeypatch):
    class CustomMulticlass(MulticlassCommittee):
        pass

    monkeypatch.setitem(SCENARIOS, "custom", CustomMulticlass)
    opts = {"k": 3, "n_classes": 6, "accuracy": 0.7}
    seed, n = 234, 40
    expected, ey, _ = scenario_block(
        CustomMulticlass(**{**opts, "accuracy": 0.85}), seed, 2, n
    )
    for name in ("custom", CustomMulticlass):
        kwargs = {
            "scenario_kwargs": opts,
            "modes": "canonical_shift",
            "severities": (0.0, 1.0),
        }
        assert len(_conditions(name, seed, n, **kwargs)) == 1
        grid = _conditions(name, seed, n, canonical_shift={"accuracy": 0.85}, **kwargs)
        endpoint, y = grid["canonical_shift", 1.0]
        np.testing.assert_array_equal(endpoint, expected)
        np.testing.assert_array_equal(y, ey)
        assert endpoint.shape == (n, 3, 6)
        with pytest.raises(ValueError, match="provider mapping"):
            _conditions(name, seed, n, canonical_shift={"k": 5}, **kwargs)


def test_builtin_class_lookup_preserves_knobs_and_shift_overrides():
    seed, n = 624, 22
    opts = {
        "scenario_kwargs": {"role_seed": 19, "mixing": 0.1},
        "modes": ("canonical_shift",),
        "severities": (1.0,),
    }
    grid = _conditions(NativeSwitch, seed, n, **opts)
    override = _conditions(
        "native_switch", seed, n, **opts, canonical_shift={"mixing": 0.71}
    )
    for rows, mixing in ((grid, 1.0), (override, 0.71)):
        expected, labels, _ = scenario_block(
            NativeSwitch(role_seed=19, mixing=mixing), seed, 2, n
        )
        np.testing.assert_array_equal(rows["canonical_shift", 1][0], expected)
        np.testing.assert_array_equal(rows["canonical_shift", 1][1], labels)


def test_brier_is_full_class_sum_and_clean_only_accepts_single_rows():
    assert brier([[0, 1], [0, 1]], [0, 1]) == 1.0
    assert brier([[0.0, 0.5, 0.5]], [0]) == 1.5
    assert list(_conditions("committee", 17, 1, modes="clean")) == [("clean", 0.0)]
    with pytest.raises(ValueError, match="two rows"):
        _conditions("committee", 17, 1)
    with pytest.raises(ValueError, match="Severities"):
        _conditions("committee", 17, 10, severities=(1.5,))


def test_transfer_never_calls_oracle_and_custom_scenario_can_omit_it():
    class SampleOnly(Scenario):
        n_classes = 2

        def model_ids(self):
            return ["provider"]

        def _generate(self, n, rng):
            p = rng.random((n, 1))
            return np.stack((1 - p, p), axis=-1), rng.integers(2, size=n), {}

    class UnavailableOracle(SampleOnly):
        def bayes_oracle(self, preds, latent):
            raise AssertionError("transfer unexpectedly evaluated the oracle")

    ds = SampleOnly().build(n_per_split=8, seed=22)
    assert ds.fit_oracle is None and ds.test_oracle is None
    x, y = development(UnavailableOracle, 22, 8, q=0.15)
    assert x.shape == (8, 1, 2) and y.shape == (8,)
    assert len(_conditions(UnavailableOracle, 22, 8, canonical_shift={})) == 10
    # The old research helper retains its oracle-enabled behavior.
    with pytest.raises(AssertionError, match="evaluated the oracle"):
        scenario_block(UnavailableOracle(), 22, 1, 8)

import numpy as np
import pytest

from fusionbench.scenarios.checkpoint_bank import (
    generate_checkpoint_bank,
    generate_scene_pool,
    scene_sample_indices,
    select_checkpoints,
    signal_provider_index,
)


@pytest.mark.parametrize("law", ["equal", "signal"])
def test_pool_reuses_legacy_draws_and_extends_prefixes(law):
    seed = 7
    historical = generate_checkpoint_bank(seed, law=law)
    small = generate_scene_pool(seed, law=law, n_fit_scenes=3, n_audit_scenes=5)
    more_fit = generate_scene_pool(seed, law=law, n_fit_scenes=9, n_audit_scenes=5)
    more_audit = generate_scene_pool(seed, law=law, n_fit_scenes=3, n_audit_scenes=17)
    assert list(small) == ["A", "F", "I"]
    for name, block in small.items():
        for attr in ("P", "y", "posterior", "scene"):
            expected = getattr(block, attr)
            for pool in (more_fit, more_audit):
                np.testing.assert_array_equal(
                    getattr(pool[name], attr)[: len(expected)], expected
                )
            if name == "A" or (name == "I" and attr != "scene"):
                np.testing.assert_array_equal(
                    getattr(historical[name], attr)[: len(expected)], expected
                )
        assert block.P.shape == (len(block.scene), 8, 8)
    for name, ids in (
        ("A", np.arange(24)),
        ("F", [24, 26, 28]),
        ("I", [25, 27, 29, 31, 33]),
    ):
        np.testing.assert_array_equal(small[name].scene, np.repeat(ids, 8))
    # F has its own channel namespace for scene signal, row signal and label uniforms.
    s = np.random.default_rng([seed, 4, 0]).standard_normal((3, 1))
    e = np.random.default_rng([seed, 4, 1]).standard_normal((3, 8))
    q = 1 / (1 + np.exp(-1.8 * (0.9 * s + 0.8 * e - 1.1)))
    np.testing.assert_array_equal(small["F"].posterior, q.ravel())
    u = np.random.default_rng([seed, 4, 2]).random((3, 8))
    np.testing.assert_array_equal(small["F"].y, (u < q).ravel())


def test_signal_pairs_all_selected_columns_except_the_planted_provider():
    seed = 23
    equal = generate_scene_pool(seed, n_fit_scenes=7, n_audit_scenes=11)
    signal = generate_scene_pool(seed, law="signal", n_fit_scenes=7, n_audit_scenes=11)
    planted = signal_provider_index(seed)
    others = np.arange(8) != planted
    slots_equal = select_checkpoints(equal["A"], 8)
    slots_signal = select_checkpoints(signal["A"], 8)
    np.testing.assert_array_equal(slots_equal[others], slots_signal[others])
    for name, ref in equal.items():
        block = signal[name]
        for attr in ("y", "scene", "posterior"):
            np.testing.assert_array_equal(getattr(block, attr), getattr(ref, attr))
        np.testing.assert_array_equal(block.P[:, others], ref.P[:, others])
        np.testing.assert_array_equal(
            block.P[:, planted], np.repeat(block.posterior[:, None], 8, axis=1)
        )


def test_scene_indices_allow_nested_counts_and_pools_larger_than_default():
    pool = generate_scene_pool(4, n_fit_scenes=np.int64(1537), n_audit_scenes=1)
    full = scene_sample_indices(1537, 8)
    np.testing.assert_array_equal(full, np.arange(len(pool["F"].scene)))
    small = scene_sample_indices(3, 2)
    np.testing.assert_array_equal(small, [0, 1, 8, 9, 16, 17])
    assert small.dtype == np.int64 and np.isin(small, scene_sample_indices(5, 3)).all()
    np.testing.assert_array_equal(pool["F"].scene[small], [24, 24, 26, 26, 28, 28])


@pytest.mark.parametrize(
    "key,value",
    [
        ("n_fit_scenes", 0),
        ("n_audit_scenes", -1),
        ("n_audit_scenes", 2.0),
        ("n_fit_scenes", True),
    ],
)
def test_invalid_pool_size(key, value):
    with pytest.raises(ValueError, match=key):
        generate_scene_pool(5, **{key: value})


@pytest.mark.parametrize(
    "scenes,rows", [(0, 1), (1, 0), (1, 9), (False, 1), (1, np.bool_(True)), (1.5, 1)]
)
def test_invalid_scene_indices(scenes, rows):
    with pytest.raises(ValueError):
        scene_sample_indices(scenes, rows)

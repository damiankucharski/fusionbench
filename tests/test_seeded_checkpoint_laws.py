import numpy as np
import pytest

from fusionbench.scenarios.checkpoint_bank import (
    Block,
    generate_checkpoint_bank,
    select_checkpoints,
    signal_provider_index,
)


@pytest.mark.parametrize("seed", [0, 19, 107])
def test_paired_signal_changes_only_its_provider(seed):
    equal = generate_checkpoint_bank(seed)
    signal = generate_checkpoint_bank(seed, law="signal")
    expected_id = int(np.random.default_rng([seed, 0, 1]).integers(8))
    assert signal_provider_index(seed) == expected_id
    others = np.arange(8) != expected_id
    for name, baseline in equal.items():
        changed = signal[name]
        np.testing.assert_array_equal(changed.y, baseline.y)
        np.testing.assert_array_equal(changed.scene, baseline.scene)
        np.testing.assert_array_equal(changed.posterior, baseline.posterior)
        np.testing.assert_array_equal(changed.P[:, others], baseline.P[:, others])
        np.testing.assert_array_equal(
            changed.P[:, expected_id], np.repeat(changed.posterior[:, None], 8, axis=1)
        )
        assert not np.array_equal(changed.P[:, expected_id], baseline.P[:, expected_id])
        for k in (1, 8):
            chosen = select_checkpoints(changed, k)
            assert chosen[expected_id] == 0
            np.testing.assert_array_equal(
                chosen[others], select_checkpoints(baseline, k)[others]
            )


def test_latent_posterior_and_expected_full_brier():
    seed = 42
    equal = generate_checkpoint_bank(seed, law="equal")
    default = generate_checkpoint_bank(seed)
    for block_id, (name, groups) in enumerate((("A", 24), ("B", 24), ("I", 256)), 1):
        block = equal[name]
        np.testing.assert_array_equal(default[name].P, block.P)
        np.testing.assert_array_equal(default[name].y, block.y)
        np.testing.assert_array_equal(default[name].scene, block.scene)
        s = np.random.default_rng([seed, block_id, 0]).standard_normal((groups, 1))
        e = np.random.default_rng([seed, block_id, 1]).standard_normal((groups, 8))
        q = 1 / (1 + np.exp(-1.8 * (0.9 * s + 0.8 * e - 1.1)))
        label_u = np.random.default_rng([seed, block_id, 2]).random((groups, 8))
        np.testing.assert_array_equal(block.posterior, q.ravel())
        np.testing.assert_array_equal(block.y, (label_u < q).ravel())
        # Score both possible labels with their actual probabilities.
        p = block.P[:, 0, 0]
        q = block.posterior
        risk = 2 * (q * (1 - p) ** 2 + (1 - q) * p**2)
        np.testing.assert_allclose(
            risk - 2 * q * (1 - q), 2 * (p - q) ** 2, rtol=1e-12, atol=1e-15
        )


def test_oracle_field_optional_and_unused_by_checkpoint_selection():
    b = generate_checkpoint_bank(2, law="signal")["A"]
    plain = Block(b.P, b.y, b.scene)
    assert plain.posterior is None
    np.testing.assert_array_equal(
        select_checkpoints(plain, 8), select_checkpoints(b, 8)
    )
    np.testing.assert_array_equal(
        select_checkpoints(Block(b.P, b.y, b.scene, 1 - b.posterior), 8),
        select_checkpoints(b, 8),
    )


def test_unknown_checkpoint_law_is_rejected():
    with pytest.raises(ValueError, match="law"):
        generate_checkpoint_bank(3, law="wrong")


@pytest.mark.parametrize("seed", [-2, False, 2.5])
def test_signal_provider_rejects_invalid_seeds(seed):
    with pytest.raises(ValueError, match="seed"):
        signal_provider_index(seed)

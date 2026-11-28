import numpy as np
import pytest

from fusionbench.scenarios.rare_shortcut_noise import RareShortcutNoise
from fusionbench.scenarios.source_environments import (
    source_environment_bank,
    source_environment_preset,
)


def test_matched_draws_change_only_mechanism_and_order_is_shared_with_targets():
    h, v, c, m = [
        source_environment_preset(name, seed=23)
        for name in ("homogeneous", "varied", "case_mix", "mixed")
    ]
    for bank in (h, v, c, m):
        assert bank.x.shape == (512, 4, 2)
        np.testing.assert_array_equal(np.bincount(bank.env), [256, 256])
        np.testing.assert_array_equal(bank.y, h.y)
        np.testing.assert_array_equal(bank.env, h.env)
        assert bank.provider_order == h.provider_order
        np.testing.assert_allclose(bank.x.max(axis=2), 0.8)
        np.testing.assert_array_equal(
            bank.context, bank.x[:, bank.provider_order.index(2), 1] > 0.5
        )
    np.testing.assert_array_equal(h.context, v.context)
    np.testing.assert_array_equal(c.context, m.context)
    assert np.any(c.context != h.context)
    for left, right in [(h, v), (c, m)]:
        stable = np.arange(4) != h.shortcut_index
        np.testing.assert_array_equal(left.x[:, stable], right.x[:, stable])
        assert np.any(left.x[:, h.shortcut_index] != right.x[:, h.shortcut_index])
    for left, right in [(h, c), (v, m)]:
        stable = [h.provider_order.index(0), h.shortcut_index]
        np.testing.assert_array_equal(left.x[:, stable], right.x[:, stable])
    other = source_environment_preset("mixed", seed=26)
    assert m.provider_order != other.provider_order
    assert not np.array_equal(other.y, m.y)


def test_source_subdivision_changes_only_the_available_environment_labels():
    two = source_environment_preset("mixed", seed=39)
    four = source_environment_bank(
        region=(0.15, 0.15, 0.85, 0.85), shortcut=(0, 0, 1, 1), n_rows=512, seed=39
    )
    for field in ("x", "y", "context"):
        np.testing.assert_array_equal(getattr(two, field), getattr(four, field))
    np.testing.assert_array_equal(np.bincount(four.env), [128] * 4)
    np.testing.assert_array_equal(two.env, four.env // 2)
    for field in ("x", "mass", "q", "context"):
        np.testing.assert_array_equal(
            getattr(two.target(0.8, 0.5), field), getattr(four.target(0.8, 0.5), field)
        )


@pytest.mark.parametrize("raw", [True, False])
@pytest.mark.parametrize("pi, activity", [(0, 0), (1, 1), (0.65, 0.3)])
def test_target_matches_public_exact_law_in_anonymous_provider_order(raw, pi, activity):
    bank = source_environment_preset("mixed", seed=52, original_scale=raw)
    target = bank.target(pi, activity)
    reference, mass, q = RareShortcutNoise(
        region_test=pi, shortcut_test=activity, provider_noise_test=0
    ).support("test")
    if raw:
        np.testing.assert_array_equal(target.x, reference[:, bank.provider_order])
    np.testing.assert_array_equal(
        target.x > 0.5, reference[:, bank.provider_order] > 0.5
    )
    np.testing.assert_array_equal(target.mass, mass)
    np.testing.assert_array_equal(target.q, q)
    assert mass @ target.context.astype(float) == pytest.approx(pi)
    assert mass @ q == pytest.approx(0.5)
    expected_correct = 0.5 + 0.4 * activity
    shortcut_votes = target.x[:, bank.shortcut_index, 1] > 0.5
    assert mass @ np.where(shortcut_votes, q, 1 - q) == pytest.approx(expected_correct)
    s_votes = target.x[:, bank.specialist_index, 1] > 0.5
    assert mass @ np.where(s_votes, q, 1 - q) == pytest.approx(0.5 + 0.45 * pi)


def test_context_corruption_and_confidence_controls_leave_vote_signs_and_targets_paired():
    clean = source_environment_preset("mixed", seed=56)
    noisy = source_environment_preset("mixed", seed=56, context_noise=0.25)
    raw = source_environment_preset("mixed", seed=56, original_scale=True)
    np.testing.assert_array_equal(clean.env, noisy.env)
    np.testing.assert_array_equal(clean.y, noisy.y)
    np.testing.assert_array_equal(clean.x, noisy.x)
    flips = np.random.default_rng([56, 3]).random(512) < 0.25
    assert 0 < np.count_nonzero(flips) < 512
    np.testing.assert_array_equal(noisy.context, clean.context ^ flips)
    np.testing.assert_array_equal(clean.context, raw.context)
    np.testing.assert_array_equal(clean.x > 0.5, raw.x > 0.5)
    assert np.any(clean.x != raw.x)
    for name in ("x", "mass", "q", "context"):
        np.testing.assert_array_equal(
            getattr(clean.target(0.5, 0), name), getattr(noisy.target(0.5, 0), name)
        )


def test_balanced_and_custom_allocations_and_independent_source_environments():
    opts = {"region": (0.5,) * 3, "shortcut": (0.5,) * 3, "seed": 5}
    odd = source_environment_bank(n_rows=29, **opts)
    fixed = source_environment_bank(n_rows=(10, 10, 9), **opts)
    np.testing.assert_array_equal(np.bincount(odd.env), [10, 10, 9])
    np.testing.assert_array_equal(odd.x, fixed.x)
    np.testing.assert_array_equal(odd.y, fixed.y)
    custom = source_environment_bank(n_rows=(1, 2, 17), **opts)
    np.testing.assert_array_equal(np.bincount(custom.env), [1, 2, 17])
    for bad in (True, 2, [1, 0, 2], [1], [1, False, 2]):
        with pytest.raises(ValueError, match="row"):
            source_environment_bank(n_rows=bad, **opts)


def test_each_sampled_source_matches_its_own_joint_pattern_and_label_law():
    region = (0.1, 0.9)
    shortcut = (0, 1)
    b = source_environment_bank(region=region, shortcut=shortcut, seed=11, n_rows=65536)
    for i, (pi, share) in enumerate(zip(region, shortcut, strict=True)):
        exact = b.target(pi, share)
        # Match each anonymous forecast to its exact support row by the signs
        # seen in the forecast. This check uses no structural provider indices.
        key = [8, 4, 2, 1]
        lookup = np.argsort((exact.x[:, :, 1] > 0.5) @ key)
        rows = b.env == i
        patterns = lookup[(b.x[rows, :, 1] > 0.5) @ key]
        freq = np.bincount(2 * patterns + b.y[rows], minlength=32).reshape(16, 2)
        joint = np.column_stack((exact.mass * (1 - exact.q), exact.mass * exact.q))
        np.testing.assert_allclose(freq / rows.sum(), joint, atol=0.007, rtol=0)

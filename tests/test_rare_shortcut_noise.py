from itertools import product
from math import prod

import numpy as np
import pytest

from fusionbench.scenarios.rare_shortcut_noise import RareShortcutNoise
from fusionbench.scenarios.rare_specialist import RareSpecialist


def brute_joint(prevalence, share, report_noise, label_noise):
    """Expand every latent event directly, including the fair bypass sign."""
    mass = np.zeros((16, 2))
    for events in product((0, 1), repeat=10):
        y, region, g_ok, s_ok, gate, h_ok, fair_h, flip_g, flip_s, flip_y = events
        rates = (
            0.5,
            prevalence,
            0.75,
            0.95 if region else 0.5,
            share,
            0.9,
            0.5,
            report_noise,
            report_noise,
            label_noise,
        )
        weight = prod(
            rate if bit else 1 - rate for bit, rate in zip(events, rates, strict=True)
        )
        g, s = y ^ (1 - g_ok) ^ flip_g, y ^ (1 - s_ok) ^ flip_s
        h = y ^ (1 - h_ok) if gate else fair_h
        mass[8 * g + 4 * s + 2 * region + h, y ^ flip_y] += weight
    return mass


@pytest.mark.parametrize("split", ["fit", "test"])
def test_joint_mass_and_posterior_match_latent_enumeration(split):
    params = {
        "region_fit": 0.13,
        "region_test": 0.64,
        "shortcut_fit": 0.24,
        "shortcut_test": 0.73,
        "provider_noise_fit": 0.2,
        "provider_noise_test": 0.82,
        "label_noise_fit": 0.8,
        "label_noise_test": 0.125,
    }
    scenario = RareShortcutNoise(**params)
    settings = [
        params[f"{k}_{split}"]
        for k in ("region", "shortcut", "provider_noise", "label_noise")
    ]
    p, m, q = scenario.support(split)
    votes = (p[:, :, 1] > 0.5).astype(int)
    np.testing.assert_array_equal(votes, list(product((0, 1), repeat=4)))
    np.testing.assert_allclose(p.sum(axis=2), 1)
    assert scenario.model_ids() == [
        "generalist",
        "specialist",
        "routing_cue",
        "shortcut",
    ]
    ref = brute_joint(*settings)
    np.testing.assert_allclose(m, ref.sum(axis=1), atol=1e-15)
    np.testing.assert_allclose(m * q, ref[:, 1], atol=1e-15)
    np.testing.assert_allclose(m @ q, 0.5, atol=1e-15)
    # Giving the oracle an unrelated hidden region cannot change its output.
    np.testing.assert_allclose(
        scenario.bayes_oracle(p, {"region": 1 - votes[:, 2], "split": split})[:, 1], q
    )
    conditional_m, conditional_mq = [], []
    for r in (0, 1):
        rp, rm, rq = scenario.support(split, region=r)
        np.testing.assert_array_equal(rp, p)
        assert rm.sum() == pytest.approx(1)
        assert (rq[rm == 0] == 0.5).all() and (rm[votes[:, 2] != r] == 0).all()
        np.testing.assert_allclose(rq[rm > 0], q[rm > 0])
        conditional_m.append(rm)
        conditional_mq.append(rm * rq)
    pi = settings[0]
    np.testing.assert_allclose(m, (1 - pi) * conditional_m[0] + pi * conditional_m[1])
    np.testing.assert_allclose(
        m * q, (1 - pi) * conditional_mq[0] + pi * conditional_mq[1]
    )


def test_paired_draws_report_and_label_flips_and_regional_reuse():
    def sample(**opts):
        rng = np.random.default_rng(31)
        p, y, latent = RareShortcutNoise(**opts).sample_split(1400, rng, "fit")
        return p, y, latent, rng.random()

    pp, yy, group, u = sample()
    old = RareSpecialist(
        region_prevalence=0.05,
        generalist_accuracy=0.75,
        cue_accuracy=1,
        specialist_gain=0.45,
    )
    op, oy, ol = old.sample(1400, np.random.default_rng(31))
    np.testing.assert_array_equal(pp[:, :3], op)
    np.testing.assert_array_equal(yy, oy)
    np.testing.assert_array_equal(group["region"], ol["region"])
    assert group["split"] == "fit"
    changed, labels, latent, other_u = sample(provider_noise_fit=1, label_noise_fit=1)
    np.testing.assert_array_equal(changed[:, :2], pp[:, :2, ::-1])
    np.testing.assert_array_equal(changed[:, 2:], pp[:, 2:])
    np.testing.assert_array_equal(labels, 1 - yy)
    np.testing.assert_array_equal(latent["region"], group["region"])
    assert u == other_u
    inactive, _, _, v = sample(shortcut_fit=0)
    np.testing.assert_array_equal(inactive[:, :3], pp[:, :3])
    assert (inactive[:, 3] != pp[:, 3]).any() and v == u
    shifted, _, _, _ = sample(region_fit=0.95)
    np.testing.assert_array_equal(shifted[:, [0, 3]], pp[:, [0, 3]])


def test_zero_mass_and_noise_endpoints():
    for pi in (0, 1):
        s = RareShortcutNoise(region_fit=pi, provider_noise_fit=0.5, shortcut_fit=0)
        p, m, q = s.support("fit")
        assert (m[(p[:, 2, 1] > 0.5) != pi] == 0).all()
        np.testing.assert_allclose(q, 0.5)
        assert s.support("fit", region=1 - pi)[1].sum() == pytest.approx(1)
    base = RareShortcutNoise().support("test")
    for eta in (0.5, 1):
        p, m, q = RareShortcutNoise(label_noise_test=eta).support("test")
        np.testing.assert_array_equal(p, base[0])
        np.testing.assert_array_equal(m, base[1])
        np.testing.assert_allclose(q, eta + (1 - 2 * eta) * base[2])


def test_sampled_joint_matches_exact_observed_label_law():
    s = RareShortcutNoise(
        region_test=0.33,
        shortcut_test=0.37,
        provider_noise_test=0.19,
        label_noise_test=0.24,
    )
    p, y, latent = s.sample_split(100_000, np.random.default_rng(234), "test")
    pat = (p[:, :, 1] > 0.5) @ np.array([8, 4, 2, 1])
    np.testing.assert_array_equal(latent["region"], p[:, 2, 1] > 0.5)
    support, m, q = s.support("test")
    np.testing.assert_array_equal(support[pat], p)
    emp = np.bincount(2 * pat + y, minlength=32).reshape(16, 2) / len(y)
    np.testing.assert_allclose(
        emp, np.column_stack((m * (1 - q), m * q)), atol=0.003, rtol=0
    )
    np.testing.assert_array_equal(s.bayes_oracle(p, latent)[:, 1], q[pat])


@pytest.mark.parametrize(
    "key",
    [
        "region_fit",
        "region_test",
        "shortcut_fit",
        "shortcut_test",
        "provider_noise_fit",
        "provider_noise_test",
        "label_noise_fit",
        "label_noise_test",
    ],
)
def test_validate_constructor_probability(key):
    for value in (-0.1, 1.1, np.nan, True, ".5"):
        with pytest.raises(ValueError, match=key):
            RareShortcutNoise(**{key: value})


def test_validate_split_region_and_build():
    s = RareShortcutNoise()
    with pytest.raises(ValueError, match="split"):
        s.sample_split(2, np.random.default_rng(2), "unknown")
    with pytest.raises(ValueError, match="region"):
        s.support("fit", region=1.5)
    data = s.build(n_per_split=12, seed=9)
    assert data.fit_oracle.shape == data.test_oracle.shape == (12, 2)
    assert data.meta["evaluation_groups"]["test"]["region"] == list(
        data.test_preds[:, 2, 1] > 0.5
    )

"""Test the observable posterior and pairing of mechanism-specific stress settings."""

from itertools import pairwise, product

import numpy as np
import pytest
from scipy.special import expit, logit
from scipy.stats import norm

from fusionbench import generate_dataset, get_scenario, load_dataset, save_dataset

GATES = ("hardveto", "hardrescue", "weakest_link", "two_pathways", "compound")


def truth(state, name):
    if name == "hardveto":
        a, b = state
        return a and b, a and not b
    if name == "hardrescue":
        return any(state), sum(state) == 1
    if name == "weakest_link":
        return all(state), sum(state) == len(state) - 1
    if name == "two_pathways":
        a, b, c, d = state
        return (a and b) or (c and d), c and d and not (a and b)
    a, b, c = state
    return (a or b) and c, not a and b and c


@pytest.mark.parametrize("name", GATES)
def test_reweighted_oracle_enumerates_observable_evidence(name):
    prevalence, fraction = 0.27, 0.78
    model = get_scenario(
        name, rep=1, positive_prevalence=prevalence, decisive_fraction=fraction
    )
    x, _, latent = model.sample(93, np.random.default_rng(415))
    states = list(product((0, 1), repeat=x.shape[1]))
    labels, decisive = np.asarray([truth(bits, name) for bits in states], dtype=bool).T
    eligible_positive = name not in ("hardveto", "weakest_link")
    eligible = labels == eligible_positive
    mass = prevalence if eligible_positive else 1 - prevalence
    prior = np.empty(len(states))
    for mask, prob in (
        (~eligible, 1 - mass),
        (decisive, mass * fraction),
        (eligible & ~decisive, mass * (1 - fraction)),
    ):
        prior[mask] = prob / mask.sum()
    # Under the balanced reference prior, each detector reports its own
    # bit posterior. The common per-observation normalizer cancels.
    likelihood = np.prod(
        np.where(np.asarray(states)[:, None, :], x[None, :, :, 1], x[None, :, :, 0]),
        axis=2,
    )
    weighted = prior[:, None] * likelihood
    reference = weighted[labels].sum(axis=0) / weighted.sum(axis=0)
    oracle = model.bayes_oracle(x, {"decisive": ~latent["decisive"]})
    np.testing.assert_allclose(oracle[:, 1], reference, atol=4e-13, rtol=1e-12)


@pytest.mark.parametrize("name", GATES)
def test_controlled_decisive_rate_at_fixed_labels(name):
    def sample(q):
        return get_scenario(name, positive_prevalence=0.37, decisive_fraction=q).sample(
            2400, np.random.default_rng(221)
        )

    _, y, low = sample(0.2)
    _, y_high, high = sample(0.8)
    np.testing.assert_array_equal(y, y_high)
    eligible = y == int(name not in ("hardveto", "weakest_link"))
    for masks, rate in ((low, 0.2), (high, 0.8)):
        assert not masks["decisive"][~eligible].any()
        assert masks["decisive"][eligible].mean() == pytest.approx(rate, abs=0.05)
    assert np.all(~low["decisive"] | high["decisive"])
    assert y.mean() == pytest.approx(0.37, abs=0.03)
    for prevalence in (0, 1):
        model = get_scenario(name, positive_prevalence=prevalence)
        x, y, latent = model.sample(25, np.random.default_rng(909))
        assert (y == prevalence).all()
        assert (model.bayes_oracle(x, latent)[:, 1] == prevalence).all()


def test_matched_prevalence_depth_preserves_labels_and_decisive_mask(tmp_path):
    draws = []
    for m in (2, 4, 40):
        model = get_scenario(
            "weakest_link",
            m=m,
            rep=1,
            positive_prevalence=np.float64(0.0625),
            decisive_fraction=0.4,
        )
        x, y, latent = model.sample(151, np.random.default_rng(89))
        assert x.shape[1] == m and np.isfinite(model.bayes_oracle(x, {})).all()
        draws.append((y, latent["decisive"]))
    for current in draws[1:]:
        np.testing.assert_array_equal(current, draws[0])
    bank = generate_dataset(
        "weakest_link",
        n_per_split=15,
        seed=1,
        m=np.int64(4),
        positive_prevalence=np.float32(0.25),
    )
    restored = load_dataset(save_dataset(bank, tmp_path))
    assert restored.meta["parameters"]["positive_prevalence"] == 0.25
    for split in ("fit", "test"):
        group = restored.meta["evaluation_groups"][split]["decisive"]
        assert len(group) == 15 and all(isinstance(case, bool) for case in group)
    assert "evaluation_groups" not in get_scenario("weakest_link").build(15, 1).meta


def test_gate_censor_mass_uses_gaussian_tail_probability():
    model = get_scenario(
        "hardveto", rep=1, positive_prevalence=1 / 4, decisive_fraction=1 / 3
    )
    # These masses recover independent fair bits. The second detector is
    # uninformative and the first observes a lower-tail clipping event.
    x = np.array([[[1 - 1e-6, 1e-6], [0.5, 0.5]]])
    s = norm.ppf(model.acc_detect)
    z = logit(1e-6) / (2 * s)
    ratio = norm.logcdf(z - s) - norm.logcdf(z + s)
    np.testing.assert_allclose(
        model.bayes_oracle(x, {})[0, 1], expit(ratio) * 0.5, rtol=1e-12, atol=0
    )


def test_temperature_and_parity_keep_other_evidence_paired():
    base = get_scenario("miscalibration")
    x, y, _ = base.sample(477, np.random.default_rng(3))
    for severity in (0.0, 0.4, 1.0):
        model = get_scenario("miscalibration", temperature_range=severity)
        z, labels, latent = model.sample(477, np.random.default_rng(3))
        np.testing.assert_array_equal(labels, y)
        np.testing.assert_array_equal(x.argmax(axis=2), z.argmax(axis=2))
        np.testing.assert_allclose(
            latent["ks"], np.linspace(0.5, 2.5, z.shape[1]) ** severity
        )
        if severity != 1:
            raw = logit(z[:, :, 1]) / latent["ks"]
            np.testing.assert_allclose(
                model.bayes_oracle(z, latent)[:, 1], expit(raw.sum(axis=1)), atol=2e-13
            )
    baseline = get_scenario("parity")
    blank = get_scenario("parity", second_accuracy=0.5)
    base_x, base_y, _ = baseline.sample(101, np.random.default_rng(4))
    z, y, _ = blank.sample(101, np.random.default_rng(4))
    np.testing.assert_array_equal(base_y, y)
    np.testing.assert_array_equal(base_x[:, :2], z[:, :2])
    assert np.all(z[:, 2:] == 0.5) and np.all(blank.bayes_oracle(z, {}) == 0.5)


@pytest.mark.parametrize("name", ("native_switch", "native_close_switch"))
def test_switch_mass_preserves_thresholds_uniforms_and_distractors(name):
    reference = get_scenario(name, role_seed=77)
    originals = reference.draw(2103, np.random.default_rng(11))
    roles = reference.roles
    other = np.delete(np.arange(6), roles)
    flags = []
    for mass in (0, 0.3, 0.8, 1):
        model = get_scenario(name, material_fraction=mass, role_seed=77)
        rng = np.random.default_rng(11)
        draw = model.draw(2103, rng)
        mate = np.random.default_rng(11)
        reference.draw(2103, mate)
        assert rng.random() == mate.random()
        np.testing.assert_array_equal(draw.uniform, originals.uniform)
        np.testing.assert_array_equal(draw.preds[:, other], originals.preds[:, other])
        a, b = draw.preds[:, roles[0], 1], draw.preds[:, roles[1], 1]
        material = abs(a - b) >= 0.25
        flags.append(material)
        x, y, latent = model.sample(2103, np.random.default_rng(11))
        np.testing.assert_array_equal(latent["material"], material)
        p = model.bayes_oracle(x, {})[:, 1]
        np.testing.assert_array_equal(y, draw.uniform < p)
        assert material.mean() == pytest.approx(mass, abs=0.035)
    assert all(np.all(~prev | nxt) for prev, nxt in pairwise(flags))


def test_similar_classes_private_arm_changes_only_wrong_destinations():
    base = get_scenario("similar_classes_38")
    first, y, _ = base.sample(400, np.random.default_rng(11))
    last, targets, _ = get_scenario("similar_classes_38", deviation_mix=1).sample(
        400, np.random.default_rng(11)
    )
    part = (y == 3) | (y == 8)
    np.testing.assert_array_equal(targets, y)
    np.testing.assert_array_equal(first[~part], last[~part])
    np.testing.assert_array_equal(
        first[np.arange(len(y)), :, y], last[np.arange(len(y)), :, y]
    )
    np.testing.assert_array_equal(np.sort(first, axis=2), np.sort(last, axis=2))
    for provider in range(3):
        x = last[part, provider].copy()
        x[np.arange(len(x)), y[part]] = -np.inf
        np.testing.assert_array_equal(x[:, provider], x.max(axis=1))
    assert np.any(first[part] != last[part])


@pytest.mark.parametrize(
    "name,kwargs",
    [
        ("hardveto", {"positive_prevalence": -0.1}),
        ("hardrescue", {"decisive_fraction": 1.5}),
        ("weakest_link", {"decisive_fraction": 0.5, "m": 1}),
        ("miscalibration", {"temperature_range": np.inf}),
        ("parity", {"second_accuracy": 0.3}),
        ("native_switch", {"material_fraction": -0.1}),
    ],
)
def test_invalid_scores(name, kwargs):
    with pytest.raises(ValueError):
        get_scenario(name, **kwargs)

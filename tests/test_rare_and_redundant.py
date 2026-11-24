"""Observed-information oracles and paired severity/positive-control arms."""

from itertools import product

import numpy as np
import pytest

from fusionbench.scenarios.rare_specialist import RareSpecialist
from fusionbench.scenarios.redundant_pool import RedundantPool


def test_specialist_oracle_enumerates_hidden_region_and_all_observable_votes():
    scenario = RareSpecialist(
        region_prevalence=0.17,
        cue_accuracy=0.92,
        generalist_accuracy=0.71,
        specialist_gain=0.43,
    )
    cases = list(product((0, 1), repeat=3))
    encoded = np.where(cases, 0.75, 0.25)
    x = np.stack([1 - encoded, encoded], axis=-1)
    reference = []
    for g, s, c in cases:
        joint = np.zeros(2)
        for y, r in product((0, 1), repeat=2):
            gen = 0.71 if g == y else 0.29
            alpha = 0.93 if r else 0.5
            spec = alpha if s == y else 1 - alpha
            gate = 0.92 if c == r else 0.08
            joint[y] += 0.5 * (0.17 if r else 0.83) * gen * spec * gate
        reference.append(joint / joint.sum())
    np.testing.assert_allclose(
        scenario.bayes_oracle(x, {}), reference, rtol=0, atol=1e-15
    )
    np.testing.assert_array_equal(
        scenario.bayes_oracle(x, {"region": np.zeros(8)}),
        scenario.bayes_oracle(x, {"region": np.ones(8)}),
    )


def test_specialist_arms_keep_labels_and_cases_paired_and_have_exact_nulls():
    n = 2000

    def sample(**kw):
        s = RareSpecialist(**kw)
        x, y, h = s.sample(n, np.random.default_rng(21840))
        return x, y, h["region"], s.bayes_oracle(x, {})

    x, y, h, _ = sample()
    x0, y0, h0, oracle0 = sample(specialist_gain=0)
    np.testing.assert_array_equal(y, y0)
    np.testing.assert_array_equal(h, h0)
    np.testing.assert_array_equal(x[~h], x0[~h])
    np.testing.assert_allclose(oracle0, x0[:, 0], atol=1e-15, rtol=0)
    diffuse, yd, hd, _ = sample(cue_accuracy=0.5)
    np.testing.assert_array_equal(x[:, :2], diffuse[:, :2])
    np.testing.assert_array_equal(yd, y)
    np.testing.assert_array_equal(hd, h)
    diffuse_alt = diffuse.copy()
    diffuse_alt[:, 2] = diffuse_alt[:, 2, ::-1]
    blind = RareSpecialist(cue_accuracy=0.5)
    np.testing.assert_array_equal(
        blind.bayes_oracle(diffuse, {}), blind.bayes_oracle(diffuse_alt, {})
    )
    xr, yr, hr, _ = sample(region_prevalence=0.025)
    np.testing.assert_array_equal(yr, y)
    np.testing.assert_array_equal(xr[:, 0], x[:, 0])
    assert (hr <= h).all()


@pytest.mark.parametrize("prevalence", [0, 1])
@pytest.mark.parametrize("generalist", [0.5, 1])
def test_specialist_boundary_votes_support(prevalence, generalist):
    scenario = RareSpecialist(
        region_prevalence=prevalence,
        generalist_accuracy=generalist,
        specialist_gain=0.5,
        cue_accuracy=1,
    )
    x, y, latent = scenario.sample(100, np.random.default_rng(42))
    oracle = scenario.bayes_oracle(x, {})
    assert np.all(latent["region"] == bool(prevalence))
    if generalist == 1 or prevalence == 1:
        np.testing.assert_array_equal(oracle[:, 1], y)
    else:
        assert np.all(oracle == 0.5)


def test_redundant_pool_preserves_both_splits_and_nested_candidates():
    small = RedundantPool(n_candidates=3, role_seed=333)
    large = RedundantPool(n_candidates=23, role_seed=333)
    positive = RedundantPool(n_candidates=23, role_seed=333, positive_control=True)
    relabeled = RedundantPool(n_candidates=3, role_seed=444)
    datasets = [
        s.build(n_per_split=150, seed=812) for s in (small, large, positive, relabeled)
    ]
    assert (
        small.model_ids()
        == relabeled.model_ids()
        == ["model_0", "model_1", "model_2", "model_3"]
    )
    assert (small.anchor_index, *small.candidate_indices) != (
        relabeled.anchor_index,
        *relabeled.candidate_indices,
    )
    for split in ("fit", "test"):
        xs = [getattr(d, split + "_preds") for d in datasets]
        for d in datasets:
            np.testing.assert_array_equal(
                getattr(d, split + "_y"), getattr(datasets[0], split + "_y")
            )
        np.testing.assert_array_equal(
            xs[0][:, small.anchor_index], xs[1][:, large.anchor_index]
        )
        np.testing.assert_array_equal(
            xs[0][:, small.candidate_indices], xs[1][:, large.candidate_indices[:3]]
        )
        np.testing.assert_array_equal(
            xs[0][:, (small.anchor_index, *small.candidate_indices)],
            xs[3][:, (relabeled.anchor_index, *relabeled.candidate_indices)],
        )
        bit_column = positive.candidate_indices[0]
        others = [i for i in range(xs[1].shape[1]) if i != bit_column]
        np.testing.assert_array_equal(xs[1][:, others], xs[2][:, others])
        assert np.any(xs[1][:, bit_column] != xs[2][:, bit_column])
        np.testing.assert_array_equal(
            getattr(datasets[1], split + "_oracle"), xs[1][:, large.anchor_index]
        )
        np.testing.assert_array_equal(
            getattr(datasets[2], split + "_oracle"), xs[2][:, bit_column]
        )


def test_redundant_positive_uses_bayes_update_and_null_votes_only_obey_anchor():
    null = RedundantPool(n_candidates=1, candidate_accuracy=0.83)
    informed = RedundantPool(
        n_candidates=1, candidate_accuracy=0.83, positive_control=True
    )
    a, y, _ = null.sample(60000, np.random.default_rng(921))
    b, yp, _ = informed.sample(len(y), np.random.default_rng(921))
    np.testing.assert_array_equal(y, yp)
    p = a[:, null.anchor_index, 1]
    for scenario, x in ((null, a), (informed, b)):
        candidate = x[:, scenario.candidate_indices[0], 1]
        plus = candidate > p
        likelihood_ratio = np.where(plus, 0.83 / 0.17, 0.17 / 0.83)
        odds = likelihood_ratio * p / (1 - p)
        np.testing.assert_allclose(candidate, odds / (1 + odds), atol=1e-14, rtol=0)
        q = p * 0.83 + (1 - p) * 0.17
        # Null B is independent of Y given p. The positive candidate instead observes a noisy Y.
        expected = np.where(y, 0.83, 0.17) if scenario.positive_control else q
        for label in (0, 1):
            assert abs(np.mean((plus - expected)[y == label])) < 0.012
        orac = scenario.bayes_oracle(x, {"y": 1 - y})[:, 1]
        if scenario.positive_control:
            assert np.mean((orac - y) ** 2) < np.mean((p - y) ** 2) - 0.03
        else:
            np.testing.assert_array_equal(orac, p)


def test_empty_pool_and_uninformative_or_perfect_positive_controls():
    only = RedundantPool(n_candidates=0).build(n_per_split=80, seed=415)
    assert only.fit_preds.shape == (80, 1, 2)
    np.testing.assert_array_equal(only.test_preds[:, 0], only.test_oracle)
    for acc in (0.5, 1.0):
        s = RedundantPool(4, True, acc)
        x, y, _ = s.sample(100, np.random.default_rng(19))
        expected = x[:, s.anchor_index, 1] if acc == 0.5 else y
        np.testing.assert_allclose(
            s.bayes_oracle(x, {})[:, 1], expected, atol=1e-15, rtol=0
        )


@pytest.mark.parametrize(
    ("cls", "knobs"),
    [
        (RareSpecialist, {"region_prevalence": -0.1}),
        (RareSpecialist, {"specialist_gain": 0.51}),
        (RareSpecialist, {"generalist_accuracy": True}),
        (RareSpecialist, {"cue_accuracy": "bad"}),
        (RedundantPool, {"positive_control": True, "n_candidates": 0}),
        (RedundantPool, {"n_candidates": False}),
        (RedundantPool, {"role_seed": -1}),
        (RedundantPool, {"positive_control": "true"}),
        (RedundantPool, {"candidate_accuracy": float("nan")}),
    ],
)
def test_invalid_knobs(cls, knobs):
    with pytest.raises((TypeError, ValueError)):
        cls(**knobs)


def test_numpy_knobs_and_private_group_metadata_survive_normal_storage(tmp_path):
    from fusionbench import load_dataset, save_dataset

    scenarios = (
        RareSpecialist(
            region_prevalence=np.float32(0.25), generalist_accuracy=np.float64(0.75)
        ),
        RedundantPool(
            n_candidates=np.int64(2),
            positive_control=np.bool_(True),
            role_seed=np.uint32(1),
        ),
    )
    for scenario in scenarios:
        ds = scenario.build(n_per_split=64, seed=91)
        restored = load_dataset(save_dataset(ds, tmp_path))
        assert ds.meta == restored.meta
        if isinstance(scenario, RareSpecialist):
            rng = np.random.default_rng(91)
            for split in ("fit", "test"):
                _, _, latent = scenario.sample(64, rng)
                np.testing.assert_array_equal(
                    restored.meta["evaluation_groups"][split]["region"],
                    latent["region"],
                )
        else:
            assert "evaluation_groups" not in restored.meta

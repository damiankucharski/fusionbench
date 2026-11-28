from itertools import permutations, product

import numpy as np
import pytest

from fusionbench import generate_dataset, load_dataset, save_dataset
from fusionbench.eval.transfer_grid import STREAMS, iter_test_conditions
from fusionbench.scenarios.environment_shortcut import EnvironmentShortcut
from fusionbench.scenarios.selective_fallback import SelectiveFallback
from fusionbench.transfer import _scenario_spec


def test_shift_pairing_and_default_domain_posteriors():
    scenario = EnvironmentShortcut()
    source, target = [
        scenario.sample_split(4000, np.random.default_rng(17), s)
        for s in ("fit", "test")
    ]
    np.testing.assert_array_equal(source[1], target[1])
    np.testing.assert_array_equal(source[0][:, 0], target[0][:, 0])
    for (p, y, info), proportions in zip(
        (source, target), (scenario.fit_mix, scenario.test_mix), strict=True
    ):
        ratios = [
            np.mean((p[:, 1].argmax(1) == y)[info["environment"] == e])
            for e in range(3)
        ]
        np.testing.assert_allclose(ratios, [0.90, 0.5, 0.10], atol=0.08)
        np.testing.assert_allclose(
            np.bincount(info["environment"], minlength=3) / len(y),
            proportions,
            atol=0.03,
        )
        # Enumerate latent environment and binary signal error probabilities.
        posterior = np.zeros((len(p), 2))
        for e, q in enumerate((0.90, 0.5, 0.10)):
            for label in range(2):
                posterior[:, label] += (
                    0.5
                    * proportions[e]
                    * np.where(p[:, 0].argmax(1) == label, 0.75, 0.25)
                    * np.where(p[:, 1].argmax(1) == label, q, 1 - q)
                )
        np.testing.assert_allclose(
            scenario.bayes_oracle(p, {"split": info["split"]}),
            posterior / posterior.sum(1, keepdims=True),
        )
    null = EnvironmentShortcut(stable_only=True)
    p, _, info = null.sample_split(4000, np.random.default_rng(17), "test")
    np.testing.assert_array_equal(p[:, 0], target[0][:, 0])
    np.testing.assert_array_equal(p[:, 1], 0.5)
    np.testing.assert_array_equal(null.bayes_oracle(p, info), p[:, 0])


def test_split_aware_transfer_and_group_metadata(tmp_path):
    opts = {"fit_mix": (1, 0, 0), "test_mix": (0, 0, 1)}
    ds = generate_dataset("environment_shortcut", n_per_split=17, seed=33, **opts)
    assert ds.meta["evaluation_groups"] == {
        "fit": {"environment": [0] * 17},
        "test": {"environment": [2] * 17},
    }
    back = load_dataset(save_dataset(ds, tmp_path))
    assert back.meta["evaluation_groups"] == ds.meta["evaluation_groups"]
    assert (
        _scenario_spec("environment_shortcut", {"seed_slot": 91, "knobs": opts})[
            "canonical_shift"
        ]
        == {}
    )
    for overrides in (None, {"test_mix": [0, 1, 0]}):
        rows = list(
            iter_test_conditions(
                "environment_shortcut",
                12,
                256,
                scenario_kwargs=opts,
                modes=["canonical_shift"],
                severities=[1],
                canonical_shift=overrides,
            )
        )
        params = {**opts, **(overrides or {})}
        scenario = EnvironmentShortcut(**params)
        for mode, _, p, y in rows:
            split, stream = (
                ("fit", "clean_test")
                if mode == "clean"
                else ("test", "canonical_shift")
            )
            exp, labels, _ = scenario.sample_split(
                len(y), np.random.default_rng([12, STREAMS[stream]]), split
            )
            np.testing.assert_array_equal(p, exp)
            np.testing.assert_array_equal(y, labels)


def test_stationary_for_equal_mixtures_and_chance_or_perfect_signals():
    for accuracy in (0.5, 1.0):
        sc = EnvironmentShortcut(
            fit_mix=(0, 0, 1),
            test_mix=(0, 0, 1),
            stable_accuracy=accuracy,
            shortcut_accuracy=accuracy,
        )
        fit, test = [
            sc.sample_split(27, np.random.default_rng(51), s) for s in ("fit", "test")
        ]
        np.testing.assert_array_equal(fit[0], test[0])
        np.testing.assert_array_equal(fit[1], test[1])
        for p, y, meta in (fit, test):
            expected = np.eye(2)[y] if accuracy == 1 else np.full((len(y), 2), 0.5)
            np.testing.assert_array_equal(sc.bayes_oracle(p, meta), expected)


def test_matched_quota_and_ordinary_reports():
    seed, n = 8802, 101
    draws = {}
    for mode in ("random", "difficulty"):
        for rate, flag in ((0.0, False), (0.25, False), (0.25, True)):
            sc = SelectiveFallback(fallback_rate=rate, mechanism=mode, expose_flag=flag)
            draws[mode, rate, flag] = sc.sample(n, np.random.default_rng(seed))
        baseline, labels, _ = draws[mode, 0.0, False]
        p, y, latent = draws[mode, 0.25, False]
        flags, _, _ = draws[mode, 0.25, True]
        mask = latent["fallback"]
        assert mask.sum() == 25 and latent["hard"].sum() == 30
        np.testing.assert_array_equal(y, labels)
        np.testing.assert_array_equal(p[:, 0], baseline[:, 0])
        np.testing.assert_array_equal(p[~mask], baseline[~mask])
        np.testing.assert_array_equal(p[mask, 1], 0.5)
        np.testing.assert_array_equal(flags[:, :2], p)
        np.testing.assert_array_equal(flags[:, 2, 1], ~mask)
        assert np.any((baseline[:, 1, 1] == 0.5) & ~mask)
        if mode == "difficulty":
            assert np.all(latent["hard"][mask])
    np.testing.assert_array_equal(
        draws["random", 0.0, False][0], draws["difficulty", 0.0, False][0]
    )


@pytest.mark.parametrize(
    "mode,flag,rate",
    list(product(("difficulty", "random"), (False, True), (0.0, 0.25, 0.75, 1.0))),
)
def test_observed_fallback_posterior_by_exhaustive_quota_and_signal_enumeration(
    mode, flag, rate
):
    n = 4
    scenario = SelectiveFallback(
        mechanism=mode, expose_flag=flag, hard_fraction=0.5, fallback_rate=rate
    )
    # For a representative row, enumerate both independent permutations rather
    # than using the generator's closed-form stratum weights.
    cells = np.zeros((2, 2))
    orders = list(permutations(range(n)))
    for d, r in product(orders, repeat=2):
        order = d if mode == "difficulty" else r
        cells[int(0 in d[:2]), int(0 in order[: int(n * rate)])] += 1 / len(orders) ** 2
    observed = {}
    for h, f, y, a, b, neutral in product(range(2), repeat=6):
        ref_accuracy = (0.82, 0.60)[h]
        expert_accuracy = (0.94, 0.72)[h]
        mass = 0.5 * cells[h, f] * (ref_accuracy if a == y else 1 - ref_accuracy)
        mass *= (expert_accuracy if b == y else 1 - expert_accuracy) * (
            0.2 if neutral else 0.8
        )
        if mass:
            key = ((0.25, 0.75)[a], 0.5 if neutral or f else (0.15, 0.85)[b]) + (
                (1 - f,) if flag else ()
            )
            observed.setdefault(key, np.zeros(2))[y] += mass
    p = np.array(list(observed))
    expected = np.array(list(observed.values()))
    # Supply only the design size, even in the availability-hidden arm.
    actual = scenario.bayes_oracle(np.stack((1 - p, p), axis=-1), {"bank_size": n})
    np.testing.assert_allclose(
        actual, expected / expected.sum(axis=1, keepdims=True), atol=1e-13, rtol=0
    )


def test_invalid_parameters():
    for args in (
        {"fit_mix": [1, 1]},
        {"test_mix": [0.5] * 3},
        {"stable_accuracy": np.nan},
        {"stable_only": 1},
    ):
        with pytest.raises((ValueError, TypeError)):
            EnvironmentShortcut(**args)
    for args in (
        {"fallback_rate": np.nan},
        {"hard_fraction": -1},
        {"mechanism": "other"},
        {"expose_flag": "yes"},
    ):
        with pytest.raises((ValueError, TypeError)):
            SelectiveFallback(**args)

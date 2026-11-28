from itertools import product

import numpy as np
import pytest

from fusionbench import generate_dataset, get_scenario, load_reference_dataset
from fusionbench.eval.transfer_grid import STREAMS, iter_test_conditions
from fusionbench.transfer import load_preset


def test_factorial_changes_only_the_intended_reports_on_matched_rows():
    banks = {
        (c, r): generate_dataset(
            "cloned_shortcut",
            clone_probability=c,
            test_reversal_probability=r,
            n_per_split=257,
            seed=113,
        )
        for c, r in product((0, 1), repeat=2)
    }
    clean = banks[0, 0]
    for (c, r), bank in banks.items():
        assert bank.model_ids == [f"stable_{i}" for i in range(2)] + [
            f"shortcut_{i}" for i in range(6)
        ]
        for split in ("fit", "test"):
            x = getattr(bank, split + "_preds")
            source = getattr(clean, split + "_preds")
            np.testing.assert_array_equal(
                getattr(bank, split + "_y"), getattr(clean, split + "_y")
            )
            np.testing.assert_array_equal(x[:, :2], source[:, :2])
            expected = source[:, 2:]
            if c:
                expected = np.repeat(expected[:, :1], 6, axis=1)
            if r and split == "test":
                expected = expected[:, :, ::-1]
            np.testing.assert_array_equal(x[:, 2:], expected)
            flags = bank.meta["evaluation_groups"][split]
            assert flags["shortcut_cloned"] == [bool(c)] * 257
            assert flags["shortcut_reversed"] == [bool(r and split == "test")] * 257
    assert not np.array_equal(clean.fit_y, clean.test_y)
    assert not np.array_equal(clean.fit_preds[:, 0, :], clean.fit_preds[:, 1, :])


@pytest.mark.parametrize("cloning,reversal", [(0.0, 0.0), (1.0, 1.0), (0.37, 0.64)])
@pytest.mark.parametrize("split", ["fit", "test"])
def test_observed_posterior_matches_independent_exhaustive_enumeration(
    cloning, reversal, split
):
    # Enumerate pristine votes and both hidden binary events, accumulate their
    # *joint* probabilities at the resulting observed ordered vote vector.
    scenario = get_scenario(
        "cloned_shortcut", clone_probability=cloning, test_reversal_probability=reversal
    )
    votes = np.array(list(product((0, 1), repeat=8)))
    basis = 2 ** np.arange(7, -1, -1)
    accuracy = np.array([0.75] * 2 + [0.8] * 6)
    joint = np.zeros((256, 2))
    for label in (0, 1):
        p_votes = 0.5 * np.prod(
            np.where(votes == label, accuracy, 1 - accuracy), axis=1
        )
        for clone, flip in product((False, True), repeat=2):
            x = votes.copy()
            if clone:
                x[:, 2:] = x[:, 2:3]
            if flip:
                x[:, 2:] = 1 - x[:, 2:]
            r = reversal if split == "test" else 0.0
            probability = (cloning if clone else 1 - cloning) * (r if flip else 1 - r)
            np.add.at(joint[:, label], x @ basis, p_votes * probability)
    assert joint.sum() == pytest.approx(1)
    possible = joint.sum(axis=1) > 0
    positive = np.where(votes[possible], accuracy, 1 - accuracy)
    observed = np.stack((1 - positive, positive), axis=-1)
    expected = joint[possible] / joint[possible].sum(1, keepdims=True)
    np.testing.assert_allclose(
        scenario.bayes_oracle(observed, {"split": split}), expected, atol=1e-14
    )


def test_transfer_endpoint_seed_slots_and_offline_bank():
    previous = load_preset("standard-v3")
    current = load_preset("standard-v4")
    assert list(current["scenarios"]) == [*previous["scenarios"], "cloned_shortcut"]
    assert current["scenarios"]["cloned_shortcut"] == {"seed_slot": 24}
    assert {
        **previous,
        "scenarios": {**previous["scenarios"], "cloned_shortcut": {"seed_slot": 24}},
    } == current
    assert list(load_preset("standard")["scenarios"])[:25] == list(current["scenarios"])
    sc = get_scenario("cloned_shortcut")
    conditions = list(
        iter_test_conditions(
            "cloned_shortcut", 31, 32, modes=["canonical_shift"], severities=[1]
        )
    )
    for (mode, severity, p, y), split, stream in zip(
        conditions, ("fit", "test"), ("clean_test", "canonical_shift"), strict=True
    ):
        assert severity == (mode == "canonical_shift")
        expected, labels, _ = sc.sample_split(
            32, np.random.default_rng([31, STREAMS[stream]]), split
        )
        np.testing.assert_array_equal(p, expected)
        np.testing.assert_array_equal(y, labels)
    frozen = load_reference_dataset("cloned_shortcut")
    live = generate_dataset("cloned_shortcut", n_per_split=128, seed=1729)
    assert frozen.meta["reference_bank"] == "reference-v4"
    assert frozen.meta["evaluation_groups"] == live.meta["evaluation_groups"]
    for suffix in ("preds", "oracle", "y"):
        for split in ("fit", "test"):
            a = getattr(frozen, split + "_" + suffix)
            np.testing.assert_array_equal(
                a, getattr(live, split + "_" + suffix).astype(a.dtype)
            )


@pytest.mark.parametrize("v", [-0.1, 1.01, np.nan, np.inf, True, [1], "0"])
def test_invalid_knobs_and_domain(v):
    from fusionbench.scenarios.cloned_shortcut import ClonedShortcut

    for name in ("clone_probability", "test_reversal_probability"):
        with pytest.raises(ValueError, match=name):
            ClonedShortcut(**{name: v})
    with pytest.raises(ValueError, match="split"):
        ClonedShortcut().sample_split(5, np.random.default_rng(13), "validation")

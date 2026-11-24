"""Discovery-only MIN proposals and independent GP/partition seeds."""

import json
from collections import Counter

import numpy as np
import pytest

from fusionbench.methods.native_pattern import NativePatternOKAPI


def _fit(x, y, **options):
    return NativePatternOKAPI(**options).fit(
        x, y, task="binary" if x.shape[2] == 2 else "multiclass", n_classes=x.shape[2]
    )


def test_appended_minimum_has_original_provider_ids_and_replays_on_new_rows():
    pytest.importorskip("okapi.pair_basis")
    from fusionbench.eval.suite_point_gp import predict_tree

    rng = np.random.default_rng(32)
    p = rng.uniform(0.02, 0.98, (240, 9))
    x = np.stack((1 - p, p), axis=2)
    y = ((p[:, 2] > 0.5) ^ (p[:, 6] > 0.5)).astype(int)
    baseline = _fit(x, y, seed=8)
    with_pairs = _fit(x, y, seed=8, pair_basis=True)
    assert "pair_basis" not in baseline.counts_
    assert baseline.counts_["gp_path_digest"] == with_pairs.counts_["gp_path_digest"]
    z = rng.dirichlet([1.0, 1.0], size=(43, 9))
    for name in ("flat", "gp"):
        np.testing.assert_array_equal(
            predict_tree(baseline.trees_[name], z),
            predict_tree(with_pairs.trees_[name], z),
        )
    counts = with_pairs.counts_
    pairs = counts["pair_basis"]
    assert pairs["programs"] == 36 and pairs["evaluations"] == 0
    assert 0 < pairs["unique_novel_features"] <= pairs["programs"]
    assert 0 < pairs["selected_pair_count"] == len(pairs["selected_pairs"]) <= 3
    assert pairs["additional_discovery_calls"] == pairs["screen_calls"]
    positions = [row["archive_index"] for row in counts["screen"]["selected"]]
    for index, expected_pair in zip(
        pairs["selected_archive_indices"], pairs["selected_pairs"], strict=True
    ):
        assert pairs["pairs"][index - pairs["gp_archive_size"]] == expected_pair
        for family in (with_pairs.discovery_trees_, with_pairs.trees_):
            head = family["augmented"].root.children[-1]
            basis = head.children[x.shape[1] - 1 + positions.index(index)]
            assert basis.id == f"m{expected_pair[0]}.npy"
            (op,) = basis.children
            assert type(op).__name__ == "MinNode"
            (other,) = op.children
            assert other.id == f"m{expected_pair[1]}.npy"
    np.testing.assert_array_equal(
        with_pairs.predict(z), predict_tree(with_pairs.trees_[with_pairs.chosen_], z)
    )
    json.dumps(counts)


def test_duplicate_pair_programs_never_cost_a_screening_call():
    pytest.importorskip("okapi.pair_basis")
    rng = np.random.default_rng(52)
    p = np.tile(rng.uniform(0.02, 0.98, (90, 1)), (1, 5))
    x = np.stack((1 - p, p), axis=2)
    model = _fit(x, rng.integers(2, size=len(x)), seed=10, pair_basis=True)
    pairs = model.counts_["pair_basis"]
    assert pairs["programs"] == 10
    assert pairs["selected_pairs"] == []
    assert (
        pairs["unique_novel_features"]
        == pairs["screen_calls"]
        == pairs["additional_discovery_calls"]
        == 0
    )


def test_multiclass_ranking_and_accounting_use_only_discovery_rows():
    pytest.importorskip("okapi.pair_basis")
    rng = np.random.default_rng(936)
    x = rng.dirichlet(np.ones(3), size=(140, 14))
    y = x[:, 8].argmax(axis=1)
    calls = Counter()

    def accuracy(p, labels):
        assert p.shape == (len(labels), 3)
        calls[len(labels)] += 1
        return np.mean(p.argmax(axis=1) == labels)

    seed = 19
    model = _fit(x, y, seed=seed, pair_basis=True, fitness=accuracy)
    counts = model.counts_
    pairs = counts["pair_basis"]
    discovery = np.random.default_rng([seed, 51]).permutation(len(x))[:98]
    scores = (x[discovery].argmax(axis=2) == y[discovery, None]).mean(axis=0)
    expected = sorted(sorted(range(14), key=scores.__getitem__, reverse=True)[:12])
    assert pairs["provider_indices"] == expected
    assert pairs["programs"] == 66 and counts["score_calls"]["pair_ranking"] == 14
    assert pairs["additional_discovery_calls"] == pairs["screen_calls"] + 14
    assert (
        counts["total_score_calls"]
        == sum(calls.values())
        == sum(counts["score_calls"].values())
    )
    assert calls[42] == counts["score_calls"]["gp_validation"] + 3
    assert (
        calls[140]
        == counts["score_calls"]["flat_refit"]
        + counts["score_calls"]["augmented_refit"]
    )


def test_split_seed_changes_only_the_partition_rng():
    pytest.importorskip("okapi.pattern_logit")
    rng = np.random.default_rng(7713)
    x = rng.dirichlet([1.0, 1.0], size=(68, 3))
    y = rng.integers(2, size=len(x))
    default = _fit(x, y, seed=45)
    explicit = _fit(x, y, seed=45, split_seed=45)
    alternate = _fit(x, y, seed=46, split_seed=45)
    for model in (explicit, alternate):
        np.testing.assert_array_equal(
            default.validation_indices_, model.validation_indices_
        )
        for key in ("discovery_sha256", "validation_sha256"):
            assert model.counts_[key] == default.counts_[key]
    assert default.counts_["gp_path_digest"] == explicit.counts_["gp_path_digest"]
    np.testing.assert_array_equal(default.predict(x), explicit.predict(x))
    assert alternate.counts_["gp_path_digest"] != default.counts_["gp_path_digest"]

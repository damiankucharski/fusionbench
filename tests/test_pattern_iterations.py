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


def test_positive_l2_regularizes_four_heads_and_keeps_selection_on_raw_fitness(
    monkeypatch,
):
    import inspect

    native = pytest.importorskip("okapi.pattern_logit")
    if "l2_strength" not in inspect.signature(native.fit_pattern_tree).parameters:
        pytest.skip("Native OKAPI lacks optional head regularization")
    from fusionbench.eval.suite_point_gp import predict_tree

    rng = np.random.default_rng(713)
    x = rng.dirichlet([1.0, 1.0], size=(160, 5))
    y = (x[:, 0, 1] + 0.6 * x[:, 1, 1] > 0.8).astype(int)
    seed = 15
    old_fit = native.fit_pattern_tree

    def raw(p, y):
        return 1.0 - np.square(p - np.eye(2)[y]).sum(axis=1).mean() / 2.0

    # Older OKAPI wheels omit the optional keyword from their signature.
    def legacy(roots, labels, *, objective, multiclass):
        return old_fit(roots, labels, objective=objective, multiclass=multiclass)

    monkeypatch.setattr(native, "fit_pattern_tree", legacy)
    baseline = _fit(x, y, seed=seed, l2_strength=0.0)
    assert (
        "head_fit_scores" not in baseline.counts_
        and "l2_strength" not in baseline.counts_
    )
    monkeypatch.setattr(native, "fit_pattern_tree", old_fit)
    strength = 2.0
    model = _fit(x, y, seed=seed, l2_strength=strength)
    assert model.counts_["l2_strength"] == strength
    assert model.counts_["gp_path_digest"] == baseline.counts_["gp_path_digest"]
    d, v = np.split(np.random.default_rng([seed, 51]).permutation(len(x)), [112])
    records = model.counts_["head_fit_scores"]
    assert len(records) == 4
    for stage, trees, rows in (
        ("discovery", model.discovery_trees_, d),
        ("refit", model.trees_, np.arange(len(x))),
    ):
        for family in ("flat", "augmented"):
            record = records[f"{family}_{stage}"]
            actual = raw(predict_tree(trees[family], x[rows]), y[rows])
            assert record["raw_score"] == pytest.approx(actual, abs=1e-13)
            assert record["penalty"] > 0
            assert record["penalized_score"] == pytest.approx(
                actual - record["penalty"], abs=1e-13
            )
    valid = {
        family: raw(predict_tree(tree, x[v]), y[v])
        for family, tree in model.discovery_trees_.items()
    }
    assert model.counts_["validation_scores"] == pytest.approx(valid, abs=1e-13)
    assert model.chosen_ == max(valid, key=valid.get)
    assert model.counts_["screen"]["initial_score"] == pytest.approx(
        records["flat_discovery"]["raw_score"], abs=1e-13
    )
    json.dumps(model.counts_)


@pytest.mark.parametrize("invalid", [-1.0, float("nan"), float("inf"), True, [1.0]])
def test_invalid_l2_strength_fails_before_discovery(invalid):
    with pytest.raises(ValueError, match="finite nonnegative"):
        NativePatternOKAPI(l2_strength=invalid)

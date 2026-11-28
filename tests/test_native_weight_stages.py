import numpy as np
import pytest

from fusionbench.eval.seed_geometry_standard import fit_seed_geometry
from fusionbench.eval.suite_point_gp import predict_tree
from fusionbench.methods.native_pattern import (
    NativePatternOKAPI,
    RegularizedNativePatternOKAPI,
)

pytest.importorskip("okapi.pattern_logit")


def example():
    rng = np.random.default_rng(7721)
    x = rng.uniform(0.02, 0.98, (88, 4))
    y = ((x[:, 0] + x[:, 2]) > 1).astype(int)
    swap = x[:, 1] > 0.8
    y[swap] = 1 - y[swap]
    weight = np.where(swap, 9.0, 1.0)
    return x, y, weight


def skill(p, y, w):
    loss = np.square(p - np.eye(p.shape[1])[y]).sum(axis=1)
    return 1 - 0.5 * np.average(loss, weights=w)


@pytest.fixture(scope="module")
def fits():
    x, y, w = example()
    inputs = {
        "base": {},
        "uniform": {"sample_weight": np.ones(len(y))},
        **{
            stage: {"sample_weight": w, "weight_stages": (stage,)}
            for stage in ("discovery", "validation", "refit")
        },
        "all": {"sample_weight": w},
    }
    return {
        name: RegularizedNativePatternOKAPI(
            seed=3, split_seed=0, gp_validation_top_k=8
        ).fit(x, y, task="binary", n_classes=2, **kwargs)
        for name, kwargs in inputs.items()
    }


def test_uniform_weights_are_bit_exact_through_search_selection_and_refit(fits):
    base, uniform = fits["base"], fits["uniform"]
    x, _, _ = example()
    assert base.chosen_ == uniform.chosen_
    np.testing.assert_array_equal(base.predict(x), uniform.predict(x))
    for name in base.trees_:
        assert (
            base.trees_[name].structural_key() == uniform.trees_[name].structural_key()
        )
    # Only wall-clock duration varies.
    assert {k: v for k, v in base.counts_.items() if k != "seconds"} == {
        k: v for k, v in uniform.counts_.items() if k != "seconds"
    }


@pytest.mark.parametrize("name", ["discovery", "validation", "refit", "all"])
def test_stage_scores_match_independent_full_brier_and_preserve_counters(fits, name):
    m, base = fits[name], fits["base"]
    x, y, w = example()
    d, v = m.discovery_indices_, m.validation_indices_
    stages = m.counts_["weight_stages"]
    assert stages == (["discovery", "validation", "refit"] if name == "all" else [name])
    assert m.counts_["gp_selection_scale"] == "unit_brier_3_decimals"
    assert m.counts_["score_calls"]["gp_discovery"] == 600
    assert m.counts_["total_score_calls"] == sum(m.counts_["score_calls"].values())
    np.testing.assert_array_equal(base.discovery_indices_, d)
    np.testing.assert_array_equal(base.validation_indices_, v)
    for tree, val in m.counts_["validation_scores"].items():
        probs = predict_tree(m.discovery_trees_[tree], x[v])
        assert val == pytest.approx(
            skill(probs, y[v], w[v] if "validation" in stages else None), abs=2e-12
        )
    for tree in ("flat", "augmented"):
        heads = m.counts_["head_fit_scores"]
        assert heads[tree + "_discovery"]["raw_score"] == pytest.approx(
            skill(
                predict_tree(m.discovery_trees_[tree], x[d]),
                y[d],
                w[d] if "discovery" in stages else None,
            ),
            abs=2e-12,
        )
        assert heads[tree + "_refit"]["raw_score"] == pytest.approx(
            skill(predict_tree(m.trees_[tree], x), y, w if "refit" in stages else None),
            abs=2e-12,
        )
    if "discovery" not in stages:
        assert m.counts_["gp_path_digest"] == base.counts_["gp_path_digest"]
        assert m.counts_["screen"] == base.counts_["screen"]
    if stages == ["refit"]:
        assert m.chosen_ == base.chosen_
        assert m.counts_["validation_scores"] == base.counts_["validation_scores"]


def test_gp_shortlist_and_champion_follow_stage_brier_with_unrounded_priorities():
    x, y, w = example()
    d, v = np.arange(60), np.arange(60, len(y))
    kept = []

    def capture(tree, p):
        kept.append((tree.copy(), np.average((p[:, 0] - y[d]) ** 2, weights=w[d])))

    result = fit_seed_geometry(
        x[d],
        y[d],
        x[v],
        y[v],
        seed=3,
        shape="stock",
        budgets=(75,),
        gp_validation_top_k=4,
        on_candidate=capture,
        discovery_weight=w[d],
        validation_weight=w[v],
    )[75]
    eligible = sorted(range(len(kept)), key=lambda i: (kept[i][1], i))[:4]
    val_losses = {
        i: np.average((predict_tree(kept[i][0], x[v])[:, 1] - y[v]) ** 2, weights=w[v])
        for i in eligible
    }
    best = min(eligible, key=lambda i: (val_losses[i], kept[i][1], i))
    assert result.val_tree.structural_key() == kept[best][0].structural_key()
    assert result.fit_brier == pytest.approx(min(loss for _, loss in kept))
    assert result.validation_brier == pytest.approx(val_losses[best])


@pytest.mark.parametrize(
    "w", [[1.0], np.zeros(88), np.full(88, np.nan), np.full(88, -1), np.ones((88, 1))]
)
def test_invalid_weights(w):
    x, y, _ = example()
    with pytest.raises(ValueError, match="weight"):
        NativePatternOKAPI().fit(x, y, task="binary", n_classes=2, sample_weight=w)


def test_ambiguous_fitness_and_invalid_stages_rejected():
    x, y, w = example()
    model = NativePatternOKAPI(fitness=lambda x, y: 0.0)
    with pytest.raises(ValueError, match="built-in"):
        model.fit(x, y, task="binary", n_classes=2, sample_weight=w)
    for subset in ("validation", ("refit", "refit"), ("unknown",)):
        with pytest.raises((TypeError, ValueError), match="weight_stages"):
            NativePatternOKAPI().fit(
                x, y, task="binary", n_classes=2, weight_stages=subset
            )

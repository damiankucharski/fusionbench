import numpy as np
import pytest

from fusionbench.methods.native_pattern import (
    NativePatternOKAPI,
    RegularizedNativePatternOKAPI,
)


@pytest.mark.parametrize("invalid", [0, -1, True, np.bool_(False), 1.25, "8"])
def test_gp_validation_budget_requires_a_positive_integer(invalid):
    for cls in (NativePatternOKAPI, RegularizedNativePatternOKAPI):
        with pytest.raises(ValueError, match="gp_validation_top_k"):
            cls(gp_validation_top_k=invalid)


def test_custom_discovery_score_and_ties_on_each_requested_prefix():
    pytest.importorskip("okapi.pattern_logit")
    from fusionbench.eval.seed_geometry_standard import fit_seed_geometry
    from fusionbench.eval.suite_point_gp import predict_tree

    rng = np.random.default_rng(503)
    fit = rng.uniform(0.01, 0.99, (48, 4))
    val = rng.uniform(0.01, 0.99, (27, 4))
    y = rng.integers(2, size=len(fit))
    vy = rng.integers(2, size=len(val))
    kept = []

    def score(p, labels):
        return float(np.sum((p[:, 0] > 0.7) == labels))

    def capture(tree, pred):
        kept.append((tree.copy(), score(pred, y)))

    result = fit_seed_geometry(
        fit,
        y,
        val,
        vy,
        seed=3,
        shape="stock",
        budgets=(50, 80, 120),
        fitness=score,
        on_candidate=capture,
        gp_validation_top_k=np.int64(3),
    )
    for snapshot in result.values():
        pool = kept[: snapshot.n_trees]
        eligible = sorted(range(len(pool)), key=lambda i: -pool[i][1])[:3]
        best = max(
            eligible,
            key=lambda i: (
                score(predict_tree(pool[i][0], val)[:, 1:], vy),
                pool[i][1],
                -i,
            ),
        )
        assert snapshot.val_tree.structural_key() == pool[best][0].structural_key()
        assert snapshot.validation_calls <= snapshot.n_trees

import numpy as np
import pytest
from sparse_gp import fit_sparse, fit_sparse_pool

from fusionbench.methods.okapi import _task_postprocessing


@pytest.mark.parametrize("classes", [2, 3])
def test_sparse_and_random_seeds_replay_with_matched_full_training_reference(classes):
    rng = np.random.default_rng(771)
    x = rng.dirichlet(np.ones(classes), size=(29, 8))
    y = rng.integers(classes, size=len(x))
    queries = rng.dirichlet(np.ones(classes), size=(19, 8))
    arms = fit_sparse(x, y)
    for name in ("sparse", "random"):
        model, pool = arms[f"{name}_gp"], arms[f"{name}_pool"]
        ids = pool.selected_indices_
        manual_subset = queries[:, ids, :].mean(1)
        assert len(ids) == 3 and ids == model.selected_indices_
        assert set(pool.best_tree.unique_value_node_ids) == {f"m{i}.npy" for i in ids}
        standalone = fit_sparse_pool(x, random_subset=name == "random")
        assert ids == standalone.selected_indices_
        np.testing.assert_allclose(pool.predict(queries), manual_subset, rtol=0, atol=1e-14)
        np.testing.assert_array_equal(pool.predict(queries), standalone.predict(queries))
        objective, = model.okapi.objective_functions
        full_mean = x.mean(1)
        subset = x[:, ids, :].mean(1)
        distance = np.square(full_mean - subset).sum(1).mean()
        brier = np.square(subset - np.eye(classes)[y]).sum(1).mean()
        strength = np.sqrt(8 / len(y))
        with _task_postprocessing(model.task):
            score = objective(model.seed_tree_.predict(), model.okapi.gt_tensor)
        assert score == pytest.approx(1 - (brier + strength * distance) / (2 * (1 + strength)))
    expected = np.sort(np.random.default_rng(421).choice(8, 3, replace=False))
    np.testing.assert_array_equal(arms["random_gp"].selected_indices_, expected)


def test_source_column_order_with_more_than_ten_ids():
    x = np.broadcast_to([.25, .75], (16, 12, 2))
    assert fit_sparse_pool(x).selected_indices_ == (0, 1, 2, 3)

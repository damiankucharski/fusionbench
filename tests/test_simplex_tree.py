import numpy as np
import pytest

pytest.importorskip("okapi.simplex")
from fusionbench.methods.simplex_tree import SimplexTree


@pytest.mark.parametrize("c", [2, 4])
@pytest.mark.parametrize("regularized", [False, True])
def test_default_brier_replays_native_weighted_pool(c, regularized):
    rng = np.random.default_rng(134)
    x = rng.dirichlet(np.ones(c), size=(24, 4))
    y = x[:, 2].argmax(axis=1)
    query = rng.dirichlet(np.ones(c), size=(13, 4))
    model = SimplexTree(regularized=regularized).fit(
        x, y, task="binary" if c == 2 else "multiclass", n_classes=c
    )
    p = model.predict(query)
    np.testing.assert_allclose(
        p, np.einsum("m,nmc->nc", model.weights_, query), rtol=0, atol=2e-15
    )
    fit = model.predict(x)
    info = model.fit_info_
    assert info["raw_score"] == pytest.approx(
        -np.square(fit - np.eye(c)[y]).sum(axis=1).mean()
    )
    assert info["evaluations"] == 53 and info["lambda"] == pytest.approx(
        np.sqrt(4 / 24) if regularized else 0
    )
    assert {node.id for node in model.tree_.value_nodes} == {
        f"m{i}.npy" for i in info["active_indices"]
    }
    assert np.all(model.weights_ >= 0) and model.weights_.sum() == pytest.approx(1)
    np.testing.assert_allclose(p.sum(1), 1)


def test_custom_objective_shapes_and_exact_zero_probabilities():
    data = np.broadcast_to(np.eye(3), (6, 3, 3))
    calls = []

    def objective(p, y):
        calls.append(p.shape)
        return p[:, 1].mean()

    model = SimplexTree(False, fitness=objective).fit(
        data, np.zeros(6, dtype=int), task="multiclass", n_classes=3
    )
    assert calls == [(6, 3)] * 40
    assert model.weights_.tolist() == [0.0, 1.0, 0.0] and not model.tree_.op_nodes
    np.testing.assert_array_equal(model.predict(data), data[:, 1])
    with pytest.raises(ValueError, match="counts must match"):
        model.predict(data[:, :2])

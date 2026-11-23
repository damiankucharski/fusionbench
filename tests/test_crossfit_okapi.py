import numpy as np
import pytest

from fusionbench.methods.crossfit_okapi import CrossFitOKAPI, brier_weights


def test_simplex_solution_interior_edges_and_collinearity():
    # Three component error vectors have equal squared length and disjoint support.
    labels = np.zeros(3, dtype=int)
    positive = 0.6 * np.eye(3)
    p = np.stack((1 - positive, positive), axis=2)
    np.testing.assert_allclose(brier_weights(p, labels), np.full(3, 1 / 3))
    np.testing.assert_allclose(brier_weights(p[:, :2], labels), [0.5, 0.5])
    p[:, 2] = p[:, 0]
    w = brier_weights(p, labels)
    np.testing.assert_allclose([w[0] + w[2], w[1]], [0.5, 0.5])
    p[:, 0] = [1.0, 0.0]
    np.testing.assert_allclose(brier_weights(p, labels), [1.0, 0.0, 0.0])


def test_model_registration_and_predict_uses_final_components():
    pytest.importorskip("okapi")
    from fusionbench.methods.registry import default_methods

    (model,) = default_methods(["okapi_crossfit_stack"])
    assert isinstance(model, CrossFitOKAPI)
    rng = np.random.default_rng(42)
    x = rng.uniform(0.1, 0.9, size=(72, 4))
    preds = np.stack([1 - x, x], axis=2)
    labels = np.arange(len(preds)) % 2
    assert model.fit(preds, labels, task="binary", n_classes=2) is model
    assert model.oof_.shape == (72, 3, 2)
    assert sum(f["holdout"] for f in model.fold_records_) == 72
    test_x = preds[:4][::-1]
    expected = (
        model.component_predictions(test_x) * model.weights_[None, :, None]
    ).sum(1)
    np.testing.assert_allclose(model.predict(test_x), expected)
    np.testing.assert_allclose(expected.sum(1), 1.0)

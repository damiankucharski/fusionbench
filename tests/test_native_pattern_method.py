"""A native deployment and a discontinuous fitness use neither logistic estimator."""

import numpy as np
import pytest


def test_native_fit_and_public_prediction_with_discrete_score(monkeypatch):
    pytest.importorskip("okapi.pattern_logit")
    from sklearn.linear_model import LogisticRegression

    from fusionbench.eval.suite_point_gp import predict_tree
    from fusionbench.methods.native_pattern import NativePatternOKAPI
    from fusionbench.methods.registry import default_methods

    def forbidden(*args, **kwargs):
        raise AssertionError("A gradient-fitted logistic model entered the challenger")

    monkeypatch.setattr(LogisticRegression, "fit", forbidden)
    rng = np.random.default_rng(407)
    pos = rng.uniform(0.01, 0.99, (120, 4))
    x = np.stack([1 - pos, pos], axis=2)
    y = (
        np.maximum(np.minimum(pos[:, 0], pos[:, 1]), np.minimum(pos[:, 2], pos[:, 3]))
        > 0.5
    ).astype(int)
    calls = []

    def accuracy(p, labels):
        assert p.shape == (len(labels), 2)
        calls.append(len(p))
        return np.count_nonzero(p.argmax(axis=1) == labels) / len(labels)

    assert isinstance(default_methods(["okapi_native_pattern"])[0], NativePatternOKAPI)
    method = NativePatternOKAPI(seed=12, fitness=accuracy)
    method.fit(x[:100], y[:100], task="binary", n_classes=2)
    p = method.predict(x[100:])
    assert len(calls) > 600
    np.testing.assert_array_equal(
        p, predict_tree(method.trees_[method.chosen_], x[100:])
    )

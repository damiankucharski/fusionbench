"""Standalone derivative-free fitting of one native weighted probability tree."""

import numpy as np

from fusionbench.methods.base import Method
from fusionbench.types import _check_probabilities


def _probabilities(preds):
    x = np.asarray(preds)
    if x.ndim != 3 or min(x.shape) < 1 or x.shape[2] < 2:
        raise ValueError("Supply probability arrays shaped (rows, providers, classes)")
    _check_probabilities("Provider predictions", x, x.shape)
    x = x.astype(np.float64, copy=False)
    return x / x.sum(axis=2, keepdims=True)


class SimplexTree(Method):
    """Fit common nonnegative provider weights with optional shrinkage to uniform.

    The default higher-is-better criterion is negative full-class Brier. A custom
    fitness(probabilities, labels) should use the same units if the regularizer
    is to keep its Brier-scale interpretation. Native fitting uses only source rows.
    """

    name = "okapi_simplex_tree"

    def __init__(self, regularized=True, fitness=None):
        if not isinstance(regularized, (bool, np.bool_)):
            raise ValueError("regularized must be boolean")
        if fitness is not None and not callable(fitness):
            raise TypeError("fitness must be callable")
        self.regularized, self.fitness = regularized, fitness

    @classmethod
    def available(cls):
        try:
            from okapi.simplex import fit_simplex_tree
        except ImportError:
            return False
        return callable(fit_simplex_tree)

    def fit(self, fit_preds, fit_y, *, task, n_classes):
        from okapi.globals import set_backend
        from okapi.node import ValueNode
        from okapi.simplex import fit_simplex_tree

        from fusionbench.eval.suite_point_gp import _bind

        x = _probabilities(fit_preds)
        expected_task = "binary" if n_classes == 2 else "multiclass"
        if x.shape[2] != n_classes or task != expected_task:
            raise ValueError(
                "Task and class count must agree with the probability class axis"
            )
        y = np.asarray(fit_y)
        if (
            y.shape != (len(x),)
            or y.dtype.kind not in "biu"
            or not np.all((y >= 0) & (y < n_classes))
        ):
            raise ValueError("Expect one integer class index in range per source row")
        y = y.astype(np.int64, copy=False)
        if self.fitness is None:
            target = np.eye(n_classes)[y]

            def objective(p, labels):
                return -float(np.square(p - target).sum(axis=1).mean())
        else:
            objective = self.fitness
        set_backend("numpy")
        roots = [ValueNode([], values, key) for key, values in _bind(x).items()]
        tree, info = fit_simplex_tree(
            roots,
            y,
            objective=objective,
            multiclass=n_classes > 2,
            regularized=self.regularized,
        )
        super().fit(x, y, task=task, n_classes=n_classes)
        self.tree_, self.fit_info_ = tree, info
        self.weights_ = np.asarray(info["weights"])
        self.n_providers_ = len(self.weights_)
        return self

    def predict(self, preds):
        from okapi.globals import set_backend

        from fusionbench.eval.suite_point_gp import _bind, _task_postprocessing

        if not hasattr(self, "tree_"):
            raise ValueError("Fit the simplex tree before predicting")
        x = _probabilities(preds)
        if x.shape[1:] != (self.n_providers_, self.n_classes):
            raise ValueError("Query provider and class counts must match training")
        set_backend("numpy")
        with _task_postprocessing(self.task):
            result = np.asarray(
                self.tree_.do_pred_on_another_tensors(current_tensors=_bind(x))
            )
        if self.n_classes == 2:
            p = np.clip(result[:, 0], 0.0, 1.0)
            return np.column_stack((1 - p, p))
        return result / result.sum(axis=1, keepdims=True)

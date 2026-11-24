"""Standalone, derivative-free native multiclass probability-affine trees."""

import numpy as np

from fusionbench.methods.base import Method
from fusionbench.methods.native_pattern import (
    _brier_skill,
    _full_probability,
    _head,
    _validate_class_head,
)
from fusionbench.types import _check_probabilities


class _OKAPIClassHead(Method):
    mode: str

    def __init__(self, *, l2_strength=0.25, max_evaluations=2400, fitness=None):
        _validate_class_head(self.mode, max_evaluations)
        strength = np.asarray(l2_strength)
        if strength.ndim != 0 or strength.dtype.kind not in "iuf" or not np.isfinite(strength) or strength < 0:
            raise ValueError("l2_strength must be a finite nonnegative scalar")
        if fitness is not None and not callable(fitness):
            raise TypeError("fitness must be callable")
        self.l2_strength = float(strength)
        self.max_evaluations = max_evaluations
        self.fitness = fitness

    @classmethod
    def available(cls):
        try:
            from okapi.multiclass_affine import fit_multiclass_affine_tree  # noqa: F401
        except ImportError:
            return False
        return True

    @staticmethod
    def _probabilities(preds):
        x = np.asarray(preds)
        if x.ndim != 3 or min(x.shape) < 1 or x.shape[2] < 3:
            raise ValueError("Expected [row, provider, class] probabilities with at least three classes")
        _check_probabilities("Provider probabilities", x, x.shape)
        return x.astype(np.float64, copy=False)

    def fit(self, fit_preds, fit_y, *, task, n_classes):
        from okapi.globals import set_backend

        from fusionbench.eval.suite_point_gp import _bind

        x = self._probabilities(fit_preds)
        if task != "multiclass" or n_classes != x.shape[2]:
            raise ValueError("Expected a multiclass task matching the provider class axis")
        y = np.asarray(fit_y)
        if y.shape != (len(x),) or y.dtype.kind not in "biu" or np.any((y < 0) | (y >= n_classes)):
            raise ValueError("Expected one integer class index in range per row")
        fitness = _brier_skill if self.fitness is None else self.fitness

        def score(p, labels):
            return fitness(_full_probability(p), labels)

        set_backend("numpy")
        tree, info = _head(_bind(x), y.astype(np.int64, copy=False), (), score, True, self.l2_strength, self.mode, self.max_evaluations)
        super().fit(fit_preds, fit_y, task=task, n_classes=n_classes)
        self.tree_ = tree
        self.fit_info_ = info
        self.n_providers_ = x.shape[1]
        self.classes_ = np.arange(n_classes)
        return self

    def predict(self, preds):
        from fusionbench.eval.suite_point_gp import predict_tree

        if not hasattr(self, "tree_"):
            raise ValueError("Fit the native class head before prediction")
        x = self._probabilities(preds)
        if x.shape[1:] != (self.n_providers_, self.n_classes):
            raise ValueError("Query provider and class axes must match training")
        return predict_tree(self.tree_, x)


class OKAPIClassDiagonal(_OKAPIClassHead):
    """Fit one signed coefficient per provider and class, plus a class bias."""

    name = "okapi_class_diagonal"
    mode = "diagonal"


class OKAPIClassFull(_OKAPIClassHead):
    """Fit a full input-to-output class matrix per provider, plus a class bias."""

    name = "okapi_class_full"
    mode = "full"

"""Hold-out Brier shrinkage of a fitted OKAPI tree toward the base-pool mean."""

from __future__ import annotations

import numpy as np

from fusionbench.methods.averaging import SimpleAverage
from fusionbench.methods.base import Method
from fusionbench.methods.okapi import OKAPIMethod


def brier_tree_weight(tree, anchor, labels):
    """Least-squares coefficient of the tree in a convex probability mixture.

    Identical predictors use the anchor. Arrays are normalized [example, class]
    probabilities on examples that took no part in fitting the candidate tree.
    """
    tree, anchor = np.asarray(tree, dtype=float), np.asarray(anchor, dtype=float)
    y = np.asarray(labels).reshape(-1)
    if tree.ndim != 2 or tree.shape != anchor.shape or len(tree) != len(y) or not len(y):
        raise ValueError("expected matching nonempty probability matrices and labels")
    if not np.issubdtype(y.dtype, np.integer) or y.min() < 0 or y.max() >= tree.shape[1]:
        raise ValueError("invalid class labels")
    if not np.isfinite(tree).all() or not np.isfinite(anchor).all():
        raise ValueError("probabilities must be finite")
    difference = tree - anchor
    norm = np.sum(difference ** 2)
    if norm == 0:
        return 0.0
    # Dot with the one-hot target without allocating a new label matrix.
    numerator = difference[np.arange(len(y)), y].sum() - np.sum(anchor * difference)
    return float(np.clip(numerator / norm, 0.0, 1.0))


class ShrunkOKAPI(Method):
    """Search on a random subset; estimate one blend coefficient on the rest."""

    name = "okapi_shrunk"

    def __init__(self, cal_fraction=0.25, split_seed=0, **okapi_options):
        if not 0 < cal_fraction < 1:
            raise ValueError("cal_fraction must be strictly between 0 and 1")
        self.cal_fraction = cal_fraction
        self.split_seed = split_seed
        self.okapi_options = okapi_options

    @classmethod
    def available(cls):
        return OKAPIMethod.available()

    def fit(self, fit_preds, fit_y, *, task, n_classes):
        super().fit(fit_preds, fit_y, task=task, n_classes=n_classes)
        n = len(fit_y)
        if n < 2:
            raise ValueError("at least two examples are required for fit and calibration")
        n_cal = max(1, min(n - 1, int(round(self.cal_fraction * n))))
        order = np.random.default_rng(self.split_seed).permutation(n)
        cal, train = order[:n_cal], order[n_cal:]
        model = OKAPIMethod(**self.okapi_options).fit(
            fit_preds[train], fit_y[train], task=task, n_classes=n_classes)
        weight = brier_tree_weight(model.predict(fit_preds[cal]),
                                   SimpleAverage().predict(fit_preds[cal]), fit_y[cal])
        self.tree, self.tree_weight = model, weight
        return self

    def predict(self, preds):
        if self.tree_weight == 0:
            return SimpleAverage().predict(preds)
        tree = self.tree.predict(preds)
        if self.tree_weight == 1:
            return tree
        return self.tree_weight * tree + (1 - self.tree_weight) * SimpleAverage().predict(preds)

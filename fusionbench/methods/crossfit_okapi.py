"""Fuse one native GP tree and two logistic stackers with out-of-fold Brier weights."""

from __future__ import annotations

from itertools import combinations

import numpy as np
from sklearn.model_selection import StratifiedKFold

from fusionbench.methods.base import Method
from fusionbench.methods.okapi import OKAPIMethod
from fusionbench.methods.stacking import StackingLogit, StackingLR


def brier_weights(probs, labels):
    """Solve the two- or three-component Brier simplex on labeled predictions."""
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels, dtype=int)
    if p.ndim != 3 or p.shape[1] not in (2, 3) or y.shape != (len(p),):
        raise ValueError(
            "Expected probabilities [sample, 2 or 3 methods, class] and aligned labels"
        )
    errors = (
        (p - np.eye(p.shape[2])[y, None, :]).transpose(1, 0, 2).reshape(p.shape[1], -1)
    )
    vertices = np.eye(len(errors))
    candidates = list(vertices)
    for i, j in combinations(range(len(errors)), 2):
        d = errors[j] - errors[i]
        norm = d @ d
        if norm > 0:
            a = np.clip(-(errors[i] @ d) / norm, 0.0, 1.0)
            candidates.append((1 - a) * vertices[i] + a * vertices[j])
    if len(errors) == 3:
        offsets = (errors[1:] - errors[0]).T
        a, b = np.linalg.lstsq(offsets, -errors[0], rcond=None)[0]
        w = np.array([1 - a - b, a, b])
        if w.min() >= 0:
            candidates.append(w)
    gram = errors @ errors.T
    return min(candidates, key=lambda w: w @ gram @ w).copy()


class CrossFitOKAPI(Method):
    """Estimate convex fusion weights on rows unseen by each component."""

    name = "okapi_crossfit_stack"

    @classmethod
    def available(cls):
        return OKAPIMethod.available()

    def __init__(self, seed=0):
        self.seed = seed

    def _fit_components(self, x, y, indices, fold):
        from fusionbench.eval.full_suite_point import fingerprint
        from fusionbench.eval.seed_geometry_standard import fit_seed_geometry

        sub_seed = int(
            np.random.SeedSequence([self.seed, fold, 781]).generate_state(1)[0]
        )
        order = np.random.default_rng(sub_seed).permutation(indices)
        fit, select = np.split(order, [int(0.8 * len(order))])
        gp = fit_seed_geometry(
            x[fit],
            y[fit],
            x[select],
            y[select],
            seed=sub_seed,
            shape="mixed_branch",
            budgets=(600,),
        )[600]
        stackers = [
            s.fit(x[indices], y[indices], task=self.task, n_classes=self.n_classes)
            for s in (StackingLogit(), StackingLR())
        ]
        counts = {
            "train": len(indices),
            "fit_gp": len(fit),
            "select_gp": len(select),
            "calls": gp.calls,
            "unique": gp.n_trees,
            "gp_digest": gp.path_digest,
            "train_sha256": fingerprint(indices),
            "fit_sha256": fingerprint(fit),
            "select_sha256": fingerprint(select),
        }
        return (*stackers, gp.val_tree), counts

    @staticmethod
    def _predict_components(models, x):
        from fusionbench.eval.suite_point_gp import predict_tree

        logit, raw, tree = models
        return np.stack(
            [logit.predict(x), raw.predict(x), predict_tree(tree, x)], axis=1
        )

    def fit(self, fit_preds, fit_y, *, task, n_classes):
        from fusionbench.eval.full_suite_point import fingerprint

        super().fit(fit_preds, fit_y, task=task, n_classes=n_classes)
        x = np.asarray(fit_preds, dtype=np.float64)
        y = np.asarray(fit_y).ravel()
        _, counts = np.unique(y, return_counts=True)
        if x.ndim != 3 or len(x) != len(y) or x.shape[2] != n_classes or x.shape[1] < 2:
            raise ValueError(
                "Expected aligned labels and [sample, provider, class] probabilities"
            )
        if len(counts) < 2 or min(counts) < 3:
            raise ValueError(
                "Three-fold fitting needs at least three rows per observed class"
            )
        folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.seed)
        self.fold_records_ = []
        oof = np.full((len(y), 3, n_classes), np.nan)
        for k, (train, holdout) in enumerate(folds.split(x, y)):
            models, info = self._fit_components(x, y, train, k)
            oof[holdout] = self._predict_components(models, x[holdout])
            self.fold_records_.append(
                {
                    **info,
                    "holdout_sha256": fingerprint(holdout),
                    "holdout": len(holdout),
                }
            )
        if not np.isfinite(oof).all():
            raise ValueError("Missing or invalid held-out component probability")
        self.oof_ = oof
        self.weights_ = brier_weights(oof, y)
        self.linear_weights_ = brier_weights(oof[:, :2], y)
        self.models_, info = self._fit_components(x, y, np.arange(len(y)), 3)
        self.fold_records_.append({**info, "holdout_sha256": None, "holdout": 0})
        return self

    def component_predictions(self, preds):
        return self._predict_components(self.models_, preds)

    def predict(self, preds):
        return np.einsum("nmc,m->nc", self.component_predictions(preds), self.weights_)

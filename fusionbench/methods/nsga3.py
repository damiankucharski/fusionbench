"""NSGA-III weighted-average baseline (multi-objective weight evolution).

Evolves a non-negative weight vector over models (sum-to-one), optimising
macro precision / recall / F1 simultaneously, then selects the Pareto member with
best fit macro-F1. Expresses only linear weighted averages — no gating.
"""

from __future__ import annotations

import numpy as np

from fusionbench.eval.metrics import normalize_proba
from fusionbench.methods.base import Method


class NSGA3(Method):
    name = "nsga3"

    def __init__(self, pop_size: int = 40, n_gen: int = 30, seed: int = 0):
        self.pop_size, self.n_gen, self.seed = pop_size, n_gen, seed

    @classmethod
    def available(cls) -> bool:
        try:
            import pymoo  # noqa: F401
            return True
        except Exception:
            return False

    def fit(self, fit_preds, fit_y, *, task, n_classes):
        super().fit(fit_preds, fit_y, task=task, n_classes=n_classes)
        from pymoo.algorithms.moo.nsga3 import NSGA3 as PymooNSGA3
        from pymoo.core.problem import Problem
        from pymoo.optimize import minimize
        from pymoo.util.ref_dirs import get_reference_directions
        from sklearn.metrics import f1_score, precision_score, recall_score

        k = fit_preds.shape[1]
        y = fit_y.ravel()

        def macro_prf(weights):
            w = np.clip(weights, 0, None)
            w = w / w.sum() if w.sum() > 0 else np.ones(k) / k
            fused = np.tensordot(w, fit_preds, axes=(0, 1))
            yhat = fused.argmax(axis=1)
            return (
                precision_score(y, yhat, average="macro", zero_division=0),
                recall_score(y, yhat, average="macro", zero_division=0),
                f1_score(y, yhat, average="macro", zero_division=0),
            )

        class WeightProblem(Problem):
            def __init__(self):
                super().__init__(n_var=k, n_obj=3, xl=0.0, xu=1.0)

            def _evaluate(self, X, out, *args, **kwargs):
                F = np.array([[-v for v in macro_prf(row)] for row in X])
                out["F"] = F

        ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=8)
        algo = PymooNSGA3(pop_size=self.pop_size, ref_dirs=ref_dirs)
        res = minimize(WeightProblem(), algo, ("n_gen", self.n_gen),
                       seed=self.seed, verbose=False)

        X = np.atleast_2d(res.X)
        best_i = int(np.argmax([macro_prf(row)[2] for row in X]))
        w = np.clip(X[best_i], 0, None)
        self.w = w / w.sum() if w.sum() > 0 else np.ones(k) / k
        return self

    def predict(self, preds):
        return normalize_proba(np.tensordot(self.w, preds, axes=(0, 1)))

"""TPOT AutoML baseline (tree-based pipeline search over the proba features).

Searches the sklearn pipeline space (incl. nonlinear meta-models) on the
concatenated probability features ``[N, K*C]``. It can approximate gating /
routing, at substantial fit-time cost. Optional dependency; skipped when TPOT is
not installed.
"""

from __future__ import annotations

import numpy as np

from fusionbench.eval.metrics import normalize_proba
from fusionbench.methods.base import Method, flatten


class TPOTMethod(Method):
    name = "tpot"

    def __init__(self, max_time_mins: float = 5.0, max_eval_time_mins: float = 1.0,
                 cv: int = 5, seed: int = 0, n_jobs: int = 1):
        self.max_time_mins = max_time_mins
        self.max_eval_time_mins = max_eval_time_mins
        self.cv = cv
        self.seed = seed
        self.n_jobs = n_jobs

    @classmethod
    def available(cls) -> bool:
        try:
            import tpot  # noqa: F401
            return True
        except Exception:
            return False

    def fit(self, fit_preds, fit_y, *, task, n_classes):
        super().fit(fit_preds, fit_y, task=task, n_classes=n_classes)
        from tpot import TPOTClassifier

        self.model = TPOTClassifier(
            scorers=["f1_macro"],
            scorers_weights=[1],
            cv=self.cv,
            max_time_mins=self.max_time_mins,
            max_eval_time_mins=self.max_eval_time_mins,
            n_jobs=self.n_jobs,
            random_state=self.seed,
            verbose=0,
        )
        self.model.fit(flatten(fit_preds), fit_y.ravel())
        self.classes_ = np.unique(fit_y)
        return self

    def predict(self, preds):
        proba = self.model.predict_proba(flatten(preds))
        out = np.zeros((proba.shape[0], self.n_classes))
        for j, cls in enumerate(self.classes_):
            out[:, int(cls)] = proba[:, j]
        return normalize_proba(out)

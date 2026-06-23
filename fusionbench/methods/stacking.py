"""Logistic-regression stackers (learned linear meta-models).

``StackingLR`` operates on the raw probability features (the classic stacking
baseline). ``StackingLogit`` operates on logit-transformed features; since for
calibrated independent models the Bayes rule is a sum of logits, the logit
stacker is near-oracle on the conditionally-independent pathologies — a strong,
fair competitor that is *not* OKAPI.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fusionbench.eval.metrics import normalize_proba
from fusionbench.methods.base import Method, flatten


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


class _Stacker(Method):
    use_logit = False

    def __init__(self, c_value: float = 1.0, max_iter: int = 2000):
        self.c_value, self.max_iter = c_value, max_iter

    def _features(self, preds: np.ndarray) -> np.ndarray:
        x = flatten(preds)
        return _logit(x) if self.use_logit else x

    def fit(self, fit_preds, fit_y, *, task, n_classes):
        super().fit(fit_preds, fit_y, task=task, n_classes=n_classes)
        self.model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=self.c_value, solver="lbfgs", max_iter=self.max_iter),
        )
        self.model.fit(self._features(fit_preds), fit_y.ravel())
        self.classes_ = self.model.classes_
        return self

    def predict(self, preds):
        proba = self.model.predict_proba(self._features(preds))
        # Re-order/pad columns to canonical 0..C-1 in case a class is absent.
        out = np.zeros((proba.shape[0], self.n_classes))
        for j, cls in enumerate(self.classes_):
            out[:, int(cls)] = proba[:, j]
        return normalize_proba(out)


class StackingLR(_Stacker):
    name = "stacking_lr"
    use_logit = False


class StackingLogit(_Stacker):
    name = "stacking_logit"
    use_logit = True

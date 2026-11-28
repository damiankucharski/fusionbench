"""Averaging / selection baselines (no learned meta-model)."""

from __future__ import annotations

import numpy as np

from fusionbench.eval.metrics import HIGHER_BETTER, normalize_proba, score
from fusionbench.methods.base import Method


def _model_val_f1(
    fit_preds: np.ndarray, fit_y: np.ndarray, n_classes: int
) -> np.ndarray:
    return np.array(
        [
            score(fit_y, fit_preds[:, m, :], n_classes, metrics=("macro_f1",))[
                "macro_f1"
            ]
            for m in range(fit_preds.shape[1])
        ]
    )


class SimpleAverage(Method):
    name = "simple_average"

    def predict(self, preds):
        return normalize_proba(preds.mean(axis=1))


class BestSingle(Method):
    name = "best_single"

    def fit(self, fit_preds, fit_y, *, task, n_classes):
        super().fit(fit_preds, fit_y, task=task, n_classes=n_classes)
        self.best = int(np.argmax(_model_val_f1(fit_preds, fit_y, n_classes)))
        return self

    def predict(self, preds):
        return normalize_proba(preds[:, self.best, :])


class TopK(Method):
    def __init__(self, k: int = 5):
        self.k = k
        self.name = f"top_{k}"

    def fit(self, fit_preds, fit_y, *, task, n_classes):
        super().fit(fit_preds, fit_y, task=task, n_classes=n_classes)
        f1 = _model_val_f1(fit_preds, fit_y, n_classes)
        k = min(self.k, fit_preds.shape[1])
        self.idx = np.argsort(-f1)[:k]
        return self

    def predict(self, preds):
        return normalize_proba(preds[:, self.idx, :].mean(axis=1))


class TopN(Method):
    """Average the best-scoring prefix of providers ranked on the fit split.

    Score each provider individually, sort best to worst, then score the mean
    of each prefix, including the full pool. Equal individual scores retain
    input order. Equal prefix scores favor the shorter prefix.
    """

    name = "top_n"

    def __init__(self, metric: str | None = None):
        if metric is not None and metric not in HIGHER_BETTER:
            raise ValueError(f"Unknown selection metric {metric!r}")
        self.metric = metric

    def fit(self, fit_preds, fit_y, *, task, n_classes, primary_metric="macro_f1"):
        super().fit(fit_preds, fit_y, task=task, n_classes=n_classes)
        metric = self.metric or primary_metric
        if metric not in HIGHER_BETTER:
            raise ValueError(f"Unknown selection metric {metric!r}")
        preds = np.asarray(fit_preds)
        labels = np.asarray(fit_y).ravel()
        if (
            preds.ndim != 3
            or preds.shape[0] != len(labels)
            or preds.shape[2] != n_classes
            or min(preds.shape) < 1
        ):
            raise ValueError(
                "Expected nonempty predictions [sample, provider, class] and matching labels"
            )

        def value(p):
            result = score(labels, p, n_classes, metrics=(metric,))[metric]
            if not np.isfinite(result):
                raise ValueError(f"{metric} is undefined for the fitting sample")
            return result

        individual = np.array([value(preds[:, i, :]) for i in range(preds.shape[1])])
        direction = -1 if HIGHER_BETTER[metric] else 1
        ranking = np.argsort(direction * individual, kind="stable")
        total = np.zeros((len(labels), n_classes), dtype=float)
        prefix_scores = []
        for n, i in enumerate(ranking, start=1):
            total += preds[:, i, :]
            prefix_scores.append(value(total / n))
        self.metric_ = metric
        self.ranking_ = ranking
        self.individual_scores_ = individual
        self.prefix_scores_ = np.asarray(prefix_scores)
        self.n_selected_ = int(np.argmin(direction * self.prefix_scores_)) + 1
        self.selected_indices_ = ranking[: self.n_selected_].copy()
        return self

    def predict(self, preds):
        if not hasattr(self, "selected_indices_"):
            raise ValueError("Fit top_n before prediction")
        preds = np.asarray(preds)
        if preds.ndim != 3 or preds.shape[1:] != (len(self.ranking_), self.n_classes):
            raise ValueError("Prediction provider and class axes must match fitting")
        total = np.zeros((preds.shape[0], self.n_classes), dtype=float)
        for i in self.selected_indices_:
            total += preds[:, i, :]
        return normalize_proba(total / self.n_selected_)


class ValWeighted(Method):
    """Weighted average with weights softmax(beta * (val_f1 - 0.5))."""

    name = "val_weighted"

    def __init__(self, beta: float = 8.0):
        self.beta = beta

    def fit(self, fit_preds, fit_y, *, task, n_classes):
        super().fit(fit_preds, fit_y, task=task, n_classes=n_classes)
        f1 = _model_val_f1(fit_preds, fit_y, n_classes)
        z = self.beta * (f1 - 0.5)
        z -= z.max()
        w = np.exp(z)
        self.w = w / w.sum()
        return self

    def predict(self, preds):
        fused = np.tensordot(self.w, preds, axes=(0, 1))  # [N, C]
        return normalize_proba(fused)

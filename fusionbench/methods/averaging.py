"""Averaging / selection baselines (no learned meta-model)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fusionbench.eval.metrics import HIGHER_BETTER, normalize_proba, score
from fusionbench.methods.base import Method


def _model_val_f1(fit_preds: np.ndarray, fit_y: np.ndarray, n_classes: int) -> np.ndarray:
    return np.array([
        score(fit_y, fit_preds[:, m, :], n_classes)["macro_f1"]
        for m in range(fit_preds.shape[1])
    ])


@dataclass
class CompiledAverage:
    """Raw-input probability prediction and total-variation bounds to the arithmetic pool."""

    raw_model: object
    classes: np.ndarray

    def predict(self, x):
        probability = self.raw_model.predict(x)
        normalized = normalize_proba(np.atleast_2d(probability))
        return normalized[0] if probability.ndim == 1 else normalized

    def predict_labels(self, x):
        return self.classes[np.argmax(self.predict(x), axis=-1)]

    def error_bound(self, x):
        return self.raw_model.error_bound(x)

    def oscillation_bound(self, x):
        return self.raw_model.oscillation_bound(x)


class _AffineAverage(Method):
    def _compile_masses(self, count):
        return np.ones(count)

    def compile(self, expert_weights, expert_biases, *, keep_sources=False):
        """Approximate the fitted arithmetic fusion by one softmax on shared raw features.

        Weights have shape [expert, class, feature] and biases [expert, class].
        Both returned bounds apply after FusionBench probability normalization.
        """
        from okapi.softmax_pool import compile_softmax_pool

        shape = np.shape(expert_weights)
        if len(shape) != 3 or not shape[0] or shape[1] != getattr(self, "n_classes", shape[1]):
            raise ValueError("Expected affine expert weights on the fitted class axis")
        model = compile_softmax_pool(
            expert_weights, expert_biases, self._compile_masses(shape[0]), keep_sources=keep_sources
        )
        return CompiledAverage(model, np.arange(shape[1]))

    def compile_estimators(self, estimators, *, feature_indices=None, n_features=None, keep_sources=False):
        """Compile fitted sklearn affine classifiers, optionally trained on different feature subsets."""
        from fusionbench.methods.sklearn_logits import sklearn_affine_logits

        weights, biases, classes = sklearn_affine_logits(
            estimators, feature_indices=feature_indices, n_features=n_features
        )
        compiled = self.compile(weights, biases, keep_sources=keep_sources)
        compiled.classes = classes
        return compiled


class SimpleAverage(_AffineAverage):
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
    """Select the best equally weighted prefix of one fit-time provider ranking."""

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
        if (preds.ndim != 3 or preds.shape[0] != len(labels) or preds.shape[2] != n_classes
                or min(preds.shape) < 1):
            raise ValueError("Expected nonempty predictions [sample, provider, class] and matching labels")

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
        self.selected_indices_ = ranking[:self.n_selected_].copy()
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


class ValWeighted(_AffineAverage):
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

    def _compile_masses(self, count):
        if not hasattr(self, "w"):
            raise ValueError("Fit val_weighted before compiling its learned weights")
        if len(self.w) != count:
            raise ValueError("Expert count must match the fitted validation weights")
        return self.w

    def predict(self, preds):
        fused = np.tensordot(self.w, preds, axes=(0, 1))  # [N, C]
        return normalize_proba(fused)

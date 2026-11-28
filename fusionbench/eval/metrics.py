"""Classification + calibration metrics over probability matrices."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

METRIC_NAMES = (
    "accuracy",
    "balanced_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "roc_auc",
    "average_precision",
    "log_loss",
    "ece",
    "brier",
)

# Higher-is-better flag per metric (for ranking / "winner" logic).
HIGHER_BETTER = {m: True for m in METRIC_NAMES}
HIGHER_BETTER["log_loss"] = False
HIGHER_BETTER["ece"] = False
HIGHER_BETTER["brier"] = False


def normalize_proba(proba: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(proba, dtype=float), 1e-12, None)
    return p / p.sum(axis=1, keepdims=True)


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    """Top-label expected calibration error."""
    conf = p.max(axis=1)
    pred = p.argmax(axis=1)
    correct = (pred == y).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y)
    for lo, hi in pairwise(bins):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += (m.sum() / n) * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def brier_score(y: np.ndarray, p: np.ndarray, n_classes: int) -> float:
    onehot = label_binarize(y, classes=list(range(n_classes)))
    if n_classes == 2:
        onehot = np.hstack([1 - onehot, onehot])
    return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))


def score(
    y_true: np.ndarray, proba: np.ndarray, n_classes: int, *, metrics=METRIC_NAMES
) -> dict[str, float]:
    """Compute requested metrics for a single-label probability matrix [N, C]."""
    y = np.asarray(y_true).astype(int).ravel()
    p = normalize_proba(proba)
    y_hat = p.argmax(axis=1)

    def auc():
        if n_classes == 2:
            return roc_auc_score(y, p[:, 1])
        return roc_auc_score(y, p, multi_class="ovr", average="macro")

    def average_precision():
        if n_classes == 2:
            return average_precision_score(y, p[:, 1])
        return average_precision_score(
            label_binarize(y, classes=np.arange(n_classes)), p, average="macro"
        )

    scorers = {
        "accuracy": lambda: accuracy_score(y, y_hat),
        "balanced_accuracy": lambda: balanced_accuracy_score(y, y_hat),
        "macro_precision": lambda: precision_score(
            y, y_hat, average="macro", zero_division=0
        ),
        "macro_recall": lambda: recall_score(
            y, y_hat, average="macro", zero_division=0
        ),
        "macro_f1": lambda: f1_score(y, y_hat, average="macro", zero_division=0),
        "roc_auc": auc,
        "average_precision": average_precision,
        "log_loss": lambda: log_loss(y, p, labels=list(range(n_classes))),
        "ece": lambda: expected_calibration_error(y, p),
        "brier": lambda: brier_score(y, p, n_classes),
    }
    out = {}
    for metric in metrics:
        try:
            out[metric] = float(scorers[metric]())
        except ValueError:
            if metric != "roc_auc":
                raise
            out[metric] = float("nan")
    return out

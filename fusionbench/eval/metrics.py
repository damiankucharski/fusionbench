"""Classification + calibration metrics over probability matrices."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
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
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += (m.sum() / n) * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def brier_score(y: np.ndarray, p: np.ndarray, n_classes: int) -> float:
    onehot = label_binarize(y, classes=list(range(n_classes)))
    if n_classes == 2:
        onehot = np.hstack([1 - onehot, onehot])
    return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))


def score(y_true: np.ndarray, proba: np.ndarray, n_classes: int) -> dict[str, float]:
    """Full metric set for a single-label proba matrix ``[N, C]``."""
    y = np.asarray(y_true).astype(int).ravel()
    p = normalize_proba(proba)
    y_hat = p.argmax(axis=1)
    out = {
        "accuracy": float(accuracy_score(y, y_hat)),
        "balanced_accuracy": float(balanced_accuracy_score(y, y_hat)),
        "macro_precision": float(precision_score(y, y_hat, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y, y_hat, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y, y_hat, average="macro", zero_division=0)),
    }
    try:
        if n_classes == 2:
            out["roc_auc"] = float(roc_auc_score(y, p[:, 1]))
        else:
            out["roc_auc"] = float(roc_auc_score(y, p, multi_class="ovr", average="macro"))
    except ValueError:
        out["roc_auc"] = float("nan")
    out["log_loss"] = float(log_loss(y, p, labels=list(range(n_classes))))
    out["ece"] = expected_calibration_error(y, p)
    out["brier"] = brier_score(y, p, n_classes)
    return out

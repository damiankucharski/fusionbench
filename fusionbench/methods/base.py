"""Fusion-method interface.

Every method selects/fits on the ``fit`` split and produces a probability matrix
``[N, C]`` on a query split. Methods only ever see base-model predictions, never
labels at predict time and never the oracle.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Method(ABC):
    name: str = "method"

    @classmethod
    def available(cls) -> bool:
        """Whether optional dependencies for this method are importable."""
        return True

    def fit(self, fit_preds: np.ndarray, fit_y: np.ndarray, *, task: str, n_classes: int):
        """Fit/select on ``fit_preds`` ``[N, K, C]``. Returns self."""
        self.task = task
        self.n_classes = n_classes
        return self

    @abstractmethod
    def predict(self, preds: np.ndarray) -> np.ndarray:
        """Return fused probabilities ``[N, C]`` for ``preds`` ``[N, K, C]``."""


def flatten(preds: np.ndarray) -> np.ndarray:
    """``[N, K, C]`` -> ``[N, K*C]`` feature matrix for meta-learners."""
    n = preds.shape[0]
    return preds.reshape(n, -1)

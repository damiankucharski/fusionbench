"""Fixed min, max and threshold rules over all providers.

For binary tasks each rule combines the positive-class probabilities. For
multiclass tasks it combines each class separately and normalizes the result.
These baselines measure how well a single native tree operator fits a scenario.
"""

from __future__ import annotations

import numpy as np

from fusionbench.eval.metrics import normalize_proba
from fusionbench.methods.base import Method
from fusionbench.rules import eval_rule


class OperatorMethod(Method):
    def __init__(self, op: str, tau: float = 0.5):
        self.op = op
        self.tau = tau
        self.name = f"op_{op}"

    def _reduce(self, mat: np.ndarray) -> np.ndarray:
        """Reduce ``[N, K]`` scores over the K models with this operator."""
        if self.op == "min":
            return mat.min(axis=1)
        if self.op == "max":
            return mat.max(axis=1)
        if self.op == "mean":
            return mat.mean(axis=1)
        # far / close: per-row pick the score farthest / closest to tau
        dist = (mat - self.tau) ** 2
        idx = dist.argmax(axis=1) if self.op == "far" else dist.argmin(axis=1)
        return np.take_along_axis(mat, idx[:, None], axis=1)[:, 0]

    def predict(self, preds):
        if preds.shape[2] == 2:
            # Binary: operate on the positive-class probability so the operator
            # has coherent semantics (min == AND/veto, max == OR/rescue,
            # far == most-confident), matching OKAPI's column-1 decision.
            pos = self._reduce(preds[:, :, 1])
            return np.stack([1.0 - pos, pos], axis=1)
        # Multiclass: per-cell selection over models, then renormalize.
        k = preds.shape[1]
        rule: dict = {"op": self.op, "args": [{"model": i} for i in range(k)]}
        if self.op in ("far", "close"):
            rule["tau"] = self.tau
        return normalize_proba(eval_rule(preds, rule))


class OpMin(OperatorMethod):
    def __init__(self):
        super().__init__("min")


class OpMax(OperatorMethod):
    def __init__(self):
        super().__init__("max")


class OpFar(OperatorMethod):
    def __init__(self, tau: float = 0.5):
        super().__init__("far", tau)


class OpClose(OperatorMethod):
    def __init__(self, tau: float = 0.5):
        super().__init__("close", tau)

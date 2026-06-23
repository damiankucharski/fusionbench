"""Fixed symbolic-operator baselines (OKAPI's atoms, parameter-free).

Each applies one OKAPI operator over *all* base models. They are legitimate,
cheap, calibration-preserving fusion rules and serve two roles: (1) a baseline
family showing that the *matching* operator wins its pathology (min for
AND/veto, max for OR/rescue, far-from-0.5 for confidence routing); (2) a local,
OKAPI-free check of what OKAPI's operator class can achieve — OKAPI's added value
is discovering the right operator / composition automatically and parsimoniously.
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

    def predict(self, preds):
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

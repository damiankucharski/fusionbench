"""Symbolic fusion-rule evaluator (mirrors OKAPI alternating-tree semantics).

A rule is a nested dict expression over base-model predictions ``[N, K, C]``:

- Leaf:        ``{"model": i}``                      -> ``preds[:, i, :]``
- mean/max/min:``{"op": "mean", "args": [...]}``     -> element-wise over args
- weighted:    ``{"op": "weighted", "args": [...], "weights": [...]}``
- far/close:   ``{"op": "far", "tau": 0.5, "args": [...]}``  per-cell selection
               of the arg farthest / closest to ``tau`` (squared distance)

This is exactly the operator set OKAPI evolves. It is used ONLY for *post-hoc
interpretation* — e.g. to check how closely a simple symbolic operator
approximates the analytic Bayes oracle, or to express an evolved tree. It is
deliberately **never** used to generate data or to define a scenario's oracle
(the ceiling is the analytic Bayes posterior in :meth:`Scenario.bayes_oracle`),
so the benchmark is not circular / partial toward operator-based methods.
"""

from __future__ import annotations

import numpy as np


def eval_rule(preds: np.ndarray, rule: dict) -> np.ndarray:
    """Evaluate a fusion ``rule`` over ``preds`` ``[N, K, C]`` -> ``[N, C]``."""
    if "model" in rule:
        return preds[:, rule["model"], :]

    op = rule["op"]
    args = [eval_rule(preds, a) for a in rule["args"]]
    stack = np.stack(args, axis=0)  # [A, N, C]

    if op == "mean":
        return stack.mean(axis=0)
    if op == "max":
        return stack.max(axis=0)
    if op == "min":
        return stack.min(axis=0)
    if op == "weighted":
        w = np.asarray(rule["weights"], dtype=float)
        w = w / w.sum()
        return np.tensordot(w, stack, axes=(0, 0))
    if op in ("far", "close"):
        tau = float(rule.get("tau", 0.5))
        dist = (stack - tau) ** 2  # [A, N, C]
        idx = np.argmax(dist, axis=0) if op == "far" else np.argmin(dist, axis=0)
        return np.take_along_axis(stack, idx[None, ...], axis=0)[0]
    raise ValueError(f"unknown op {op!r}")

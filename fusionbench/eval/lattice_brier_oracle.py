"""Exact empirical Brier minimization over all nonconstant MIN/MAX trees.

The binary64 input scores are dyadic rationals. Python-integer capacities keep
the minimum-cut certificate exact even when two empirical losses nearly tie.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np


def _scores(x, dimension=None):
    x = np.asarray(x, dtype=np.float64)
    if (
        x.ndim != 2
        or not 1 <= x.shape[1] <= 12
        or (dimension is not None and x.shape[1] != dimension)
    ):
        raise ValueError(
            "supply a matrix with 1–12 provider columns matching the fitted dimension"
        )
    if not np.all(np.isfinite(x)) or np.any((x < 0) | (x > 1)):
        raise ValueError("provider scores must be finite probabilities")
    return x


@dataclass(frozen=True)
class LatticeBrierOracle:
    lower: np.ndarray
    upper: np.ndarray
    minimum_loss: Fraction

    @property
    def dimension(self):
        return len(self.lower).bit_length() - 1

    def bounds(self, x):
        """Pointwise predictions of the least and greatest empirical minimizers."""
        x = _scores(x, self.dimension)
        order = np.argsort(-x, axis=1, kind="stable")
        active = np.bitwise_or.accumulate(1 << order, axis=1)
        rows = np.arange(len(x))

        def predictions(truth):
            # First accepted descending prefix gives the lattice value. The
            # final prefix contains all providers and is always accepted.
            first = np.argmax(truth[active], axis=1)
            return x[rows, order[rows, first]]

        return predictions(self.lower), predictions(self.upper)

    def predict(self, x):
        return self.bounds(x)[0]

    def terms(self):
        """Minimal accepted subsets giving the least optimum in max-of-mins form."""
        terms = []
        for mask in np.flatnonzero(self.lower):
            columns = tuple(j for j in range(self.dimension) if mask & (1 << j))
            if all(not self.lower[mask ^ (1 << j)] for j in columns):
                terms.append(columns)
        return tuple(terms)


def _reachable(residual, root, reverse=False):
    seen = {root}
    queue = [root]
    while queue:
        node = queue.pop()
        neighbors = residual.pred[node] if reverse else residual.succ[node]
        for other, edge in neighbors.items():
            if other not in seen and edge["capacity"] > edge["flow"]:
                seen.add(other)
                queue.append(other)
    return seen


def fit_lattice_brier(x, y):
    """Return the exact fit optimum with a minimum-weight closure on 2**K cells.

    For each labelled row, t_1 <= ... <= t_K are the sorted inputs and
    t_0 = 0. Add (t_j-y)^2 - (t_{j-1}-y)^2 to the unary weight of the
    subset on or above t_j. A tree's threshold decisions form an upper set.
    Summing the increments for accepted subsets telescopes to its squared
    error minus the fixed (0-y)^2. Tied thresholds have zero increment.
    """
    from networkx import DiGraph
    from networkx.algorithms.flow import preflow_push

    x = _scores(x)
    y = np.asarray(y)
    if len(x) == 0 or y.shape != (len(x),) or not np.all((y == 0) | (y == 1)):
        raise ValueError("supply a nonempty matrix and one binary label per row")
    n, d = x.shape
    values, inverse = np.unique(x, return_inverse=True)
    ratios = [float(v).as_integer_ratio() for v in values]
    denom = max(b for _, b in ratios)
    numerators = [a * (denom // b) for a, b in ratios]
    cells = 1 << d
    full = cells - 1
    unary = [0] * cells
    for row, label in zip(inverse.reshape(x.shape), y, strict=True):
        mask = full
        target = denom if label else 0
        prev = target * target
        for j in np.argsort(row, kind="stable"):
            loss = (numerators[row[j]] - target) ** 2
            unary[mask] += loss - prev
            prev = loss
            mask ^= 1 << int(j)
    source, sink = cells, cells + 1
    hard = 1 + sum(abs(cost) for cost in unary)
    graph = DiGraph()
    for mask, cost in enumerate(unary):
        if cost < 0:
            graph.add_edge(source, mask, capacity=-cost)
        if cost > 0:
            graph.add_edge(mask, sink, capacity=cost)
        for j in range(d):
            bit = 1 << j
            if not mask & bit:
                graph.add_edge(mask, mask | bit, capacity=hard)
    # Combine anchors with any existing unary edge in the same direction.
    for u, v in ((source, full), (0, sink)):
        old = graph.get_edge_data(u, v, {}).get("capacity", 0)
        graph.add_edge(u, v, capacity=hard + old)
    residual = preflow_push(graph, source, sink)
    low = _reachable(residual, source)
    high_excluded = _reachable(residual, sink, reverse=True)
    lower = np.array([m in low for m in range(cells)])
    upper = np.array([m not in high_excluded for m in range(cells)])
    base = int(y.sum()) * denom**2
    optimum = base + sum(min(c, 0) for c in unary) + residual.graph["flow_value"]
    # The attaining closure independently checks the graph/cut offset.
    for truth in (lower, upper):
        assert not truth[0] and truth[-1]
        assert optimum == base + sum(
            c for c, chosen in zip(unary, truth, strict=True) if chosen
        )
    return LatticeBrierOracle(lower, upper, Fraction(optimum, n * denom**2))

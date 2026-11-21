"""Empirical optimum over all one-node ordered native gates on the 1/64 alphabet."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FittedGate:
    parent: int
    child: int
    direction: str
    threshold: float
    loss_numerator: int
    loss_denominator: int
    minimizers_on_grid: int

    def predict(self, x):
        a, b = x[:, self.parent], x[:, self.child]
        da, db = np.abs(a - self.threshold), np.abs(b - self.threshold)
        use_a = da <= db if self.direction == "close" else da >= db
        return np.where(use_a, a, b)


def fit_gate(x, y, *, fixed_threshold=False):
    """Count exact dyadic squared errors with stable lexicographic first tie."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y)
    if x.ndim != 2 or x.shape[1] < 2 or len(x) == 0 or y.shape != (len(x),):
        raise ValueError(
            "Expected a nonempty rows-by-providers matrix and one label per row"
        )
    if not np.all((y == 0) | (y == 1)) or not np.all(
        np.isfinite(x) & (x >= 0) & (x <= 1)
    ):
        raise ValueError("Expected finite probabilities and binary labels")
    scaled = x * 64
    if not np.array_equal(scaled, np.rint(scaled)):
        raise ValueError("Exact threshold coverage requires values on the 1/64 grid")
    scores = scaled.astype(np.int64)
    target = y.astype(np.int64) * 64
    thresholds = np.array([128]) if fixed_threshold else np.arange(257)
    best, best_gate, ties = None, None, 0
    for i in range(x.shape[1]):
        a = scores[:, i]
        cost_a = (a - target) ** 2
        for j in range(x.shape[1]):
            if i == j:
                continue
            b = scores[:, j]
            cost_b = (b - target) ** 2
            # A row compares T (in units 1/256) to its midpoint m=2(a+b).
            # Aggregate the signed cost changes per breakpoint. At exact ties
            # both operators choose the first value. T outside a breakpoint
            # selects opposite operands for CLOSE versus FAR.
            mid = 2 * (a + b)
            change = cost_a - cost_b
            plus = np.zeros(257, dtype=np.int64)
            minus = np.zeros(257, dtype=np.int64)
            np.add.at(plus, mid[a > b], change[a > b])
            np.add.at(minus, mid[a < b], change[a < b])
            left_plus = np.cumsum(plus) - plus
            left_minus = np.cumsum(minus) - minus
            right_plus = plus.sum() - left_plus - plus
            right_minus = minus.sum() - left_minus - minus
            at = plus + minus
            close = cost_b.sum() + left_plus + right_minus + at
            far = cost_b.sum() + right_plus + left_minus + at
            for direction, loss in (("close", close), ("far", far)):
                grid_loss = loss[thresholds]
                local = int(np.min(grid_loss))
                local_ties = int(np.count_nonzero(grid_loss == local))
                if best is None or local < best:
                    best, ties = local, local_ties
                    best_gate = (
                        i,
                        j,
                        direction,
                        int(thresholds[np.argmin(grid_loss)]) / 256,
                    )
                elif local == best:
                    ties += local_ties
    return FittedGate(*best_gate, best, len(x) * 64**2, ties)

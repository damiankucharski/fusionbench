"""Fit one scalar OKAPI CLOSE/FAR threshold against known downstream losses."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from fractions import Fraction

import numpy as np


@dataclass(frozen=True)
class ThresholdFit:
    threshold: float
    fit_loss: float
    n_events: int
    n_thresholds: int
    n_row_events: int
    method: str


def _scaled_integers(matrix):
    """Represent every supplied binary64 value in one common integer unit."""
    values, inverse = np.unique(matrix, return_inverse=True)
    ratios = [float(v).as_integer_ratio() for v in values]
    denominator = max(d for _, d in ratios)
    integers = np.array([n * (denominator // d) for n, d in ratios], dtype=object)
    return integers[inverse.reshape(matrix.shape)], denominator


def _event_deltas(operands, losses, values, close):
    events = {}
    left_loss = row_events = 0
    for row, scores, scaled in zip(operands, losses, values, strict=True):
        ordered = np.argsort(row, kind="stable")
        distinct = []
        for i in ordered:
            if not distinct or row[i] != row[distinct[-1]]:
                distinct.append(i)
            elif scores[i] != scores[distinct[-1]]:
                raise ValueError(
                    "Equal operands in a row must have equal downstream losses"
                )
        left_loss += scores[distinct[0] if close else distinct[-1]]
        if len(distinct) == 1:
            continue
        # Left and right identify the chosen operand on either side of the event.
        pairs = itertools.pairwise(distinct) if close else [(distinct[-1], distinct[0])]
        for left, right in pairs:
            midpoint = scaled[left] + scaled[right]
            tie_delta, right_delta = events.get(midpoint, (0, 0))
            events[midpoint] = (
                tie_delta + scores[min(left, right)] - scores[left],
                right_delta + scores[right] - scores[left],
            )
            row_events += 1
    return left_loss, events, row_events


def _compare(value, numerator, denominator):
    n, d = value.as_integer_ratio()
    return n * denominator - numerator * d


def _interior(left, right, denominator):
    """One binary64 strictly within a rational interval, if any can fit."""
    value = float(Fraction(left + right, 2 * denominator))
    if (
        _compare(value, left, denominator) > 0
        and _compare(value, right, denominator) < 0
    ):
        return value
    lower = float(Fraction(left, denominator))
    if _compare(lower, left, denominator) <= 0:
        lower = float(np.nextafter(lower, np.inf))
    return lower if _compare(lower, right, denominator) < 0 else None


def _candidates(left_loss, events, denominator):
    """Sweep ideal-distance costs at intervals and exactly representable ties."""
    candidates = {0.0: left_loss}
    previous = 0
    for position, (tie_delta, right_delta) in sorted(events.items()):
        interior = _interior(previous, position, denominator)
        if interior is not None:
            candidates[interior] = left_loss
        pivot = float(Fraction(position, denominator))
        if _compare(pivot, position, denominator) == 0:
            candidates[pivot] = left_loss + tie_delta
        left_loss += right_delta
        previous = position
    interior = _interior(previous, denominator, denominator)
    if interior is not None:
        candidates[interior] = left_loss
    candidates[1.0] = left_loss
    return candidates


def fit_threshold(operands, downstream_losses, *, close, original):
    """Minimize mean downstream loss for one ordered, scalar threshold gate.

    Each [row, operand] entry is the loss of the whole downstream calculation
    if that operand is selected on that row. Equal-valued operands must give
    equal losses. The return prefers the incumbent on an objective tie, then
    the smallest threshold in the candidate set.

    The candidate set contains 0, 1, the incumbent, every representable exact
    operator tie and one representable point in every nonempty open interval
    between consecutive ties. This exhausts the ideal absolute-distance gate's
    decision regions on binary64 thresholds. The returned loss and selection
    target NumPy's actual binary64 squared distances *on this set*. Extremely
    close or disparate scores can create additional floating-point ties within
    an ideal region. Such incidental changes outside the candidate set are not
    covered by the native-objective guarantee.

    n_events counts distinct rational midpoints, even if not representable
    in binary64. n_row_events counts contributing row boundaries before
    merging. n_thresholds counts unique candidates including the incumbent.
    """
    x = np.asarray(operands, dtype=np.float64)
    costs = np.asarray(downstream_losses, dtype=np.float64)
    if x.ndim != 2 or min(x.shape) < 1 or x.shape != costs.shape:
        raise ValueError(
            "Expected identically shaped, nonempty row-by-operand matrices"
        )
    if not np.isfinite(x).all() or np.any((x < 0) | (x > 1)):
        raise ValueError("Operands must be finite probabilities in [0, 1]")
    if not np.isfinite(costs).all():
        raise ValueError("Downstream losses must be finite")
    if not isinstance(close, (bool, np.bool_)):
        raise TypeError("close must be boolean")
    original = float(original)
    if not 0 <= original <= 1:
        raise ValueError("original must be a finite threshold in [0, 1]")

    values, value_den = _scaled_integers(x)
    losses, loss_den = _scaled_integers(costs)
    left_loss, events, n_row_events = _event_deltas(x, losses, values, close)
    candidates = _candidates(left_loss, events, 2 * value_den)
    rows = np.arange(len(x))
    pick = np.argmin if close else np.argmax

    def native_loss(t):
        with np.errstate(under="ignore"):
            selected = pick(np.square(x - t), axis=1)
        return sum(losses[rows, selected])

    original_loss = native_loss(original)
    # With entries on a 2^-24 grid, all representatives are multiples of
    # 2^-26 or coarser. Subtraction and squaring require at most 52 fraction
    # bits. An off-grid incumbent always receives its literal native score.
    winner = original
    best = original_loss
    fast = value_den <= (1 << 24)
    candidates[original] = original_loss
    for threshold, projected in candidates.items():
        loss = projected if fast else native_loss(threshold)
        if loss < best or (loss == best and winner != original and threshold < winner):
            best, winner = loss, threshold
    if native_loss(winner) != best:
        raise ArithmeticError(
            "Event sweep disagrees with the native squared-distance selector"
        )
    return ThresholdFit(
        threshold=winner,
        fit_loss=float(Fraction(best, len(x) * loss_den)),
        n_events=len(events),
        n_thresholds=len(candidates),
        n_row_events=n_row_events,
        method="exact_event_sweep" if fast else "native_candidate_scan",
    )

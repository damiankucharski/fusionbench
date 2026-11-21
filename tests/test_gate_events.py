from fractions import Fraction

import numpy as np
import pytest

from fusionbench.eval.gate_events import fit_threshold


def literal_choices(x, t, close):
    # Compare one operand at a time; strict comparisons preserve the first tie.
    result = []
    for values in x:
        distances = [np.square(v - t) for v in values]
        best = 0
        for j, d in enumerate(distances):
            better = d < distances[best] if close else d > distances[best]
            if better:
                best = j
        result.append(best)
    return result


def literal_loss(x, costs, t, close):
    cols = literal_choices(x, t, close)
    return sum(Fraction(float(costs[i, c])) for i, c in enumerate(cols)) / len(cols)


@pytest.mark.parametrize("close", [False, True])
@pytest.mark.parametrize("seed", range(6))
def test_matches_dense_literal_scan_on_dyadic_inputs(close, seed):
    rng = np.random.default_rng(seed)
    ticks = rng.integers(0, 17, size=(13, 7))
    x = ticks / 16
    loss_by_value = rng.uniform(-1, 1, size=(len(x), 17))
    costs = np.take_along_axis(loss_by_value, ticks, axis=1)
    original = 0.373751
    fit = fit_threshold(x, costs, close=close, original=original)
    # Every exact boundary is a multiple of 1/32; even the narrowest
    # interval therefore contains a 1/64 lattice point.
    grid = list(np.arange(65) / 64) + [original]
    empirical = min(literal_loss(x, costs, t, close) for t in grid)
    assert fit.fit_loss == float(empirical)
    assert literal_loss(x, costs, fit.threshold, close) == empirical
    assert fit.method == "exact_event_sweep"
    assert 0 < fit.n_events <= fit.n_row_events


@pytest.mark.parametrize("close", [False, True])
def test_exact_tie_better_than_both_adjacent_cells(close):
    x = np.array([[0.25, 0.75], [0.75, 0.25]])
    costs = np.array([[0.0, 1.0], [0.0, 1.0]])
    fit = fit_threshold(x, costs, close=close, original=0.25)
    assert fit.threshold == 0.5 and fit.fit_loss == 0
    assert fit.n_row_events == 2 and fit.n_events == 1 and fit.n_thresholds == 5
    assert (
        literal_loss(x, costs, 0.25, close)
        == literal_loss(x, costs, 0.75, close)
        == 0.5
    )


@pytest.mark.parametrize("close", [False, True])
def test_incumbent_wins_when_all_thresholds_tie(close):
    x = [[0.125], [0.75], [0.5]]
    fit = fit_threshold(x, [[1.0], [0.5], [-1.0]], close=close, original=0.38502)
    assert fit.threshold == 0.38502 and fit.fit_loss == float(Fraction(1, 6))
    assert fit.n_events == fit.n_row_events == 0 and fit.n_thresholds == 4


@pytest.mark.parametrize("close", [False, True])
def test_squared_distance_underflow_uses_literal_fallback(close):
    tiny = np.nextafter(0.0, 1.0)
    x = np.array([[tiny, 0.0], [0.0, tiny]])
    costs = np.array([[0.25, 0.75], [-1.0, 0.75]])
    # These distinct operands always tie after squaring at these candidates.
    original = 0.125
    fit = fit_threshold(x, costs, close=close, original=original)
    with np.errstate(under="ignore"):
        direct = literal_loss(x, costs, original, close)
        assert all(
            literal_loss(x, costs, t, close) == direct for t in (0, tiny, 0.5, 1)
        )
    assert fit.method == "native_candidate_scan"
    assert fit.threshold == original and fit.fit_loss == float(direct)
    # The midpoint itself is half the smallest positive binary64.
    assert fit.n_events == 1 and fit.n_row_events == 2 and fit.n_thresholds == 4


def test_adjacent_floats_do_not_invent_a_representable_tie():
    a = 0.5
    b = np.nextafter(a, 1.0)
    x = np.array([[a, b], [b, a]])
    costs = np.array([[0.0, 1.0], [0.0, 1.0]])
    fit = fit_threshold(x, costs, close=True, original=b)
    # The order-sensitive ideal midpoint has loss zero but is unrepresentable.
    # The distances at 1 also retain their order; at 0 subtraction is exact.
    assert fit.n_events == 1
    assert fit.fit_loss == float(
        min(literal_loss(x, costs, t, True) for t in (0, 0.25, a, b, 0.75, 1))
    )
    assert fit.fit_loss == 0.5 and fit.threshold == b


@pytest.mark.parametrize("close", [False, True])
def test_reported_winner_agrees_with_okapi_native_threshold(close, monkeypatch):
    pytest.importorskip("okapi")
    from okapi.backend.backend import Backend
    from okapi.backend.numpy_backend import NumpyBackend
    from okapi.node import CloseThresholdNode, FarThresholdNode

    monkeypatch.setattr(Backend, "_current_backend", NumpyBackend)
    rng = np.random.default_rng(939)
    x = rng.random((60, 4))
    labels = rng.integers(0, 2, size=(len(x), 1))
    costs = (labels - x) ** 2
    fit = fit_threshold(x, costs, close=close, original=0.4809)
    node = (CloseThresholdNode if close else FarThresholdNode)(None, fit.threshold)
    pred = node.op(x.T).reshape((-1, 1))
    literal = (labels - pred)[:, 0] ** 2
    assert fit.fit_loss == float(sum(Fraction.from_float(v) for v in literal) / len(x))
    assert (
        fit.fit_loss
        <= np.mean(
            np.take_along_axis(
                costs, np.asarray(literal_choices(x, 0.4809, close))[:, None], axis=1
            )
        )
        + 1e-15
    )


@pytest.mark.parametrize(
    "x,costs",
    [
        ([[0.5, 0.5]], [[1, 2]]),
        ([], []),
        ([[0.1]], [[np.nan]]),
        ([[1.1]], [[0.0]]),
        ([[0.5]], [[1, 2]]),
    ],
)
def test_rejects_invalid_operands_and_costs(x, costs):
    with pytest.raises(ValueError):
        fit_threshold(x, costs, close=True, original=0.5)

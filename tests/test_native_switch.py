import itertools

import numpy as np
import pytest

from fusionbench.eval.exact_switch import fit_gate
from fusionbench.scenarios.native_switch import (
    NativeSwitch,
    draw_native_switch,
    farther,
)
from fusionbench.scenarios.registry import SCENARIOS


def test_known_switch_and_stationary_controls():
    fit = draw_native_switch(10, 0, 512)
    val = draw_native_switch(10, 1, 512)
    assert (fit.parent, fit.child) == (val.parent, val.child)
    assert fit.parent != fit.child and not np.array_equal(fit.preds, val.preds)
    assert [*SCENARIOS["native_switch"]().model_ids()] == [f"p{i}" for i in range(6)]
    for mixing in [0, 0.5, 1]:
        q = fit.probability(mixing)
        np.testing.assert_array_equal(fit.labels(mixing), fit.uniform < q)
        np.testing.assert_array_equal(
            NativeSwitch(mixing=mixing, role_seed=10).bayes_oracle(fit.preds, {})[:, 1],
            q,
        )
    np.testing.assert_allclose(
        fit.probability(0.5), (fit.probability(0) + fit.probability(1)) / 2
    )
    assert farther(0.25, 0.75, 0.5) == 0.25 and farther(0.75, 0.25, 0.5) == 0.75


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("fixed", [False, True])
def test_gate_optimum_ties_and_winner_against_literal_enumeration(seed, fixed):
    rng = np.random.default_rng(seed)
    x = rng.integers(1, 64, (23, 3)) / 64
    x[:6, 0] = 0.25
    x[:6, 1] = 0.75
    y = rng.integers(2, size=len(x))
    loss = None
    count = 0
    for i, j in itertools.permutations(range(3), 2):
        a, b = x[:, i], x[:, j]
        for op in ("close", "far"):
            for k in [128] if fixed else range(257):
                t = k / 256
                use_a = (
                    (a - t) ** 2 <= (b - t) ** 2
                    if op == "close"
                    else (a - t) ** 2 >= (b - t) ** 2
                )
                value = int(np.sum(((np.where(use_a, a, b) - y) * 64) ** 2))
                if loss is None or value < loss:
                    loss, best, count = value, (i, j, op, t), 1
                elif value == loss:
                    count += 1
    fit = fit_gate(x, y, fixed_threshold=fixed)
    assert (fit.parent, fit.child, fit.direction, fit.threshold) == best
    assert (fit.loss_numerator, fit.minimizers_on_grid) == (loss, count)
    assert (
        np.mean((fit.predict(x) - y) ** 2) == fit.loss_numerator / fit.loss_denominator
    )


def test_generated_switch_is_native_including_distance_ties(monkeypatch):
    pytest.importorskip("okapi")
    from okapi.backend.backend import Backend
    from okapi.backend.numpy_backend import NumpyBackend
    from okapi.globals import postprocessing_function
    from okapi.node import FarThresholdNode, ValueNode
    from okapi.tree import Tree

    monkeypatch.setattr(Backend, "_current_backend", NumpyBackend)
    monkeypatch.setattr(postprocessing_function, "_postprocessing_function", np.asarray)
    a, b = np.meshgrid(np.arange(1, 64) / 64, np.arange(1, 64) / 64)
    a, b = a.reshape(-1, 1), b.reshape(-1, 1)
    native = Tree(
        ValueNode([FarThresholdNode([ValueNode([], b, "b")], 27 / 64)], a, "a")
    )
    np.testing.assert_array_equal(native.predict(), farther(a, b, 27 / 64))


def test_machine_neighbors_of_breakpoints_match_the_adjacent_intervals():
    a, b = np.meshgrid(np.arange(1, 64) / 64, np.arange(1, 64) / 64)
    for t in np.arange(1, 128) / 128:
        for direction in (-1, 1):
            machine = np.nextafter(t, t + direction)
            interior = t + direction / 256
            for close in (True, False):

                def select(cut, close):
                    da, db = (a - cut) ** 2, (b - cut) ** 2
                    return np.where(da <= db if close else da >= db, a, b)

                np.testing.assert_array_equal(
                    select(machine, close), select(interior, close)
                )


def test_close_control_reuses_anon_features_and_native_tie_priority():
    from fusionbench.scenarios.native_switch import (
        NativeCloseSwitch,
        conditional_probability,
    )

    far = NativeSwitch(role_seed=905)
    close = NativeCloseSwitch(role_seed=905)
    sample = close.draw(40, np.random.default_rng(22))
    partner = far.draw(40, np.random.default_rng(22))
    np.testing.assert_array_equal(sample.preds, partner.preds)
    np.testing.assert_array_equal(sample.uniform, partner.uniform)
    a, b = sample.preds[:, sample.parent, 1], sample.preds[:, sample.child, 1]
    expected = np.where(abs(a - 37 / 64) <= abs(b - 37 / 64), a, b)
    np.testing.assert_array_equal(sample.probability(0), expected)
    np.testing.assert_array_equal(close.bayes_oracle(sample.preds, {})[:, 1], expected)
    np.testing.assert_array_equal(sample.probability(1), partner.probability(1))
    assert (
        conditional_probability(36 / 64, 38 / 64, 0, close=True, threshold=37 / 64)
        == 36 / 64
    )
    assert SCENARIOS["native_close_switch"] is NativeCloseSwitch

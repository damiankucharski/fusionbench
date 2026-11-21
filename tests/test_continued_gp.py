import importlib

import numpy as np
import pytest

from fusionbench.eval.continued_gp import continued_gp
from fusionbench.eval.native_switch import search, tensors
from fusionbench.eval.structural_key_audit import training_files


@pytest.fixture
def inputs(monkeypatch):
    pytest.importorskip("okapi")
    from okapi.backend.backend import Backend
    from okapi.globals import postprocessing_function

    monkeypatch.setattr(Backend, "_current_backend", Backend._current_backend)
    monkeypatch.setattr(
        postprocessing_function,
        "_postprocessing_function",
        postprocessing_function._postprocessing_function,
    )
    rng = np.random.default_rng(483)
    scores = rng.uniform(size=(40, 6))
    labels = rng.integers(0, 2, size=40)
    with training_files(scores[:23], labels[:23]) as (paths, gt):
        yield paths, gt, tensors(scores[23:]), labels[23:]


@pytest.mark.parametrize(
    "offsets, budgets", [([], [2]), ([0, 0], [2]), ([3, 0, np.int64(1), 3], [2, 3, 5])]
)
def test_exact_objective_cost_and_callback_isolation(
    monkeypatch, inputs, offsets, budgets
):
    from okapi import Okapi

    module = importlib.import_module("fusionbench.eval.continued_gp")
    clock = [0]
    evaluations = []
    native = Okapi._calculate_fitnesses

    def observed(engine, trees):
        evaluations.extend(trees)
        clock[0] += len(trees)
        return native(engine, trees)

    monkeypatch.setattr(Okapi, "_calculate_fitnesses", observed)
    monkeypatch.setattr(module, "perf_counter", lambda: clock[0])
    calls_at_callback = []

    def refine(fit, val):
        calls_at_callback.append(len(evaluations))
        fit.root.id = val.root.id = "callback copy"
        clock[0] += 10000
        for offset in offsets:
            clock[0] += 10
            yield offset

    result = continued_gp(*inputs, seed=251, refine_callback=refine, initial_calls=2)
    assert calls_at_callback == [2]
    assert list(result) == budgets
    assert len(evaluations) == max(budgets)
    for calls, checkpoint in result.items():
        assert (
            checkpoint.n_trees == calls
        )  # First six trees are different singleton providers.
        assert checkpoint.fit_tree.root.id != "callback copy"
        assert checkpoint.validation_tree.root.id != "callback copy"
        assert checkpoint.elapsed_seconds == calls


@pytest.mark.parametrize("offset", [-1, 1.25, True])
def test_invalid_offsets(inputs, offset):
    with pytest.raises(ValueError, match="offsets"):
        continued_gp(
            *inputs, seed=251, refine_callback=lambda *_: [offset], initial_calls=1
        )


def test_keeps_native_trajectory_across_checkpoints(inputs):
    reference = search(*inputs, seed=257, use_thresholds=True)
    extended = continued_gp(
        *inputs, seed=257, refine_callback=lambda *_: [0, 1, 900], initial_calls=300
    )
    assert list(extended) == [300, 301, 1200]
    for calls, visited, _, fit, val in reference:
        actual = extended[calls]
        assert actual.n_trees == visited
        assert actual.fit_tree.structural_key() == fit.structural_key()
        assert actual.validation_tree.structural_key() == val.structural_key()

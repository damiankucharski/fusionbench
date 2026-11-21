"""Continue one six-operator native OKAPI run at externally measured fit costs."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np

from fusionbench.eval.search_continuous import brier

if TYPE_CHECKING:
    from okapi.tree import Tree


@dataclass(frozen=True)
class Checkpoint:
    fit_tree: Tree
    validation_tree: Tree
    n_trees: int
    elapsed_seconds: float


def continued_gp(
    paths,
    gt,
    validation_tensors,
    validation_y,
    seed,
    refine_callback,
    initial_calls=1200,
):
    """Checkpoint a single native run after the initial and all requested extra budgets.

    After `initial_calls` true objective calls, invoke `refine_callback` once
    with copies of the fit and validation champions. The callback returns extra
    objective-call counts relative to that checkpoint. Empty or all-zero offsets
    stop immediately. Trees supplied to the callback share read-only provider
    tensors with the optimizer. Elapsed wall time includes archive scoring and
    excludes the callback and consumption of its offsets.
    """
    from okapi import Okapi
    from okapi.operators import (
        CLOSE_THRESHOLD,
        FAR_THRESHOLD,
        MAX,
        MEAN,
        MIN,
        WEIGHTED_MEAN,
    )

    if (
        isinstance(initial_calls, bool)
        or not isinstance(initial_calls, Integral)
        or initial_calls < 1
    ):
        raise ValueError("initial_calls must be a positive integer")
    initial_calls = int(initial_calls)
    remaining = {initial_calls}
    archive, checkpoints = {}, {}
    calls = 0
    paused = 0.0
    validation_y = np.asarray(validation_y).ravel()
    training_loss = None

    def objective(pred, y):
        nonlocal training_loss
        training_loss = brier(pred.ravel(), y.ravel())
        return 1 - training_loss

    class ReachedFinalCheckpoint(Exception):
        pass

    def checkpoint():
        nonlocal paused
        fit = min(archive.values(), key=lambda entry: entry[0])[2]
        val = min(archive.values(), key=lambda entry: (entry[1], entry[0]))[2]
        checkpoints[calls] = Checkpoint(
            fit, val, len(archive), perf_counter() - start - paused
        )
        remaining.remove(calls)
        if calls == initial_calls:
            pause_start = perf_counter()
            for offset in refine_callback(fit.copy(), val.copy()):
                if (
                    isinstance(offset, bool)
                    or not isinstance(offset, Integral)
                    or offset < 0
                ):
                    raise ValueError(
                        "Extra objective-call offsets must be nonnegative integers"
                    )
                if offset:
                    remaining.add(calls + int(offset))
            paused += perf_counter() - pause_start
        if not remaining:
            raise ReachedFinalCheckpoint

    class Observed(Okapi):
        def _calculate_fitnesses(self, trees=None):
            nonlocal calls
            results = []
            for tree in self.population if trees is None else trees:
                score = super()._calculate_fitnesses([tree])[0]
                calls += 1
                key = tree.structural_key()
                if key not in archive:
                    candidate = tree.copy()
                    pred = candidate.do_pred_on_another_tensors(
                        current_tensors=validation_tensors
                    ).ravel()
                    archive[key] = training_loss, brier(pred, validation_y), candidate
                if calls in remaining:
                    checkpoint()
                results.append(score)
            return np.stack(results)

    start = perf_counter()
    gp = Observed(
        paths,
        gt,
        len(paths),
        2,
        2,
        backend="numpy",
        seed=seed,
        allowed_ops=(MEAN, MIN, MAX, WEIGHTED_MEAN, FAR_THRESHOLD, CLOSE_THRESHOLD),
        postprocessing_function=np.asarray,
        objective_functions=(objective,),
        reuse_survivor_fitness=False,
    )
    try:
        while True:
            gp.run_iteration()
    except ReachedFinalCheckpoint:
        pass
    return checkpoints

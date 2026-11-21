"""Cost-matched native GP with the standard FusionBench adapter initialization."""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np

from fusionbench.eval.metrics import normalize_proba
from fusionbench.methods.okapi import OKAPIMethod, _task_postprocessing

if TYPE_CHECKING:
    from okapi.tree import Tree


@dataclass(frozen=True)
class SuiteCheckpoint:
    fit_tree: Tree
    val_tree: Tree
    calls: int
    n_trees: int
    generation: int
    path_digest: str
    elapsed_seconds: float
    fit_brier: float
    validation_brier: float

    @property
    def validation_tree(self):
        return self.val_tree


def _full(preds):
    """Interpret two-dimensional input as a matrix of binary positive probabilities."""
    p = np.asarray(preds, dtype=np.float64)
    if p.ndim == 3 and p.shape[2] == 1:
        p = p[:, :, 0]
    if p.ndim == 2:
        p = np.stack((1 - p, p), axis=2)
    if p.ndim != 3 or min(p.shape) < 1 or p.shape[2] < 2:
        raise ValueError(
            "Expected [rows,providers] binary scores or [rows,providers,classes] probabilities"
        )
    return p


def _targets(labels, n, c):
    values = np.asarray(labels).ravel()
    y = values.astype(np.int64)
    if len(y) != n or not np.array_equal(y, values) or np.any((y < 0) | (y >= c)):
        raise ValueError(
            "Expected one integer class index in range for every input row"
        )
    return y, (y[:, None] if c == 2 else np.eye(c)[y])


def _loss(p, target):
    p = np.asarray(p)
    if target.shape[1] == 1:
        return float(np.mean(np.square(p[:, 0] - target[:, 0])))
    return 0.5 * float(np.mean(np.sum(np.square(p - target), axis=1)))


def _bind(full):
    arr = full[:, :, 1:2] if full.shape[2] == 2 else full
    return {f"m{i}.npy": arr[:, i, :] for i in range(arr.shape[1])}


def predict_tree(tree, preds):
    """Return canonical [row,class] probabilities, even after a different task ran."""
    from okapi.globals import set_backend

    full = _full(preds)
    c = full.shape[2]
    with _task_postprocessing("binary" if c == 2 else "multiclass"):
        set_backend("numpy")
        result = np.asarray(
            tree.do_pred_on_another_tensors(current_tensors=_bind(full))
        )
    if c == 2:
        pos = np.clip(result.ravel(), 0.0, 1.0)
        return np.column_stack((1 - pos, pos))
    return normalize_proba(result)


def fit_suite_point(
    preds,
    fit_y,
    validation_preds,
    validation_y,
    *,
    seed,
    point_mutation,
    budgets=(600, 1200),
):
    """Use the adapter's shallow seeds and compare native point mutation on/off.

    The stock routine scores whole native batches, truncating only the final
    batch at the last budget. Archive criteria use unrounded training Brier and
    validation Brier, with training Brier breaking validation ties. For a binary
    task these are positive-channel Brier values; for a multiclass task they
    are half the usual sum-of-squared-class-errors Brier. The ordered path
    digest includes repeated training evaluations. A checkpoint inside a batch
    can include time spent calculating that batch's later scores. The returned
    trees can be used with predict_tree on any later split of the same task.
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
    from okapi.pareto import maximize

    fit, validation = _full(preds), _full(validation_preds)
    if fit.shape[1:] != validation.shape[1:] or fit.shape[1] < 2:
        raise ValueError(
            "Need matching provider and class counts, and at least two providers"
        )
    n, m, c = fit.shape
    labels, target = _targets(fit_y, n, c)
    _, val_target = _targets(validation_y, len(validation), c)
    budgets = tuple(budgets)
    if not budgets or any(
        isinstance(b, bool) or not isinstance(b, Integral) or b < 1 for b in budgets
    ):
        raise ValueError("Provide nonempty positive integer budgets")
    if not isinstance(point_mutation, (bool, np.bool_)):
        raise TypeError("point_mutation must be boolean")
    budgets = {int(b) for b in budgets}
    maximum = max(budgets)
    bindings = _bind(validation)
    checkpoints, archive = {}, {}
    digest = hashlib.sha256()
    batch_losses = []
    calls = generation = 0

    def objective(p, _):
        loss = _loss(p, target)
        batch_losses.append(loss)
        return 1.0 - loss

    class StopBudget(Exception):
        pass

    class Observed(Okapi):
        def _calculate_fitnesses(self, trees=None):
            nonlocal calls
            cohort = self.population if trees is None else trees
            batch = cohort[: maximum - calls]
            batch_losses.clear()
            scores = super()._calculate_fitnesses(batch)
            for tree, loss in zip(batch, batch_losses, strict=True):
                calls += 1
                key = tree.structural_key()
                digest.update((repr(key) + "\n").encode("utf-8"))
                digest.update(struct.pack("<d", loss))
                if key not in archive:
                    ref = tree.copy()
                    ref.mutation_chance = tree.mutation_chance
                    p = ref.do_pred_on_another_tensors(current_tensors=bindings)
                    archive[key] = loss, _loss(p, val_target), ref
                if calls in budgets:
                    best = min(archive.values(), key=lambda r: r[0])
                    chosen = min(archive.values(), key=lambda r: (r[1], r[0]))
                    checkpoints[calls] = SuiteCheckpoint(
                        best[2],
                        chosen[2],
                        calls,
                        len(archive),
                        generation,
                        digest.hexdigest(),
                        perf_counter() - start,
                        best[0],
                        chosen[1],
                    )
            if calls == maximum:
                raise StopBudget
            return scores

    with _task_postprocessing("binary" if c == 2 else "multiclass"):
        start = perf_counter()
        adapter = OKAPIMethod(seed=seed, fitness="brier")
        try:
            directory = adapter._write_split(fit, "fit")
            gt = Path(adapter._tmp.name) / "gt.npy"
            np.save(gt, labels)
            pop = max(2, min(adapter.population_size, m))
            engine = Observed(
                directory,
                gt,
                pop,
                adapter.population_multiplier,
                max(2, min(adapter.tournament_size, pop)),
                backend="numpy",
                seed=seed,
                objective_functions=(objective,),
                objectives=(maximize,),
                minimize_node_count=adapter.minimize_node_count,
                postprocessing_function=None,
                allowed_ops=(
                    MEAN,
                    MIN,
                    MAX,
                    WEIGHTED_MEAN,
                    FAR_THRESHOLD,
                    CLOSE_THRESHOLD,
                ),
                operator_point_mutation=bool(point_mutation),
                reuse_survivor_fitness=False,
            )
            adapter._seed_population(engine, np.random.default_rng(seed))
            try:
                while True:
                    generation += 1
                    engine.run_iteration()
            except StopBudget:
                pass
        finally:
            adapter._tmp.cleanup()
    return checkpoints

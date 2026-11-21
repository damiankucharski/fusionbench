"""Generation-boundary, archive-only and population-insertion controls for native GP."""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Integral
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np

from fusionbench.eval.search_continuous import brier

if TYPE_CHECKING:
    from okapi.tree import Tree


@dataclass(frozen=True)
class ProposalPlan:
    tree: Tree
    fit_brier: float
    offsets: Iterable[int]


@dataclass(frozen=True)
class Checkpoint:
    calls: int
    generation: int
    fit_tree: Tree
    validation_tree: Tree
    fit_brier: float
    validation_brier: float
    n_trees: int
    elapsed_seconds: float
    path_digest: str


@dataclass(frozen=True)
class Probe:
    base: Checkpoint
    plain: dict[int, Checkpoint]
    sham: Checkpoint
    plan: ProposalPlan
    boundary_digest: str
    followup_calls: int


@dataclass(frozen=True)
class Insertion:
    checkpoint: Checkpoint
    survived: bool
    inserted: bool
    status: str


def _count(value, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"Expected an integer call count >= {minimum}")
    return int(value)


def _copy(tree):
    # Native Tree.copy reconstructs the prediction program, resetting mutation chance.
    copy = tree.copy()
    copy.mutation_chance = tree.mutation_chance
    return copy


class _LastCall(Exception):
    pass


class _Run:
    def __init__(self, paths, gt, validation_tensors, validation_y, seed):
        from okapi import Okapi
        from okapi.operators import (
            CLOSE_THRESHOLD,
            FAR_THRESHOLD,
            MAX,
            MEAN,
            MIN,
            WEIGHTED_MEAN,
        )

        self.calls = self.generation = 0
        self.archive = {}
        self.validation = validation_tensors
        self.validation_y = np.asarray(validation_y).ravel()
        self.digest = hashlib.sha256()
        self.deadlines = set()
        self.checkpoints = {}
        self.paused = 0.0
        self.sham_archive = self.sham_at = self.sham = None

        def objective(p, y):
            self.last_loss = brier(p.ravel(), y.ravel())
            return 1 - self.last_loss

        class Observed(Okapi):
            def _calculate_fitnesses(engine, trees=None):
                scores = []
                for tree in engine.population if trees is None else trees:
                    score = super()._calculate_fitnesses([tree])[0]
                    self.observe(tree)
                    scores.append(score)
                return np.stack(scores)

        self.start = perf_counter()
        self.gp = Observed(
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

    def entry(self, tree, fit_brier):
        ref = _copy(tree)
        pred = ref.do_pred_on_another_tensors(current_tensors=self.validation).ravel()
        return fit_brier, brier(pred, self.validation_y), ref

    def observe(self, tree):
        self.calls += 1
        key = tree.structural_key()
        # Ordered exact program, raw Brier, and repeated parent evaluations are fingerprinted.
        self.digest.update((repr(key) + "\n").encode())
        self.digest.update(struct.pack("<d", self.last_loss))
        if key not in self.archive:
            self.archive[key] = self.entry(tree, self.last_loss)
        if self.sham_archive is not None:
            self.sham_archive.setdefault(key, self.archive[key])
        if self.calls in self.deadlines:
            self.capture()

    def snapshot(self, archive):
        fit = min(archive.values(), key=lambda e: e[0])
        val = min(archive.values(), key=lambda e: (e[1], e[0]))
        return Checkpoint(
            self.calls,
            self.generation,
            _copy(fit[2]),
            _copy(val[2]),
            fit[0],
            val[1],
            len(archive),
            perf_counter() - self.start - self.paused,
            self.digest.hexdigest(),
        )

    def capture(self):
        self.checkpoints[self.calls] = self.snapshot(self.archive)
        if self.sham_at == self.calls:
            self.sham = self.snapshot(self.sham_archive)
        self.deadlines.remove(self.calls)
        if not self.deadlines:
            raise _LastCall

    def generation_step(self):
        self.generation += 1
        self.gp.run_iteration()

    def boundary(self, minimum):
        while self.calls < minimum:
            self.generation_step()
        return self.snapshot(self.archive)

    def boundary_digest(self):
        digest = hashlib.sha256()
        for tree in self.gp.population:
            state = tree.structural_key(), tree.mutation_chance
            digest.update((repr(state) + "\n").encode())
        digest.update(np.asarray(self.gp.fitnesses, dtype="<f8").tobytes())
        name, keys, at, gaussian, cached = self.gp.rng.get_state()
        digest.update(name.encode())
        digest.update(np.asarray(keys, dtype="<u4").tobytes())
        digest.update(struct.pack("<iid", at, gaussian, cached))
        return digest.hexdigest()

    def prepare(self, plan):
        """Bind the proposal to this run without a training forward or objective call."""
        loss = float(plan.fit_brier)
        if not np.isfinite(loss) or loss < 0:
            raise ValueError(
                "The proposal must carry a finite nonnegative training Brier"
            )
        tree = _copy(plan.tree)
        for node in tree.value_nodes:
            node.value = self.gp.train_tensors[node.id]
        key = tree.structural_key()
        if key in self.archive:
            known = self.archive[key]
            if abs(loss - known[0]) > 1e-14:
                raise ValueError("Proposed Brier disagrees with the native archive")
            entry = known
        else:
            entry = self.entry(tree, loss)
        return tree, key, entry

    def continue_to(self, calls):
        self.deadlines = set(calls)
        if min(self.deadlines) < self.calls:
            raise ValueError(
                "A continuation checkpoint precedes the generation boundary"
            )
        try:
            if self.calls in self.deadlines:
                self.capture()
            while self.deadlines:
                self.generation_step()
        except _LastCall:
            pass


def probe_and_continue(
    paths,
    gt,
    validation_tensors,
    validation_y,
    seed,
    callback,
    *,
    initial_calls=1200,
    followup_calls=300,
):
    """Fit once, obtain one proposal, and track both plain and archive-only continuations.

    The callback receives detached fit and validation champions plus the boundary
    checkpoint, and returns a ProposalPlan carrying its already measured training
    Brier and all desired plain continuation offsets. C counts actual native fit
    evaluations, including repeated parents. Validation scoring and the callback
    do not consume native training calls. Elapsed time excludes the callback.
    Ties preserve first encounter, with fit Brier breaking validation ties.
    """
    initial_calls, followup_calls = _count(initial_calls, 1), _count(followup_calls)
    run = _Run(paths, gt, validation_tensors, validation_y, seed)
    base = run.boundary(initial_calls)
    boundary = run.boundary_digest()
    begin = perf_counter()
    supplied = callback(_copy(base.fit_tree), _copy(base.validation_tree), base)
    plan = ProposalPlan(
        _copy(supplied.tree),
        float(supplied.fit_brier),
        tuple(_count(offset) for offset in supplied.offsets),
    )
    run.paused += perf_counter() - begin
    _, key, entry = run.prepare(plan)
    run.sham_archive = dict(run.archive)
    run.sham_archive.setdefault(key, entry)
    run.sham_at = base.calls + followup_calls
    target = {base.calls + offset for offset in (0, followup_calls, *plan.offsets)}
    run.continue_to(target)
    return Probe(base, run.checkpoints, run.sham, plan, boundary, followup_calls)


def _offer(gp, proposed, base_key, fit_brier):
    from okapi.population import choose_pareto_then_proximity
    from okapi.random_state import use_rng

    key = proposed.structural_key()
    present = any(tree.structural_key() == key for tree in gp.population)
    if key == base_key or present:
        return present, False, "unchanged" if key == base_key else "duplicate_parent"
    # The builder already paid for a full native training score. Parent scores
    # are from the completed generation; native survival uses three decimals.
    joined = gp.population + [proposed]
    scores = np.concatenate((gp.fitnesses, np.array([[1 - fit_brier]]).round(3)))
    rng = gp.rng.get_state()
    with use_rng(gp.rng):
        survivors, values = choose_pareto_then_proximity(
            joined, scores, gp.objectives, len(gp.population), gp.minimize_node_count
        )
    if any(tree is proposed for tree in survivors):
        gp.population, gp.fitnesses = survivors, values
        return True, True, "admitted"
    # No original parent may be displaced when the fresh proposal is rejected.
    assert {id(tree) for tree in survivors} == {id(tree) for tree in gp.population}
    gp.rng.set_state(rng)
    return False, False, "rejected"


def replay_insertion(
    paths,
    gt,
    validation_tensors,
    validation_y,
    seed,
    probe,
    *,
    followup_calls=None,
):
    """Recreate the untouched boundary, offer the known proposal once and resume GP.

    Both insertion and sham consider the submitted parameterized tree for
    archive selection immediately. Only an admitted fresh proposal changes
    the breeding population or parent order. Fit calls count native objective
    evaluations; the supplied score is reused at the insertion.
    """
    count = probe.followup_calls if followup_calls is None else _count(followup_calls)
    run = _Run(paths, gt, validation_tensors, validation_y, seed)
    base = run.boundary(probe.base.calls)
    if (
        base.calls != probe.base.calls
        or base.path_digest != probe.base.path_digest
        or run.boundary_digest() != probe.boundary_digest
    ):
        raise ValueError("Native GP replay did not reproduce the proposal boundary")
    tree, key, entry = run.prepare(probe.plan)
    run.archive.setdefault(key, entry)
    survived, inserted, status = _offer(
        run.gp, tree, base.fit_tree.structural_key(), entry[0]
    )
    target = base.calls + count
    run.continue_to([target])
    return Insertion(run.checkpoints[target], survived, inserted, status)

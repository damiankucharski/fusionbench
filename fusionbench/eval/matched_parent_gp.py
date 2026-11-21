"""Controlled native GP continuation from a common admission and breeding slot."""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np

from fusionbench.eval.gate_proposal_gp import (
    Checkpoint,
    ProposalPlan,
    _copy,
    _count,
    _offer,
    _Run,
)

if TYPE_CHECKING:
    from okapi.tree import Tree


OFFERS = ("fit_pair_fit_cut", "fit_pair_half", "duplicate_fit_cut", "duplicate_half")
ARMS = (*OFFERS, "reorder", "archive_only")


@dataclass(frozen=True)
class MatchedPlan:
    offers: Mapping[str, tuple[Tree, float]]
    plain_offsets: Iterable[int]
    mark: Callable[[Tree, str], Tree] | None = None
    tracked: Callable[[Tree], tuple[int, int]] | None = None
    occurrences: Callable[[Tree], Iterable] | None = None


@dataclass(frozen=True)
class Admission:
    status: str
    admitted: bool
    slot: int | None
    displaced: int | None
    parent_order: tuple[int, ...]
    marginal: dict[str, str]
    rng_before: str
    rng_after: str


@dataclass(frozen=True)
class ArmState:
    rng_before: str
    rng_after: str
    rng_final: str
    population_digest: str
    slot_fitness: float | None
    duplicate_structures: int
    parent_keys: tuple[str, ...]
    parent_fitness: tuple[float, ...]
    parent_mutation_chances: tuple[float, ...]


@dataclass(frozen=True)
class Lineage:
    marked_evaluations: int
    marked_unique: int
    marked_occurrences: int
    winner_ever_marked: bool
    winner_first_marked: bool
    winner_ever_exact: bool
    winner_first_exact: bool
    winner_ever_paths: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class MatchedResult:
    base: Checkpoint
    plain: dict[int, Checkpoint]
    arms: dict[str, dict[int, Checkpoint]]
    admission: Admission
    states: dict[str, ArmState]
    lineage: dict[str, Lineage]


def _rng_digest(state):
    name, keys, at, gaussian, cached = state
    h = hashlib.sha256(name.encode())
    h.update(np.asarray(keys, dtype="<u4").tobytes())
    h.update(struct.pack("<iid", at, gaussian, cached))
    return h.hexdigest()


def _population_digest(gp):
    digest = hashlib.sha256()
    for tree in gp.population:
        indexes = {node: i for i, node in enumerate(tree.root.get_nodes())}
        node_order = tuple(
            tuple(indexes[node] for node in tree.nodes[kind])
            for kind in ("value_nodes", "op_nodes")
        )
        descriptor = tree.structural_key(), tree.mutation_chance, node_order
        digest.update((repr(descriptor) + "\n").encode())
    digest.update(np.asarray(gp.fitnesses, dtype="<f8").tobytes())
    return digest.hexdigest()


def _preserve_indexes(tree, clone):
    copies = dict(zip(tree.root.get_nodes(), clone.root.get_nodes(), strict=True))
    for name, nodes in tree.nodes.items():
        clone.nodes[name] = [copies[node] for node in nodes]
    return clone


def _parent_copy(tree):
    """Preserve the native node-list order used by indexed crossover and mutation."""
    return _preserve_indexes(tree, _copy(tree))


class _MatchedRun(_Run):
    tracked = None

    def reset_lineage(self, tracked, occurrences):
        self.tracked = tracked
        self.occurrences = occurrences
        self.first_marked_paths = {}
        self.marked_evaluations = self.marked_occurrences = 0
        self.marked_keys = set()
        self.exact_keys = set()

    def observe(self, tree):
        if self.tracked is not None:
            count, exact = self.tracked(tree)
            if count:
                key = tree.structural_key()
                self.marked_evaluations += 1
                self.marked_occurrences += count
                if self.occurrences is not None and key not in self.marked_keys:
                    self.first_marked_paths[key] = tuple(
                        tuple(record.path) for record in self.occurrences(tree)
                    )
                self.marked_keys.add(key)
                if exact:
                    self.exact_keys.add(key)
        super().observe(tree)

    def lineage(self, fit_tree):
        key = fit_tree.structural_key()
        first = self.archive[key][2]
        marked, exact = self.tracked(first) if self.tracked is not None else (0, 0)
        return Lineage(
            self.marked_evaluations,
            len(self.marked_keys),
            self.marked_occurrences,
            key in self.marked_keys,
            bool(marked),
            key in self.exact_keys,
            bool(exact),
            self.first_marked_paths.get(key, ()),
        )


class _Boundary:
    """A complete-generation rewind point for this fixed native engine."""

    def __init__(self, run, base):
        gp = run.gp
        if gp.additional_population:
            raise ValueError("A parent intervention must start after full survival")
        self.base = base
        self.parents = [_parent_copy(t) for t in gp.population]
        self.scores = gp.fitnesses.copy()
        self.rng = gp.rng.get_state()
        self.digest = run.digest.copy()
        self.boundary_digest = run.boundary_digest()

    def restore(self, run, archive, tracked=None, occurrences=None):
        # Tree.copy reconstructs independent nodes; provider tensors are immutable.
        # Each archive entry is private to the original archive and read-only.
        gp = run.gp
        gp.population = [_parent_copy(t) for t in self.parents]
        gp.fitnesses = self.scores.copy()
        gp.rng.set_state(self.rng)
        gp.additional_population = []
        run.calls, run.generation = self.base.calls, self.base.generation
        run.digest = self.digest.copy()
        run.archive = dict(archive)
        run.checkpoints, run.deadlines = {}, set()
        run.sham_archive = run.sham_at = run.sham = None
        run.start = perf_counter() - self.base.elapsed_seconds
        run.paused = 0.0
        run.reset_lineage(tracked, occurrences)
        assert run.boundary_digest() == self.boundary_digest


def _admissions(run, boundary, archive, prepared):
    """Replay each offer's ordinary gatekeeping independently, including its RNG."""
    marginal = {}
    fitted_order = tuple(range(len(boundary.parents)))
    slot = displaced = None
    post_rng = boundary.rng
    base_key = boundary.base.fit_tree.structural_key()
    for name in OFFERS:
        boundary.restore(run, archive)
        gp = run.gp
        parents = {id(t): i for i, t in enumerate(gp.population)}
        tree, _, entry = prepared[name]
        _, admitted, status = _offer(gp, tree, base_key, entry[0])
        marginal[name] = status
        if name == OFFERS[0] and admitted:
            # Match by individual identity, preserving the selected native order.
            fitted_order = tuple(
                -1 if parent is tree else parents[id(parent)]
                for parent in gp.population
            )
            [slot] = [i for i, source in enumerate(fitted_order) if source == -1]
            [displaced] = set(parents.values()).difference(fitted_order)
            post_rng = gp.rng.get_state()

    status = marginal[OFFERS[0]]
    admission = Admission(
        status,
        status == "admitted",
        slot,
        displaced,
        fitted_order,
        marginal,
        _rng_digest(boundary.rng),
        _rng_digest(post_rng),
    )
    return admission, post_rng


def _install(run, arm, admission, post_rng, prepared, mark):
    if not admission.admitted or arm == "archive_only":
        return None
    gp = run.gp
    parents = list(gp.population)
    scores = gp.fitnesses.copy()
    if arm == "reorder":
        tree = parents[admission.displaced]
        row = scores[admission.displaced]
    else:
        source, key, entry = prepared[arm]
        tree = _parent_copy(source)
        if mark is not None:
            marked = mark(tree, arm)
            tree = _preserve_indexes(tree, marked)
            if (
                tree.structural_key() != key
                or tree.mutation_chance != source.mutation_chance
            ):
                raise ValueError(
                    "Lineage marking must preserve the native program and mutation chance"
                )
        row = np.array([1 - entry[0]]).round(3)

    gp.population = [tree if i == -1 else parents[i] for i in admission.parent_order]
    gp.fitnesses = np.stack(
        [row if i == -1 else scores[i] for i in admission.parent_order]
    )
    gp.rng.set_state(post_rng)
    return float(row[0])


def matched_parent_gp(
    paths,
    gt,
    val_tensors,
    val_y,
    seed,
    callback,
    *,
    initial_calls=1200,
    followup_calls=300,
):
    """Run stock GP once to a completed generation, then compare six tail treatments.

    The callback takes copies of the fit/validation champions and the initial
    Checkpoint. It returns four (bound native tree, measured training Brier)
    offers and extra budgets for plain GP. All treated archives receive all four
    offers, in OFFERS order, before the first further evaluation. A variant
    matching a retained parent remains in its matched slot until ordinary native
    duplicate handling on the next generation. Validation never participates in
    admission. Extra fit-label work belongs to the proposal callback.

    Marking optionally returns a private annotated copy of a new slot parent.
    The tracking hook returns (total marks, marks retaining the original fitted
    direction and threshold). Neither can alter the prediction program. Reported
    lineage is over ordinary objective evaluations strictly after the boundary.
    Arm states contain before/after intervention RNG, before-tail ordered parent
    programs/fitness digest, and final RNG. An arm's checkpoints are keyed by
    total true objective calls.
    """
    initial_calls, followup_calls = _count(initial_calls, 1), _count(followup_calls)
    run = _MatchedRun(paths, gt, val_tensors, val_y, seed)
    base = run.boundary(initial_calls)
    boundary = _Boundary(run, base)
    original = dict(run.archive)
    plan = callback(_copy(base.fit_tree), _copy(base.validation_tree), base)
    if set(plan.offers) != set(OFFERS):
        raise ValueError(f"Expected exactly the four named offers {OFFERS}")
    if (plan.mark is None) != (plan.tracked is None):
        raise ValueError("Provide both lineage hooks or omit both")
    plain_targets = {base.calls + _count(offset) for offset in plan.plain_offsets}
    target = base.calls + followup_calls
    plain_targets.update((base.calls, target))

    shared = dict(original)
    prepared = {}
    for name in OFFERS:
        tree, loss = plan.offers[name]
        item = run.prepare(ProposalPlan(tree, loss, ()))
        bound, key, entry = item
        _preserve_indexes(tree, bound)
        prepared[name] = item
        known = shared.setdefault(key, entry)
        if abs(known[0] - entry[0]) > 1e-14:
            raise ValueError(
                "Duplicate offers disagree on their measured training Brier"
            )
    admission, post_rng = _admissions(run, boundary, original, prepared)

    boundary.restore(run, original)
    run.continue_to(plain_targets)
    plain = run.checkpoints
    results, states, lineage = {}, {}, {}
    for arm in ARMS:
        boundary.restore(run, shared, plan.tracked, plan.occurrences)
        slot_score = _install(run, arm, admission, post_rng, prepared, plan.mark)
        rng_after = _rng_digest(run.gp.rng.get_state())
        population_digest = _population_digest(run.gp)
        parent_keys = tuple(
            hashlib.sha256(repr(tree.structural_key()).encode()).hexdigest()
            for tree in run.gp.population
        )
        parent_fitness = tuple(map(float, run.gp.fitnesses[:, 0]))
        mutation_chances = tuple(
            float(tree.mutation_chance) for tree in run.gp.population
        )
        duplicates = len(run.gp.population) - len(
            {t.structural_key() for t in run.gp.population}
        )
        run.continue_to({base.calls, target})
        results[arm] = run.checkpoints
        states[arm] = ArmState(
            admission.rng_before,
            rng_after,
            _rng_digest(run.gp.rng.get_state()),
            population_digest,
            slot_score,
            duplicates,
            parent_keys,
            parent_fitness,
            mutation_chances,
        )
        lineage[arm] = run.lineage(run.checkpoints[target].fit_tree)
    return MatchedResult(base, plain, results, admission, states, lineage)

"""Fixed-budget native OKAPI tree search with separately replaceable deployment incumbent."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path

import numpy as np

from fusionbench.eval.metrics import normalize_proba
from fusionbench.eval.suite_point_gp import _full, _targets, predict_tree
from fusionbench.methods.okapi import OKAPIMethod, _task_postprocessing

GENERATIONS = 6
PROPOSALS = 12
CAPACITY = 8
POLICIES = ("ordinary", "open_raw", "open_kl", "strict_kl")


@dataclass
class Candidate:
    birth_id: int
    tree: object
    generation: int
    parent_ids: tuple[int, ...]
    discovery_brier: float
    provider_ids: tuple[int, ...]
    genotype: str
    confirmed: bool | None
    has_unconfirmed_ancestor: bool


@dataclass(frozen=True)
class Generation:
    generation: int
    proposals: tuple[int, ...]
    eligible: tuple[int, ...]
    survivors: tuple[int, ...]
    incumbent_before: int
    incumbent_after: int
    promoted: int | None
    gains: tuple[float, ...]
    lower_bounds: tuple[float, ...]
    gate_comparisons: int
    gate_rows: int
    path_digest: str


@dataclass(frozen=True)
class SearchResult:
    records: tuple[Candidate, ...]
    generations: tuple[Generation, ...]
    initial_survivors: tuple[int, ...]
    initial_incumbent_id: int
    champion_id: int
    incumbent_scan: int
    discovery_evaluations: int
    proposal_count: int
    path_digest: str

    @property
    def tree(self):
        return self.records[self.champion_id].tree

    @property
    def unique_providers(self):
        return len(self.records[self.champion_id].provider_ids)

    @property
    def displaced(self):
        return (
            self.records[self.initial_incumbent_id].genotype
            != self.records[self.champion_id].genotype
        )


def _split(preds, labels):
    x = _full(preds)
    y, _ = _targets(labels, len(x), x.shape[2])
    if not np.all((x >= 0) & (x <= 1)) or not np.allclose(
        x.sum(axis=2), 1, rtol=0, atol=1e-6
    ):
        raise ValueError(
            "Supply finite class probabilities summing to one for every provider and row"
        )
    return x, y


def _risk(probability, targets):
    return float(np.square(probability - targets).sum(axis=1).mean())


def _make_record(tree, records, targets, generation=0, parents=(), known_loss=None):
    if known_loss is None:
        p = np.asarray(tree.predict(clear_cache=True))
        if p.shape[1] == 1:
            p = np.column_stack((1 - p[:, 0], p[:, 0]))
        else:
            p = normalize_proba(p)
        known_loss = _risk(p, targets)
    ids = tuple(sorted({int(v.id[1:-4]) for v in tree.nodes["value_nodes"]}))
    genotype = hashlib.sha256(repr(tree.structural_key()).encode()).hexdigest()
    # Native operators report selected crossover parents. These edges certify
    # reproductive opportunity, not that both donated material to the product.
    ancestry = any(
        records[i].confirmed is False or records[i].has_unconfirmed_ancestor
        for i in parents
    )
    birth_id = len(records)
    records.append(
        Candidate(
            birth_id,
            tree,
            generation,
            parents,
            known_loss,
            ids,
            genotype,
            None,
            ancestry,
        )
    )
    return birth_id


def _digest_update(digest, generation, ids, records, *, parents=False):
    digest.update(f"g={generation}:".encode())
    for i in ids:
        r = records[i]
        digest.update(f"{i},{r.genotype}".encode())
        if parents:
            digest.update(repr(r.parent_ids).encode())
        digest.update(b";")
    digest.update(b"\n")


def _fitness(ids, records):
    # The native library works on unit-scale predictive fitness. Its historical
    # three-place rounding applies to parent tournaments and survivor selection.
    return np.round([[1 - records[i].discovery_brier / 2] for i in ids], 3)


def _survive(gp, pool, records):
    from okapi.population import choose_pareto_then_proximity

    unique = {}
    for i in pool:
        unique.setdefault(records[i].tree.structural_key(), i)
    ids = list(unique.values())
    by_object = {id(records[i].tree): i for i in ids}
    gp.population, gp.fitnesses = choose_pareto_then_proximity(
        [records[i].tree for i in ids],
        _fitness(ids, records),
        gp.objectives,
        min(CAPACITY, len(ids)),
        gp.minimize_node_count,
    )
    return tuple(by_object[id(t)] for t in gp.population)


def _initialize(gp, records, x, targets, incumbent):
    from okapi.node import ValueNode
    from okapi.paired_fitness import arithmetic_mean_tree
    from okapi.tree import Tree

    scan = 0
    known_loss = None
    if incumbent == "mean":
        tree = arithmetic_mean_tree(gp.train_tensors)
        index = None
    else:
        if incumbent == "best_single":
            scan = x.shape[1]
            losses = np.square(x - targets[:, None, :]).sum(axis=2).mean(axis=0)
            index = int(np.argmin(losses))
            known_loss = float(losses[index])
        elif (
            isinstance(incumbent, Integral)
            and not isinstance(incumbent, bool)
            and 0 <= incumbent < x.shape[1]
        ):
            index = int(incumbent)
        else:
            raise ValueError(
                "incumbent must be 'best_single', 'mean', or a valid provider index"
            )
        name = f"m{index}.npy"
        tree = Tree(ValueNode(None, gp.train_tensors[name], name))
    # The controlled mean incumbent is seeded explicitly. A best-single scan
    # across a large pool never silently injects its winner into breeding.
    if index is None and not any(
        t.structural_key() == tree.structural_key() for t in gp.population
    ):
        if len(gp.population) == CAPACITY:
            gp.population[-1] = tree
        else:
            gp.population.append(tree)
    active = tuple(_make_record(t, records, targets) for t in gp.population)
    existing = next(
        (
            i
            for i in active
            if records[i].tree.structural_key() == tree.structural_key()
        ),
        None,
    )
    if existing is None:
        existing = _make_record(tree, records, targets, known_loss=known_loss)
    # Any separate external known provider needs just one discovery evaluation.
    extra = int(existing not in active and known_loss is None)
    gp.fitnesses = _fitness(active, records)
    return active, existing, scan, len(active) + extra


def run_replacement_search(
    fit_preds,
    fit_y,
    gate_blocks=None,
    *,
    seed,
    policy="open_raw",
    incumbent="best_single",
):
    """Run six generations of twelve native offspring with at most eight breeders.

    gate_blocks holds six disjoint (predictions, labels) pairs in revelation
    order. Only ordinary discovery selection can omit them. Open policies
    share a breeding trajectory. Every slot pays its discovery and confirmation
    costs, including duplicates. The separately counted initial best-single
    scan examines all provider columns. Archived trace objects are diagnostic
    records and occupy no breeding or deployment slots.

    Candidate.confirmed marks positive gate score at birth vs the *preceding*
    incumbent, including diagnostics in ordinary if gate blocks were supplied.
    An ungated ordinary run has None for all such statuses.
    """
    from okapi import Okapi
    from okapi.replacement import select_brier_replacement

    if policy not in POLICIES:
        raise ValueError(f"Unknown replacement policy {policy!r}")
    if gate_blocks is None:
        if policy != "ordinary":
            raise ValueError(
                "Six fresh confirmation blocks are required for a replacement policy"
            )
        blocks = [None] * GENERATIONS
    else:
        blocks = list(gate_blocks)
        if len(blocks) != GENERATIONS:
            raise ValueError(f"Expected {GENERATIONS} independent confirmation blocks")
        if any(block is None for block in blocks):
            raise ValueError(
                "A supplied confirmation sequence must contain six complete blocks"
            )
    x, y = _split(fit_preds, fit_y)
    _, k, c = x.shape
    targets = np.eye(c)[y]
    records = []
    snapshots = []
    digest = hashlib.sha256()
    task = "binary" if c == 2 else "multiclass"
    with _task_postprocessing(task):
        adapter = OKAPIMethod(seed=seed, fitness="brier")
        try:
            directory = adapter._write_split(x, "fit")
            labels = Path(adapter._tmp.name) / "labels.npy"
            np.save(labels, y)
            gp = Okapi(
                directory,
                labels,
                min(k, CAPACITY),
                2,
                min(3, k),
                seed=seed,
                backend="numpy",
                minimize_node_count=True,
            )
            active, chosen, scan, initial_cost = _initialize(
                gp, records, x, targets, incumbent
            )
            initial_incumbent, initial_survivors = chosen, active
            _digest_update(digest, 0, active, records)
            for g, block in enumerate(blocks, start=1):
                previous = chosen
                # No current block has been read when this complete proposal set
                # and the previous incumbent have both become immutable.
                children = gp.propose_offspring(PROPOSALS)
                new = tuple(
                    _make_record(
                        tree, records, targets, g, tuple(active[j] for j in parents)
                    )
                    for tree, parents in children
                )
                assert len(new) == PROPOSALS
                _digest_update(digest, g, new, records, parents=True)
                decision = None
                eligible = ()
                gate_rows = 0
                if block is not None:
                    gx, gy = _split(*block)
                    if gx.shape[1:] != x.shape[1:]:
                        raise ValueError(
                            "Gate and discovery provider/class axes must match"
                        )
                    gate_rows = len(gy)
                    gate_predictions = [predict_tree(records[i].tree, gx) for i in new]
                    decision = select_brier_replacement(
                        predict_tree(records[previous].tree, gx),
                        gate_predictions,
                        gy,
                        alpha=0.05 / GENERATIONS,
                        rule="kl" if policy.endswith("kl") else "raw",
                    )
                    eligible = tuple(new[j] for j in decision.eligible)
                    for i in new:
                        records[i].confirmed = i in eligible
                breeders = eligible if policy == "strict_kl" else new
                active = _survive(gp, (*active, *breeders), records)
                _digest_update(digest, g, active, records)
                if policy == "ordinary":
                    chosen = min(
                        active,
                        key=lambda i: (
                            records[i].discovery_brier,
                            records[i].tree.nodes_count,
                        ),
                    )
                elif decision.index is not None:
                    chosen = new[decision.index]
                changed = records[chosen].genotype != records[previous].genotype
                snapshots.append(
                    Generation(
                        g,
                        new,
                        eligible,
                        active,
                        previous,
                        chosen,
                        chosen if changed else None,
                        () if decision is None else decision.gains,
                        () if decision is None else decision.lower_bounds,
                        0 if decision is None else len(new),
                        gate_rows,
                        digest.hexdigest(),
                    )
                )
        finally:
            adapter._tmp.cleanup()
    return SearchResult(
        tuple(records),
        tuple(snapshots),
        initial_survivors,
        initial_incumbent,
        chosen,
        scan,
        initial_cost + GENERATIONS * PROPOSALS,
        GENERATIONS * PROPOSALS,
        digest.hexdigest(),
    )

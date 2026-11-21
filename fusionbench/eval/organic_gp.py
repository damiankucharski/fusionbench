"""Observe a randomized edit of one organic offspring before native deduplication."""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from numbers import Integral
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np

from fusionbench.eval.search_continuous import brier

if TYPE_CHECKING:
    from okapi.tree import Tree


ARMS = ("stock", "type_only", "provider_only", "both")


@dataclass(frozen=True)
class Edit:
    tree: Tree
    proposed_type: str
    proposed_provider: object


@dataclass(frozen=True)
class OrganicCheckpoint:
    calls: int
    generation: int
    fit_tree: Tree
    validation_tree: Tree
    fit_brier: float
    validation_brier: float
    n_trees: int
    path_digest: str
    elapsed_seconds: float
    ledger: dict


@dataclass(frozen=True)
class OrganicResult:
    checkpoints: dict[str, OrganicCheckpoint]
    events: list[dict]


def _integer(value, minimum):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"Expected an integer >= {minimum}")
    return int(value)


def _copy(tree):
    candidate = tree.copy()
    candidate.mutation_chance = tree.mutation_chance
    return candidate


def _at(root, path):
    for i in path:
        root = root.children[i]
    return root


def _sha(key):
    return hashlib.sha256(repr(key).encode("utf-8")).hexdigest()


def _ledger(events, completed, fit_calls, validation_calls, edit_seconds):
    eligible = [e for e in events if e["offspring_index"] is not None]
    return {
        "generations_completed": completed,
        "opportunities": len(eligible),
        "no_eligible": len(events) - len(eligible),
        "changed": sum(e["changed"] for e in eligible),
        "type_changed": sum(e["before_type"] != e["after_type"] for e in eligible),
        "provider_changed": sum(
            e["before_provider"] != e["after_provider"] for e in eligible
        ),
        "passed_dedup": sum(e["passed_dedup"] is True for e in eligible),
        "filtered_duplicate": sum(e["passed_dedup"] is False for e in eligible),
        "evaluated": sum(e["evaluated"] is True for e in eligible),
        "survived": sum(e["survived"] is True for e in eligible),
        "equivalent_survived": sum(e["equivalent_survived"] is True for e in eligible),
        "selection_pending": sum(e["survived"] is None for e in eligible),
        "fit_calls": fit_calls,
        "validation_calls": validation_calls,
        "edit_seconds": edit_seconds,
    }


class _PartialFinalBatch(Exception):
    pass


def organic_gp(
    paths,
    gt,
    val_tensors,
    val_y,
    seed,
    arm,
    *,
    event_seed,
    transform,
    budgets=(600, 1800),
    generation_checkpoint=100,
):
    """Run native search with one label-blind organic opportunity per generation.

    Child selection is uniform over eligible offspring (crossover and mutants).
    Within the chosen child sample a qualifying operator uniformly and then one
    direct value child uniformly. At generation g all arm-specific callbacks
    receive the remaining draws from a private RandomState seeded by SeedSequence([event_seed, g]).
    The callback must leave its input unchanged and return Edit(tree, proposed_type,
    proposed_provider). Both alternatives must be planned even for stock.

    A checkpoint counts real fit objectives including repeated parents. The
    stock fitness method evaluates complete batches. Only the final batch can be
    truncated if the generation checkpoint was already reached. When a budget
    lands inside a native fitness batch the archive describes its prefix; measured
    wall time can include the rest of that batch's computation. Survivor status is
    known only upon completion of the generation. A different existing program
    with the same key is counted separately from survival of the submitted object.
    """
    from okapi import Okapi
    from okapi.interventions import operator_paths
    from okapi.operators import (
        CLOSE_THRESHOLD,
        FAR_THRESHOLD,
        MAX,
        MEAN,
        MIN,
        WEIGHTED_MEAN,
    )

    if arm not in ARMS:
        raise ValueError(f"Unknown organic intervention arm {arm}")
    budgets = frozenset(_integer(n, 1) for n in budgets)
    if not budgets:
        raise ValueError("Provide at least one objective-call checkpoint")
    last = max(budgets)
    generation_checkpoint = _integer(generation_checkpoint, 1)
    event_seed = _integer(event_seed, 0)
    from okapi.random_state import use_rng

    val_y = np.asarray(val_y).ravel()
    evaluations, validations, generation, completed = 0, 0, 0, 0
    edit_seconds = 0.0
    events, checkpoints, archive = [], {}, {}
    path = hashlib.sha256()
    event = target = key = None
    earlier_alias = False
    raw_losses = []

    def objective(p, y):
        loss = brier(p.ravel(), y.ravel())
        raw_losses.append(loss)
        return 1.0 - loss

    def capture(label):
        fit = min(archive.values(), key=lambda row: row[0])
        val = min(archive.values(), key=lambda row: (row[1], row[0]))
        checkpoints[label] = OrganicCheckpoint(
            evaluations,
            generation,
            _copy(fit[2]),
            _copy(val[2]),
            fit[0],
            val[1],
            len(archive),
            path.hexdigest(),
            perf_counter() - start,
            _ledger(events, completed, evaluations, validations, edit_seconds),
        )

    class Observed(Okapi):
        def _mutate_additional_population(self):
            nonlocal target, key, event, earlier_alias, edit_seconds
            crossover_count = len(self.additional_population)
            mutations = super()._mutate_additional_population()
            begin = perf_counter()
            offspring = self.additional_population
            candidates = []
            for i, tree in enumerate(offspring):
                positions = [(p, op) for p, op in operator_paths(tree) if op.children]
                if positions:
                    candidates.append((i, positions))
            local_seed = int(
                np.random.SeedSequence([event_seed, generation]).generate_state(1)[0]
            )
            event = {
                "event_seed": local_seed,
                "generation": generation,
                "calls_before_join": evaluations,
                "offspring_count": len(offspring),
                "crossover_children": crossover_count,
                "eligible_children": len(candidates),
                "offspring_index": None,
                "origin": None,
                "operator_path": None,
                "child_index": None,
                "before_type": None,
                "proposed_type": None,
                "after_type": None,
                "before_provider": None,
                "proposed_provider": None,
                "after_provider": None,
                "changed": False,
                "old_sha256": None,
                "new_sha256": None,
                "passed_dedup": None,
                "evaluated": False,
                "evaluation_call": None,
                "fit_brier": None,
                "survived": None,
                "equivalent_survived": None,
                "selection_call": None,
            }
            events.append(event)
            target = key = None
            if candidates:
                rng = np.random.RandomState(local_seed)
                i, positions = candidates[rng.randint(len(candidates))]
                op_path, op = positions[rng.randint(len(positions))]
                child = int(rng.randint(len(op.children)))
                original = offspring[i]
                original_key = original.structural_key()
                before_type = type(op).__name__
                before_provider = op.children[child].id
                with use_rng(rng):
                    edit = transform(
                        original, op_path, child, rng, self.train_tensors, arm
                    )
                if original.structural_key() != original_key:
                    raise ValueError("The transform mutated its input offspring")
                target = original if arm == "stock" else edit.tree
                key = target.structural_key()
                if arm == "stock" and edit.tree.structural_key() != original_key:
                    raise ValueError("The stock transform returned a changed program")
                new_op = _at(target.root, op_path)
                new_provider = new_op.children[child].id
                new_type = type(new_op).__name__
                if new_type != (
                    edit.proposed_type if arm in ("both", "type_only") else before_type
                ):
                    raise ValueError(
                        "Submitted operator disagrees with the assigned arm"
                    )
                if new_provider != (
                    edit.proposed_provider
                    if arm in ("both", "provider_only")
                    else before_provider
                ):
                    raise ValueError(
                        "Submitted provider disagrees with the assigned arm"
                    )
                earlier_alias = any(
                    t is target for t in [*self.population, *offspring[:i]]
                )
                offspring[i] = target
                event.update(
                    offspring_index=i,
                    origin="crossover" if i < crossover_count else "mutation",
                    operator_path=op_path,
                    child_index=child,
                    before_type=before_type,
                    proposed_type=edit.proposed_type,
                    after_type=new_type,
                    before_provider=before_provider,
                    proposed_provider=edit.proposed_provider,
                    after_provider=new_provider,
                    changed=key != original_key,
                    old_sha256=_sha(original_key),
                    new_sha256=_sha(key),
                )
            edit_seconds += perf_counter() - begin
            return mutations

        def _calculate_fitnesses(self, trees=None):
            nonlocal evaluations, validations
            cohort = self.population if trees is None else trees
            if target is not None:
                event["passed_dedup"] = not earlier_alias and any(
                    t is target for t in cohort
                )
            remaining = last - evaluations
            truncate = completed >= generation_checkpoint and len(cohort) > remaining
            batch = cohort[:remaining] if truncate else cohort
            raw_losses.clear()
            # A single native call per ordinary population, including the initial singletons.
            scores = super()._calculate_fitnesses(batch)
            for tree, loss in zip(batch, raw_losses, strict=True):
                evaluations += 1
                structure = tree.structural_key()
                path.update((repr(structure) + "\n").encode())
                path.update(struct.pack("<d", loss))
                if structure not in archive:
                    ref = _copy(tree)
                    val = ref.do_pred_on_another_tensors(
                        current_tensors=val_tensors
                    ).ravel()
                    archive[structure] = loss, brier(val, val_y), ref
                    validations += 1
                if tree is target and event["passed_dedup"]:
                    event.update(
                        evaluated=True, evaluation_call=evaluations, fit_brier=loss
                    )
                if evaluations in budgets:
                    capture(f"calls_{evaluations}")
            if truncate:
                raise _PartialFinalBatch
            return scores

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
        while evaluations < last or completed < generation_checkpoint:
            generation += 1
            gp.run_iteration()
            completed += 1
            if target is not None:
                event["survived"] = event["passed_dedup"] and any(
                    t is target for t in gp.population
                )
                event["equivalent_survived"] = any(
                    t.structural_key() == key for t in gp.population
                )
                event["selection_call"] = evaluations
            if evaluations in budgets:
                # When this call completes survival, report the now-observable outcome.
                capture(f"calls_{evaluations}")
            if completed == generation_checkpoint:
                capture(f"generation_{completed}")
    except _PartialFinalBatch:
        pass
    expected = {f"calls_{budget}" for budget in budgets} | {
        f"generation_{generation_checkpoint}"
    }
    assert set(checkpoints) == expected
    return OrganicResult(checkpoints, events)

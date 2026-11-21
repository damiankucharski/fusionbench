"""Replay ordinary OKAPI and independently resample, apply and trace point edits.

Run this script against the documented OKAPI checkout with the study directory
as its argument. All comparisons are exact, except that run times are omitted.
The untreated control never overrides native mutation or plans a point edit.
"""

import argparse
import ast
import csv
import gzip
import hashlib
import json
import struct
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import okapi
from loguru import logger
from okapi import Okapi
from okapi.interventions import tree_with_operand_provider
from okapi.mutation import tree_with_operator_type
from okapi.node import OperatorNode
from okapi.operators import (
    CLOSE_THRESHOLD,
    FAR_THRESHOLD,
    MAX,
    MEAN,
    MIN,
    WEIGHTED_MEAN,
)
from okapi.random_state import use_rng

from fusionbench.eval.native_switch import serialize
from fusionbench.scenarios.compositional_switch import CompositionalSwitch

PIN = "194e94a6c15994a29cdb509bd66c3e320a955c90"
OPS = (MEAN, MIN, MAX, WEIGHTED_MEAN, FAR_THRESHOLD, CLOSE_THRESHOLD)
ARMS = ("stock", "type_only", "provider_only", "both")
LAWS = ("composition", "logit")
BUDGETS = (600, 1800)
FINAL_GENERATION = 100
CHECKPOINTS = ("calls_600", "calls_1800", "generation_100")


def equal(got, expected, name):
    if got != expected:
        raise ValueError(f"{name}: replay {got!r}, saved {expected!r}")


def check_csv(value, row, column):
    stored = row[column]
    if value is None:
        equal(stored, "", column)
    elif isinstance(value, bool):
        equal(str(value), stored, column)
    elif isinstance(value, (float, int)):
        equal(value, float(stored), column)
    elif isinstance(value, tuple):
        equal(value, tuple(ast.literal_eval(stored)), column)
    else:
        equal(value, stored, column)


def tree_sha(key):
    return hashlib.sha256(repr(key).encode()).hexdigest()


def locations(tree):
    """Enumerate operator occurrences from the root in ordered depth-first order."""
    found, stack = [], [(tree.root, ())]
    while stack:
        node, path = stack.pop()
        if isinstance(node, OperatorNode) and node.children:
            found.append((path, node))
        stack.extend(
            (node.children[i], (*path, i)) for i in reversed(range(len(node.children)))
        )
    return found


@dataclass
class Model:
    program: dict
    fit: float
    validation: float


class EndOfPrefix(Exception):
    pass


class NativeAudit(Okapi):
    """Observe only actual native score batches and both running archive champions."""

    def __init__(self, paths, labels, val_inputs, val_labels, gp_seed):
        self.val_inputs = val_inputs
        self.val_labels = val_labels
        self.structures = set()
        self.seen_calls = self.completed = 0
        self.digest = hashlib.sha256()
        self.results = {}
        self.fit_winner = self.val_winner = None
        self.event = self.individual = None
        self.events = []
        self.counts = Counter()
        super().__init__(
            paths,
            labels,
            len(paths),
            2,
            2,
            backend="numpy",
            seed=gp_seed,
            objective_functions=(self.record_loss,),
            allowed_ops=OPS,
            postprocessing_function=np.asarray,
            reuse_survivor_fitness=False,
        )

    def record_loss(self, prediction, truth):
        loss = float(np.square(prediction.ravel() - truth.ravel()).mean())
        self.losses.append(loss)
        return 1 - loss

    def checkpoint(self, name, generation):
        self.results[name] = (
            self.seen_calls,
            generation,
            len(self.structures),
            self.digest.hexdigest(),
            self.fit_winner,
            self.val_winner,
            dict(self.counts),
            self.completed,
        )

    def _calculate_fitnesses(self, trees=None):
        cohort = self.population if trees is None else trees
        if self.individual is not None:
            actual = any(t is self.individual for t in cohort)
            # When the very same object was joined at an earlier index, its
            # surviving copy belongs to that earlier index of the native cohort.
            kept = actual and not self.earlier_identity
            equal(
                kept,
                self.event["passed_dedup"],
                "native first-occurrence deduplication",
            )

        remaining = BUDGETS[-1] - self.seen_calls
        incomplete = self.completed >= FINAL_GENERATION and len(cohort) > remaining
        batch = cohort[:remaining] if incomplete else cohort
        self.losses = []
        scores = super()._calculate_fitnesses(batch)
        for tree, loss in zip(batch, self.losses, strict=True):
            self.seen_calls += 1
            key = tree.structural_key()
            self.digest.update((repr(key) + "\n").encode())
            self.digest.update(struct.pack("<d", loss))
            if key not in self.structures:
                self.structures.add(key)
                predictions = tree.do_pred_on_another_tensors(
                    current_tensors=self.val_inputs
                )
                val = float(np.square(predictions.ravel() - self.val_labels).mean())
                improved_fit = self.fit_winner is None or loss < self.fit_winner.fit
                improved_val = self.val_winner is None or (val, loss) < (
                    self.val_winner.validation,
                    self.val_winner.fit,
                )
                if improved_fit or improved_val:
                    model = Model(serialize(tree.root), loss, val)
                    if improved_fit:
                        self.fit_winner = model
                    if improved_val:
                        self.val_winner = model
            if self.individual is tree and self.event["passed_dedup"]:
                self.event.update(
                    evaluated=True, evaluation_call=self.seen_calls, fit_brier=loss
                )
                self.counts["evaluated"] += 1
            if self.seen_calls in BUDGETS:
                self.checkpoint(f"calls_{self.seen_calls}", self.completed + 1)
        if incomplete:
            raise EndOfPrefix
        return scores

    def replay(self):
        while self.seen_calls < BUDGETS[-1] or self.completed < FINAL_GENERATION:
            try:
                self.run_iteration()
            except EndOfPrefix:
                break
            self.completed += 1
            if self.individual is not None:
                event = self.event
                kept_object = event["passed_dedup"] and any(
                    t is self.individual for t in self.population
                )
                kept_key = self.individual.structural_key() in {
                    t.structural_key() for t in self.population
                }
                event.update(
                    survived=kept_object,
                    equivalent_survived=kept_key,
                    selection_call=self.seen_calls,
                )
                self.counts["survived"] += kept_object
                self.counts["equivalent_survived"] += kept_key
                self.counts["selection_pending"] -= 1
            if self.seen_calls in BUDGETS:
                self.checkpoint(f"calls_{self.seen_calls}", self.completed)
            if self.completed == FINAL_GENERATION:
                self.checkpoint(f"generation_{self.completed}", self.completed)
        equal(set(self.results), set(CHECKPOINTS), "checkpoint coverage")


class EventAudit(NativeAudit):
    def _mutate_additional_population(self):
        crossover = len(self.additional_population)
        activity = super()._mutate_additional_population()
        generation = self.completed + 1
        seed = int(
            np.random.SeedSequence([self.event_seed, generation]).generate_state(1)[0]
        )
        rng = np.random.RandomState(seed)
        population = self.additional_population
        choices = [
            (i, options)
            for i, tree in enumerate(population)
            if (options := locations(tree))
        ]
        event = {
            "generation": generation,
            "event_seed": seed,
            "calls_before_join": self.seen_calls,
            "offspring_count": len(population),
            "crossover_children": crossover,
            "eligible_children": len(choices),
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
            "old_sha256": None,
            "new_sha256": None,
            "changed": False,
            "passed_dedup": None,
            "evaluated": False,
            "evaluation_call": None,
            "fit_brier": None,
            "survived": None,
            "equivalent_survived": None,
            "selection_call": None,
        }
        self.events.append(event)
        self.event, self.individual = event, None
        if not choices:
            self.counts["no_eligible"] += 1
            return activity
        at, options = choices[rng.randint(len(choices))]
        path, old = options[rng.randint(len(options))]
        child = rng.randint(len(old.children))
        source = population[at]
        prior = source.structural_key()
        original_provider = old.children[child].id
        families = [cls for cls in OPS if cls != type(old)]
        providers = sorted(set(self.train_tensors) - {original_provider})
        family = families[rng.randint(len(families))]
        provider = providers[rng.randint(len(providers))]
        with use_rng(rng):
            alternate = tree_with_operator_type(source, path, family)
        edited = alternate if self.arm in ("type_only", "both") else source
        if self.arm in ("provider_only", "both"):
            edited = tree_with_operand_provider(
                edited, path, child, provider, self.train_tensors[provider]
            )
        equal(source.structural_key(), prior, "source tree after point edit")
        population[at] = self.individual = edited
        final = edited.structural_key()
        prefix = self.population + population[:at]
        self.earlier_identity = any(t is edited for t in prefix)
        passes = not any(t.structural_key() == final for t in prefix)
        op = dict(locations(edited))[path]
        event.update(
            offspring_index=at,
            origin="crossover" if at < crossover else "mutation",
            operator_path=path,
            child_index=child,
            before_type=type(old).__name__,
            proposed_type=family.__name__,
            after_type=type(op).__name__,
            before_provider=original_provider,
            proposed_provider=provider,
            after_provider=op.children[child].id,
            old_sha256=tree_sha(prior),
            new_sha256=tree_sha(final),
            changed=prior != final,
            passed_dedup=passes,
        )
        self.counts.update(
            opportunities=1,
            selection_pending=1,
            changed=int(prior != final),
            type_changed=int(type(old) != type(op)),
            provider_changed=int(original_provider != op.children[child].id),
            passed_dedup=int(passes),
            filtered_duplicate=int(not passes),
        )
        return activity


def verify_snapshots(engine, case, winners, ledgers, detailed):
    counts = Counter()
    for checkpoint, result in engine.results.items():
        calls, generation, unique, digest, fit, validation, tally, completed = result
        row = ledgers[checkpoint]
        for column, value in (
            ("calls", calls),
            ("generation", generation),
            ("fit_calls", calls),
            ("validation_calls", unique),
            ("generations_completed", completed),
        ):
            check_csv(value, row, column)
        if detailed:
            for field in (
                "opportunities",
                "no_eligible",
                "changed",
                "type_changed",
                "provider_changed",
                "passed_dedup",
                "filtered_duplicate",
                "evaluated",
                "survived",
                "equivalent_survived",
                "selection_pending",
            ):
                check_csv(tally.get(field, 0), row, field)
            counts["verified_cost_ledgers"] += 1
        for decision, model in (("fit", fit), ("validation", validation)):
            for split, value in (("fit", model.fit), ("validation", model.validation)):
                row = winners[checkpoint, decision, split]
                check_csv(value, row, "observed_brier")
                check_csv(calls, row, "calls")
                check_csv(unique, row, "archive_trees")
                equal(
                    model.program,
                    json.loads(row["program"]),
                    f"{case}/{checkpoint}/{decision} model",
                )
                equal(
                    digest, row["path_digest"], f"{case}/{checkpoint}/{decision} path"
                )
            counts["verified_winners"] += 1
        counts["verified_checkpoints"] += 1
    counts["native_evaluations"] += engine.seen_calls
    return counts


def verify_group(task):
    try:
        return replay_group(task)
    except Exception as exc:
        exc.add_note(f"Case (bank, law, GP seed, arm): {task[0]!r}")
        raise


def replay_group(task):
    case, data = task
    seed, law, gp_seed, arm = case
    winners, ledgers, events = data
    scenario = CompositionalSwitch(role_seed=seed)
    fit = scenario.draw(512, np.random.default_rng([seed, 0]), 0.5)
    val = scenario.draw(256, np.random.default_rng([seed, 1]), 0.5)
    totals = Counter()
    with TemporaryDirectory() as temp:
        folder = Path(temp)
        paths = [folder / f"p{i}.npy" for i in range(6)]
        for i, path in enumerate(paths):
            np.save(path, fit.scores[:, i : i + 1])
        labels = folder / "labels.npy"
        np.save(labels, fit.labels(law))
        bindings = {path.name: val.scores[:, i : i + 1] for i, path in enumerate(paths)}
        args = paths, labels, bindings, val.labels(law), gp_seed
        if arm == "stock":
            # First pass uses the native mutation method and performs no intervention draws.
            bare = NativeAudit(*args)
            bare.replay()
            totals.update(
                {
                    "stock_" + k: v
                    for k, v in verify_snapshots(
                        bare, case, winners, ledgers, False
                    ).items()
                }
            )
            totals["unmodified_native_runs"] += 1
        actor = EventAudit(*args)
        actor.arm = arm
        actor.event_seed = int(
            np.random.SeedSequence([seed, gp_seed, 419]).generate_state(1)[0]
        )
        actor.replay()
        equal(len(actor.events), len(events), "number of generation records")
        for i, event in enumerate(actor.events, start=1):
            for column, value in event.items():
                check_csv(value, events[i], column)
            totals["eligible_opportunities"] += event["offspring_index"] is not None
            totals["object_survivors"] += event["survived"] is True
            totals["equivalent_key_survivors"] += event["equivalent_survived"] is True
            totals["event_rows"] += 1
        totals.update(verify_snapshots(actor, case, winners, ledgers, True))
        totals["event_replays"] += 1
    return totals


def load_rows(directory):
    groups = defaultdict(lambda: ({}, {}, {}))
    for filename in ("runs", "costs", "events"):
        with gzip.open(directory / f"{filename}.csv.gz", "rt", newline="") as f:
            for row in csv.DictReader(f):
                if filename == "runs" and (
                    not row["gp_seed"] or row["split"] not in ("fit", "validation")
                ):
                    continue
                key = (
                    int(row["data_seed"]),
                    row["fit_law"],
                    int(float(row["gp_seed"])),
                    row["method"] if filename == "runs" else row["arm"],
                )
                winners, ledgers, events = groups[key]
                if filename == "runs":
                    mapping, pos = (
                        winners,
                        (row["checkpoint"], row["decision"], row["split"]),
                    )
                elif filename == "costs":
                    mapping, pos = ledgers, row["checkpoint"]
                else:
                    mapping, pos = events, int(row["generation"])
                if pos in mapping:
                    raise ValueError(f"Repeated {filename} record for {key!r}, {pos!r}")
                mapping[pos] = row
    expected = {
        (seed, law, gp, arm)
        for seed in range(4200, 4212)
        for law in LAWS
        for gp in (331, 337)
        for arm in ARMS
    }
    equal(set(groups), expected, "study cases")
    for case, (wins, ledger, events) in groups.items():
        equal(
            set(wins),
            {
                (cp, d, s)
                for cp in CHECKPOINTS
                for d in ("fit", "validation")
                for s in ("fit", "validation")
            },
            f"{case} saved winners",
        )
        equal(set(ledger), set(CHECKPOINTS), f"{case} ledgers")
        equal(set(events), set(range(1, max(events) + 1)), f"{case} event generations")
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    root = Path(okapi.__file__).resolve().parent
    equal(
        subprocess.check_output(
            ["git", "-C", root, "rev-parse", "HEAD"], text=True
        ).strip(),
        PIN,
        "OKAPI commit",
    )
    equal(
        subprocess.check_output(
            ["git", "-C", root, "status", "--short", "--untracked-files=no"], text=True
        ),
        "",
        "OKAPI tracked files",
    )
    logger.disable("okapi")
    groups = load_rows(args.directory)
    total = Counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for counts in pool.map(verify_group, groups.items()):
            total.update(counts)
    print(json.dumps({"okapi_commit": PIN, **dict(sorted(total.items()))}, indent=2))


if __name__ == "__main__":
    main()

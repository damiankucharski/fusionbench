"""Measure four native point policies at a fitness budget and a generation boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from itertools import product
from numbers import Integral
from pathlib import Path
from time import perf_counter

import numpy as np

from fusionbench.eval.full_suite_point import data_for, fingerprint
from fusionbench.eval.metrics import score
from fusionbench.eval.native_switch import serialize
from fusionbench.eval.suite_point_gp import (
    SuiteCheckpoint,
    _bind,
    _full,
    _loss,
    _targets,
    predict_tree,
)
from fusionbench.methods.okapi import OKAPIMethod, _task_postprocessing

ARMS = ("stock", "identity", "refresh", "retype")
GP_SEEDS = (661, 673)
DATA_SEEDS = range(5100, 5106)


@dataclass(frozen=True)
class OpportunityCheckpoint(SuiteCheckpoint):
    counts: dict[str, int]
    family_transitions: dict[str, int]


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def fit_mutation_opportunity(
    preds,
    fit_y,
    validation_preds,
    validation_y,
    *,
    seed,
    arm,
    budget=1200,
    generations=10,
):
    """Observe an ordered native run through both endpoints on fixed arrays.

    The calls checkpoint stops the training/validation archive exactly at the
    requested evaluation. Proposal counters include the whole already-generated
    candidate batch at that point. The generation checkpoint is observed after
    survival. Each unique program receives one held-out validation evaluation,
    with no influence on mutation, survival or the fit archive.
    """
    from experiments.point_ablation.moves import MatchedPointControls
    from okapi import Okapi
    from okapi.operators import (
        CLOSE_THRESHOLD,
        FAR_THRESHOLD,
        MAX,
        MEAN,
        MIN,
        WEIGHTED_MEAN,
    )

    if arm not in (*ARMS, "refresh_independent"):
        raise ValueError(f"Unknown point policy {arm!r}")
    budget = _positive_integer(budget, "budget")
    generations = _positive_integer(generations, "generations")
    fit, validation = _full(preds), _full(validation_preds)
    n, m, c = fit.shape
    if m < 2 or fit.shape[1:] != validation.shape[1:]:
        raise ValueError(
            "Expected matching provider and class counts and two or more providers"
        )
    labels, target = _targets(fit_y, n, c)
    _, val_target = _targets(validation_y, len(validation), c)
    val_bindings = _bind(validation)
    counts = dict.fromkeys(
        (
            "fit_calls",
            "validation_calls",
            "initial_trees",
            "native_fitness_batches",
            "completed_generations",
            "crossover_events",
            "crossover_children",
            "point_eligible_crossover_children",
            "structural_mutations",
            "point_eligible_structural_mutations",
            "mutation_calls",
            "mutation_children",
            "joined_trees",
            "dedup_removed",
            "point_opportunities",
            "point_family_changes",
            "point_program_changes",
            "point_original_parametric",
        ),
        0,
    )
    families = Counter()
    parametric = {
        cls.__name__ for cls in (WEIGHTED_MEAN, FAR_THRESHOLD, CLOSE_THRESHOLD)
    }
    seen, checkpoints = set(), {}
    digest = hashlib.sha256()
    best_fit = best_validation = None
    batch_losses = []
    generation = 0

    def observe_event(event):
        counts["point_opportunities"] += 1
        counts["point_family_changes"] += int(
            event.original_family != event.returned_family
        )
        counts["point_program_changes"] += int(event.changed)
        counts["point_original_parametric"] += int(event.original_family in parametric)
        names = (event.original_family, event.proposed_family, event.returned_family)
        families["/".join(names)] += 1

    def objective(p, _):
        loss = _loss(p, target)
        batch_losses.append(loss)
        return 1 - loss

    def remember(tree, fit_loss):
        nonlocal best_fit, best_validation
        counts["fit_calls"] += 1
        key = tree.structural_key()
        digest.update((repr(key) + "\n").encode())
        digest.update(struct.pack("<d", fit_loss))
        if key in seen:
            return
        seen.add(key)
        p = tree.do_pred_on_another_tensors(current_tensors=val_bindings)
        val_loss = _loss(p, val_target)
        counts["validation_calls"] += 1
        fit_wins = best_fit is None or fit_loss < best_fit[0]
        validation_wins = (
            best_validation is None or (val_loss, fit_loss) < best_validation[:2]
        )
        if fit_wins or validation_wins:
            champion = tree.copy()
            champion.mutation_chance = tree.mutation_chance
            if fit_wins:
                best_fit = (fit_loss, champion)
            if validation_wins:
                best_validation = (val_loss, fit_loss, champion)

    def capture(endpoint):
        checkpoints[endpoint] = OpportunityCheckpoint(
            fit_tree=best_fit[-1],
            val_tree=best_validation[-1],
            calls=counts["fit_calls"],
            n_trees=len(seen),
            generation=generation,
            path_digest=digest.hexdigest(),
            elapsed_seconds=perf_counter() - start,
            fit_brier=best_fit[0],
            validation_brier=best_validation[0],
            counts=dict(counts),
            family_transitions=dict(families),
        )

    class Finished(Exception):
        pass

    class Observed(MatchedPointControls, Okapi):
        pending_join = 0

        def _perform_crossovers(self, fitnesses):
            count = super()._perform_crossovers(fitnesses)
            counts["crossover_events"] += count
            counts["crossover_children"] += len(self.additional_population)
            counts["point_eligible_crossover_children"] += sum(
                bool(t.op_nodes) for t in self.additional_population
            )
            return count

        def _apply_structural_mutation(self, mutation, tree):
            counts["structural_mutations"] += 1
            counts["point_eligible_structural_mutations"] += int(bool(tree.op_nodes))
            return super()._apply_structural_mutation(mutation, tree)

        def _mutate_additional_population(self):
            before = len(self.additional_population)
            count = super()._mutate_additional_population()
            counts["mutation_calls"] += count
            counts["mutation_children"] += len(self.additional_population) - before
            self.pending_join = len(self.population) + len(self.additional_population)
            counts["joined_trees"] += self.pending_join
            return count

        def _calculate_fitnesses(self, trees=None):
            cohort = self.population if trees is None else trees
            counts["native_fitness_batches"] += 1
            if self.pending_join:
                counts["dedup_removed"] += self.pending_join - len(cohort)
                self.pending_join = 0
            else:
                counts["initial_trees"] += len(cohort)
            # Split exactly at the budget so no future fitness or validation score
            # has been computed when the calls checkpoint is recorded. Native fitness
            # is an ordered deterministic per-tree map. Survivor selection still sees
            # the original entire batch once, except after both requested endpoints.
            remaining = budget - counts["fit_calls"]
            parts = (
                (cohort[:remaining], cohort[remaining:])
                if 0 < remaining < len(cohort)
                else (cohort,)
            )
            outputs = []
            for portion in parts:
                batch_losses.clear()
                result = super()._calculate_fitnesses(portion)
                outputs.append(result)
                for tree, loss in zip(portion, batch_losses, strict=True):
                    remember(tree, loss)
                if counts["fit_calls"] == budget:
                    capture("calls")
                    if "generation" in checkpoints:
                        raise Finished
            return outputs[0] if len(outputs) == 1 else np.concatenate(outputs)

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
                operator_point_mutation=arm != "stock",
                reuse_survivor_fitness=False,
            )
            engine.point_policy = "retype" if arm == "stock" else arm
            engine.point_event_callback = observe_event
            adapter._seed_population(engine, np.random.default_rng(seed))
            try:
                while len(checkpoints) < 2:
                    generation += 1
                    engine.run_iteration()
                    counts["completed_generations"] += 1
                    if generation == generations:
                        capture("generation")
            except Finished:
                pass
        finally:
            adapter._tmp.cleanup()
    return checkpoints


def evaluate_bank(
    name,
    data_seed,
    splits,
    *,
    gp_seeds=GP_SEEDS,
    arms=ARMS,
    include_baselines=True,
    budget=1200,
    generations=10,
):
    """Evaluate fixed providers on fit/validation and any named held-out splits.

    Each split is (row-by-provider probabilities, class labels, optional true class
    probabilities). Baseline selection uses fit only. The identifier data_seed can
    also denote a predefined real-data fold.
    """
    from loguru import logger

    from fusionbench.methods.averaging import TopN
    from fusionbench.methods.stacking import StackingLogit

    logger.disable("okapi")
    blocks = {
        split: (_full(x), np.asarray(y).ravel(), q)
        for split, (x, y, q) in splits.items()
    }
    x, y, _ = blocks["fit"]
    vx, vy, _ = blocks["validation"]
    classes = x.shape[2]
    task = "binary" if classes == 2 else "multiclass"
    rows = []

    def record(method, split, p, extra):
        z, labels, oracle = blocks[split]
        p = np.asarray(p)
        if p.shape != (len(labels), classes) or not np.isfinite(p).all():
            raise ValueError(
                f"Invalid prediction shape or values for {method} on {split}"
            )
        if np.any(p < -1e-14) or np.any(p > 1 + 1e-14):
            raise ValueError(f"{method} produced a class probability outside [0,1]")
        if not np.allclose(p.sum(axis=1), 1, rtol=0, atol=2e-12):
            raise ValueError(f"{method} produced nonunit probability mass")
        metrics = score(
            labels, p, classes, metrics=["accuracy", "macro_f1", "log_loss"]
        )
        rows.append(
            {
                "scenario": name,
                "data_seed": data_seed,
                "classes": classes,
                "providers": z.shape[1],
                "rows": len(labels),
                "method": method,
                "split": split,
                "prediction_sha256": fingerprint(p),
                "oracle_sha256": None if oracle is None else fingerprint(oracle),
                "brier": float(
                    np.mean(np.sum((p - np.eye(classes)[labels]) ** 2, axis=1))
                ),
                "conditional_excess": (
                    None
                    if oracle is None
                    else float(np.mean(np.sum((p - oracle) ** 2, axis=1)))
                ),
                **metrics,
                **extra,
            }
        )

    if include_baselines:
        top = TopN("brier").fit(x, y, task=task, n_classes=classes)
        stack = StackingLogit().fit(x, y, task=task, n_classes=classes)
        fixed = {"checkpoint": "fixed", "selection": "fit"}
        top_details = {
            **fixed,
            "n_selected": top.n_selected_,
            "ranking": json.dumps(top.ranking_.tolist()),
            "prefix_scores": json.dumps(top.prefix_scores_.tolist()),
        }
        for split, (z, _, _) in blocks.items():
            record("mean", split, z.mean(axis=1), fixed)
            record("top_n", split, top.predict(z), top_details)
            record("stack_logit", split, stack.predict(z), fixed)

    for gp_seed, arm in product(gp_seeds, arms):
        checkpoints = fit_mutation_opportunity(
            x, y, vx, vy, seed=gp_seed, arm=arm, budget=budget, generations=generations
        )
        for endpoint, cp in checkpoints.items():
            for selection, tree in (
                ("fit", cp.fit_tree),
                ("validation", cp.validation_tree),
            ):
                metadata = {
                    "checkpoint": endpoint,
                    "selection": selection,
                    "gp_seed": gp_seed,
                    "calls": cp.calls,
                    "generation": cp.generation,
                    "nodes": tree.nodes_count,
                    "archive_trees": cp.n_trees,
                    "search_seconds": cp.elapsed_seconds,
                    "path_digest": cp.path_digest,
                    "program": json.dumps(serialize(tree.root)),
                    "archive_selection_brier": (
                        cp.fit_brier if selection == "fit" else cp.validation_brier
                    ),
                    "family_transitions": json.dumps(
                        cp.family_transitions, sort_keys=True
                    ),
                    **cp.counts,
                }
                for split, (z, _, _) in blocks.items():
                    record(arm, split, predict_tree(tree, z), metadata)
    return rows


def _synthetic_bank(pair, intervention_only=None):
    name, data_seed = pair
    _, splits = data_for(name, data_seed)
    if intervention_only is None:
        return evaluate_bank(name, data_seed, splits)
    return evaluate_bank(
        name, data_seed, splits, arms=(intervention_only,), include_baselines=False
    )


def run(out, workers, intervention_only=None):
    """Write one row per search, decision, endpoint and split, plus baselines."""
    import pandas as pd

    from fusionbench.scenarios.registry import STANDARD_V1_SCENARIOS as SCENARIOS

    out = Path(out)
    out.mkdir(exist_ok=True, parents=True)
    if intervention_only not in (None, "refresh_independent"):
        raise ValueError(f"Unknown single-arm study {intervention_only!r}")
    job = partial(_synthetic_bank, intervention_only=intervention_only)
    filename = (
        "runs.csv.gz"
        if intervention_only is None
        else "refresh-independent-runs.csv.gz"
    )
    with ProcessPoolExecutor(max_workers=workers) as pool:
        frame = pd.DataFrame(
            [
                row
                for bank in pool.map(job, product(SCENARIOS, DATA_SEEDS))
                for row in bank
            ]
        )
    pending = out / f"partial-{filename}"
    frame.to_csv(pending, index=False)
    pending.replace(out / filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--intervention-only", choices=("refresh_independent",))
    args = parser.parse_args()
    run(args.out, args.workers, args.intervention_only)

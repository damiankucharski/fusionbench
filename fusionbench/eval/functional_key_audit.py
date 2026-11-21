"""Test functional versus structural population diversity in native MIN/MAX GP.

Beadle and Johnson (2008) tested Boolean equivalence during crossover. Here
exact MIN/MAX function signatures determine which trees survive a population
merge. See beadle2008semantic in docs/ensemble-study/literature/README.md.
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from importlib import import_module
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from fusionbench.eval.lattice_signature import lattice_signature, provider_signatures
from fusionbench.eval.search_continuous import brier
from fusionbench.eval.structural_key_audit import (
    EndOfBudget,
    reference_panel,
    training_files,
)


def first_occurrences(values):
    seen, keep = set(), []
    for value in values:
        keep.append(value not in seen)
        seen.add(value)
    return keep


def essential_count(function, provider_bits):
    # Compare matched input pairs with the selected bit off/on.
    return sum(
        bool((function ^ (function >> (1 << j))) & ~bits)
        for j, bits in enumerate(provider_bits)
    )


class JoinedPopulation:
    def __init__(self, policy, providers):
        self.policy = policy
        self.providers = providers
        self.structural = import_module("okapi.okapi")._unique_tree_mask
        self.batches = self.proposals = self.redundant = 0
        self.structural_seconds = self.functional_seconds = 0.0

    def __call__(self, trees):
        start = time.perf_counter()
        structural = self.structural(trees)
        after_structural = time.perf_counter()
        values = [lattice_signature(t, self.providers) for t in trees]
        functional = first_occurrences(values)
        after_functional = time.perf_counter()
        self.structural_seconds += after_structural - start
        self.functional_seconds += after_functional - after_structural
        self.batches += 1
        self.proposals += len(trees)
        self.redundant += sum(structural) - sum(functional)
        assert all(s or not f for s, f in zip(structural, functional, strict=True))
        chosen = functional if self.policy == "functional" else structural
        assert {v for v, keep in zip(values, chosen, strict=True) if keep} == set(
            values
        )
        return chosen


def search(paths, gt, reference, seed, policy):
    from okapi import Okapi
    from okapi.operators import MAX, MIN

    providers = provider_signatures([p.name for p in paths])
    monitor = JoinedPopulation(policy, providers)
    structures, functions = set(), {}
    snapshots = []
    calls = completed = 0
    fit_loss = None
    start = time.perf_counter()

    def objective(p, y):
        nonlocal fit_loss
        fit_loss = brier(p.ravel(), y.ravel())
        return 1 - fit_loss

    def snapshot(checkpoint):
        best_fit = min(functions.values(), key=lambda e: e[0])
        best_val = min(functions.values(), key=lambda e: (e[1], e[0]))
        arities = [e[3] for e in functions.values()]
        metadata = {
            "checkpoint": checkpoint,
            "calls": calls,
            "completed_generations": completed,
            "observed_joins": monitor.batches,
            "proposals": monitor.proposals,
            "redundant_structural_firsts": monitor.redundant,
            "distinct_structures": len(structures),
            "distinct_functions": len(functions),
            "mean_essential": float(np.mean(arities)),
            "max_essential": max(arities),
            "structural_mask_seconds": monitor.structural_seconds,
            "functional_mask_seconds": monitor.functional_seconds,
            "audit_wall_seconds": time.perf_counter() - start,
        }
        snapshots.append((metadata, best_fit[2], best_val[2]))

    class LoggedGP(Okapi):
        def _calculate_fitnesses(self, trees=None):
            nonlocal calls
            scores = []
            for tree in self.population if trees is None else trees:
                row = super()._calculate_fitnesses([tree])[0]
                calls += 1
                structures.add(tree.structural_key())
                signature = lattice_signature(tree, providers)
                if signature not in functions:
                    # All other genotypes with this signature have identical losses on every split.
                    ref = tree.copy()
                    prediction = ref.do_pred_on_another_tensors(
                        current_tensors=reference.tensors[1]
                    ).ravel()
                    functions[signature] = (
                        fit_loss,
                        brier(prediction, reference.labels[1]),
                        ref,
                        essential_count(signature, providers.values()),
                    )
                if calls in (300, 1200):
                    snapshot(f"calls_{calls}")
                    if calls == 1200 and completed >= 20:
                        raise EndOfBudget
                scores.append(row)
            return np.stack(scores)

    gp = LoggedGP(
        paths,
        gt,
        len(paths),
        2,
        2,
        backend="numpy",
        seed=seed,
        allowed_ops=(MIN, MAX),
        postprocessing_function=np.asarray,
        objective_functions=(objective,),
        reuse_survivor_fitness=False,
    )
    with patch.object(import_module("okapi.okapi"), "_unique_tree_mask", monitor):
        try:
            while calls < 1200 or completed < 20:
                gp.run_iteration()
                completed += 1
                if completed == 20:
                    snapshot("generations_20")
        except EndOfBudget:
            pass
    assert {e[0]["checkpoint"] for e in snapshots} == {
        "calls_300",
        "calls_1200",
        "generations_20",
    }
    return snapshots


def panel(seed):
    from loguru import logger

    logger.disable("okapi")
    ref = reference_panel(seed)
    rows = []
    with training_files(ref.train, ref.labels[0]) as (paths, gt):
        for gp_seed in (211, 223, 227, 229):
            for policy in ("structural", "functional"):
                snaps = search(paths, gt, ref, gp_seed, policy)
                rows.extend(ref.champion_rows(snaps, gp_seed, policy))
    return ref.baseline, rows


def run(out, workers):
    out.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        panels = list(pool.map(panel, range(3600, 3612)))
    baselines = [row.copy() for group, _ in panels for row in group]
    for row in baselines:
        for field in ("optimum_numerator", "optimum_denominator"):
            if field in row:
                row[field] = str(row[field])
    pd.DataFrame(baselines).to_csv(out / "baselines.csv", index=False)
    pd.DataFrame([row for _, group in panels for row in group]).to_csv(
        out / "gp.csv", index=False
    )


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--out", required=True, type=Path)
    cli.add_argument("--workers", default=1, type=int)
    args = cli.parse_args()
    run(args.out, args.workers)

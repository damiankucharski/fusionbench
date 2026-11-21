"""Exact conditional randomization references for semantic coverage of OKAPI search."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from math import comb
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from fusionbench.eval.search_regret import SCORES, TASKS, Problem, family, run_search


def arities(tables):
    """Number of essential variables, using interventions on every truth-table row."""
    d = int(np.log2(tables.shape[1]))
    if tables.shape[1] != 2**d:
        raise ValueError("truth tables must span a Boolean cube")
    cells = np.arange(2**d)
    return sum(np.any(tables != tables[:, cells ^ (1 << i)], axis=1) for i in range(d))


def reference(errors, strata, draws):
    """Hit probability and expected regret of draws uniformly without replacement per stratum."""
    errors = np.asarray(errors)
    if sum(draws) == 0 or len(errors) != len(strata):
        raise ValueError("at least one draw and one stratum per candidate are required")
    levels = np.unique(errors)
    survival = np.ones(len(levels))
    for h, n in enumerate(draws):
        subset = errors[strata == h]
        if n < 0 or n > len(subset):
            raise ValueError("invalid stratum sample size")
        if n:
            den = comb(len(subset), n)
            survival *= [comb(int(np.sum(subset > cut)), n) / den for cut in levels]
    # Survival at each interval's left endpoint gives E[min loss - global min].
    expected_gap = np.sum(survival[:-1] * np.diff(levels))
    return 1 - survival[0], float(expected_gap)


def panel(job):
    task, data_seed, gp_seeds = job
    p = Problem(task, data_seed, 128, 128, 128, 4096)
    tables, codes = family()
    arity = arities(tables)
    errors = p.fit.errors(tables)
    eligible = errors == p.best_errors
    lookup = {int(code): j for j, code in enumerate(codes)}
    rows = []
    with TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        paths = [tmp / f"provider{i}.npy" for i in range(4)]
        for i, f in enumerate(paths):
            np.save(f, SCORES[:, i : i + 1])
        label_file = tmp / "labels.npy"
        np.save(
            label_file,
            p.fit.pos[:, None] / np.maximum((p.fit.pos + p.fit.neg)[:, None], 1),
        )

        def audit(generation, visited):
            selected = np.array([lookup[code] for code in visited])
            counts = np.bincount(arity[selected], minlength=5)
            unconditional = reference(
                errors, np.zeros(len(errors), dtype=int), [len(visited)]
            )
            conditional = reference(errors, arity, counts)
            opt_counts = np.bincount(arity[eligible], minlength=5)
            loss = int(errors[selected].min() - p.best_errors)
            rows.append(
                dict(
                    task=task,
                    data_seed=data_seed,
                    gp_seed=gp_seed,
                    generation=generation,
                    unique=len(visited),
                    actual_hit=int(loss == 0),
                    actual_regret=loss,
                    uniform_hit=unconditional[0],
                    uniform_regret=unconditional[1],
                    matched_hit=conditional[0],
                    matched_regret=conditional[1],
                    **{f"visited_arity_{d}": int(counts[d]) for d in range(1, 5)},
                    **{f"optima_arity_{d}": int(opt_counts[d]) for d in range(1, 5)},
                )
            )

        for gp_seed in gp_seeds:
            run_search(
                p, paths, label_file, gp_seed, [4, 16, 48], "minmax", observe=audit
            )
    return rows


def run(out, seeds=range(3100, 3112), gp_seeds=(71, 73, 79), workers=1):
    out = Path(out)
    out.mkdir(exist_ok=True, parents=True)
    jobs = [(task, seed, gp_seeds) for task, seed in product(TASKS, seeds)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = [row for batch in pool.map(panel, jobs) for row in batch]
    data = pd.DataFrame(rows)
    data.to_csv(out / "coverage.csv", index=False)
    means = (
        data.groupby(["task", "generation", "data_seed"])
        .mean(numeric_only=True)
        .drop(columns="gp_seed")
    )
    means.groupby(["task", "generation"]).mean().to_csv(out / "means.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    run(args.out, workers=args.workers)

"""Exact empirical search regret and Brier risk of four graded provider scores."""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from fusionbench.eval.search_regret import Counts
from fusionbench.methods.stacking import StackingLogit

LEVELS = np.array([0.10, 0.35, 0.65, 0.90])
DIGITS = np.array(list(product(range(4), repeat=4)))
SCORES = LEVELS[DIGITS]
PROBA = np.stack([1 - SCORES, SCORES], axis=-1)
IID = np.full(len(SCORES), 1 / len(SCORES))
SHIFT_LEVELS = np.array([0.05, 0.10, 0.25, 0.60])
SHIFT = np.prod(
    np.where(np.arange(4) % 2 == 0, SHIFT_LEVELS[DIGITS], SHIFT_LEVELS[::-1][DIGITS]),
    axis=1,
)
BUDGETS = (4, 80, 300)


@lru_cache(maxsize=1)
def lattice():
    from okapi.lattice_search import SmallLattice, small_lattice_programs

    prog = small_lattice_programs(4)
    rules = np.stack([SmallLattice(p, 4, len(prog)).predict(SCORES) for p in prog])
    lookup = {row.tobytes(): i for i, row in enumerate(rules)}
    assert len(lookup) == 166
    return rules, lookup


def conditional(task, order):
    a, b, c, d = SCORES[:, order].T
    if task == "matched":
        return np.maximum(np.minimum(a, b), np.minimum(c, d))
    if task == "mean":
        return SCORES.mean(axis=1)
    if task == "additive":
        return expit(
            -0.2 + logit(SCORES[:, order]) @ np.array([0.45, 0.65, 0.35, 0.55])
        )
    raise ValueError(f"unknown task {task}")


def sample(seed, split, n, weights, eta):
    rng = np.random.default_rng([seed, split, 491])
    cells = rng.choice(len(eta), size=n, p=weights)
    return cells, rng.random(n) < eta[cells]


def counts(cells, y):
    return Counts(
        np.bincount(cells[y], minlength=len(IID)),
        np.bincount(cells[~y], minlength=len(IID)),
    )


def brier(scores, observed):
    return (
        (scores * scores) @ (observed.pos + observed.neg)
        - 2 * scores @ observed.pos
        + observed.pos.sum()
    ) / observed.size


def risk(scores, eta, weights):
    return ((scores - eta) ** 2 + eta * (1 - eta)) @ weights


def distributed(gp, seed):
    from okapi.node import ValueNode, MinNode, MaxNode
    from okapi.tree import Tree

    rng = np.random.default_rng([seed, 917])
    trees = []
    for arity in range(1, 5):
        ids = rng.choice(4, size=arity, replace=False)
        nodes = [ValueNode(None, gp.models[i], gp.ids[i]) for i in ids]
        while len(nodes) > 1:
            k = int(rng.integers(len(nodes) - 1))
            op = MinNode if rng.integers(2) == 0 else MaxNode
            nodes[k].add_child(op([nodes.pop(k + 1)]))
        trees.append(Tree(nodes[0]))
    return trees


def search(paths, file, fit, seed, mode):
    from okapi import Okapi
    from okapi.operators import MIN, MAX

    _, lookup = lattice()
    archive = {}
    snapshots = {}
    call = 0
    start = time.perf_counter()

    def objective(scores, labels):
        nonlocal call
        values = scores.ravel()
        index = lookup[values.tobytes()]
        if index not in archive:
            archive[index] = brier(values, fit)
        call += 1
        if call in BUDGETS:
            snapshots[call] = (np.array(list(archive)), time.perf_counter() - start)
        return 1 - archive[index]

    gp = Okapi(
        paths,
        file,
        4,
        2,
        2,
        backend="numpy",
        seed=seed,
        allowed_ops=(MIN, MAX),
        objective_functions=(objective,),
        postprocessing_function=np.asarray,
    )
    if mode == "one_per_arity":
        gp.population = distributed(gp, seed)
    while call < max(BUDGETS):
        gp.run_iteration()
    return snapshots


def panel(job):
    task, seed, gp_seeds = job
    rules, _ = lattice()
    order = np.random.default_rng([seed, 903]).permutation(4)
    eta = conditional(task, order)
    cells, labels = sample(seed, 0, 128, IID, eta)
    val = counts(*sample(seed, 1, 128, IID, eta))
    ti = counts(*sample(seed, 2, 4096, IID, eta))
    ts = counts(*sample(seed, 3, 4096, SHIFT, eta))
    rows = []
    with TemporaryDirectory() as directory:
        tmp = Path(directory)
        paths = [tmp / f"provider{i}.npy" for i in range(4)]
        for i, path in enumerate(paths):
            np.save(path, SCORES[:, i : i + 1])
        gt = tmp / "labels.npy"
        for size in (32, 128):
            fit = counts(cells[:size], labels[:size])
            np.save(gt, fit.pos[:, None] / np.maximum((fit.pos + fit.neg)[:, None], 1))
            loss = brier(rules, fit)
            # Decimal input levels and binary labels give integer-scaled squared
            # losses. Rounding negligible floating noise preserves exact ties.
            counts_loss = np.rint(loss * size * 400).astype(int)
            assert np.max(np.abs(counts_loss / (size * 400) - loss)) < 1e-12
            best = int(counts_loss.argmin())
            val_loss = np.rint(brier(rules, val) * val.size * 400).astype(int)
            base = dict(
                task=task,
                data_seed=seed,
                n_fit=size,
                order="".join(map(str, order)),
                family_fit=loss[best],
                family_iid=risk(rules, eta, IID).min(),
                family_shift=risk(rules, eta, SHIFT).min(),
                bayes_iid=risk(eta, eta, IID),
                bayes_shift=risk(eta, eta, SHIFT),
                tied=int(np.sum(counts_loss == counts_loss.min())),
            )

            def record(
                pred,
                mode,
                selector,
                index=None,
                search_seed=None,
                budget=None,
                unique=None,
                elapsed=None,
            ):
                measured = brier(pred, fit)
                rows.append(
                    dict(
                        **base,
                        mode=mode,
                        selector=selector,
                        index=index,
                        search_seed=search_seed,
                        budget=budget,
                        unique=unique,
                        seconds=elapsed,
                        fit=measured,
                        regret=measured - loss[best],
                        hit=bool(counts_loss[index] == counts_loss.min())
                        if index is not None
                        else None,
                        validation=brier(pred, val),
                        iid_test=brier(pred, ti),
                        shift_test=brier(pred, ts),
                        iid_risk=risk(pred, eta, IID),
                        shift_risk=risk(pred, eta, SHIFT),
                    )
                )

            record(rules[best], "exact_erm", "fit", index=best)
            record(SCORES.mean(axis=1), "average", "fixed")
            if np.unique(labels[:size]).size == 1:
                stack = np.full(len(IID), labels[0])
            else:
                lr = StackingLogit().fit(
                    PROBA[cells[:size]],
                    labels[:size].astype(int),
                    task="binary",
                    n_classes=2,
                )
                stack = lr.predict(PROBA)[:, 1]
            record(stack, "stack_logit", "fit")
            for mode, subseed in product(("singletons", "one_per_arity"), gp_seeds):
                for budget, (indices, elapsed) in search(
                    paths, gt, fit, subseed, mode
                ).items():
                    j = indices[np.argmin(counts_loss[indices])]
                    k = min(indices, key=lambda i: (val_loss[i], counts_loss[i]))
                    for selector, i in [("fit", j), ("validation", k)]:
                        record(
                            rules[i],
                            mode,
                            selector,
                            index=int(i),
                            search_seed=subseed,
                            budget=budget,
                            unique=len(indices),
                            elapsed=elapsed,
                        )
    return rows


def run(out, seeds=range(3300, 3306), gp_seeds=range(101, 109), workers=1):
    out = Path(out)
    out.mkdir(exist_ok=True, parents=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = [
            (task, seed, gp_seeds)
            for task, seed in product(("matched", "mean", "additive"), seeds)
        ]
        df = pd.DataFrame([r for batch in pool.map(panel, jobs) for r in batch])
    df.to_csv(out / "runs.csv", index=False)
    keys = ["task", "n_fit", "mode", "selector", "budget"]
    columns = [
        "regret",
        "hit",
        "fit",
        "validation",
        "iid_risk",
        "shift_risk",
        "bayes_iid",
        "family_iid",
        "unique",
        "seconds",
    ]
    df.groupby(keys + ["data_seed"], dropna=False)[columns].mean().groupby(
        keys, dropna=False
    ).mean().to_csv(out / "means.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.out, workers=args.workers)

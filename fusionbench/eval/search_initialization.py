"""Intervene on the four starting trees, comparing identical fitness-call prefixes."""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from fusionbench.eval.search_coverage import arities
from fusionbench.eval.search_regret import (
    POWERS,
    SCORES,
    TASKS,
    FitnessArchive,
    Problem,
    _decisions,
    family,
)

MODES = ("singletons", "one_per_arity", "uniform_family", "balanced")
BUDGETS = (4, 80, 300)


def starters(mode, paths, seed):
    from okapi.lattice_search import small_lattice_programs
    from okapi.node import MaxNode, MinNode, ValueNode
    from okapi.tree import Tree

    def leaf(i):
        return ValueNode(None, SCORES[:, i : i + 1], paths[i].name)

    def combine(left, right, op):
        left.add_child(op([right]))
        return left

    def conjunction(term):
        node = leaf(term[0])
        if len(term) > 1:
            node.add_child(MinNode([leaf(i) for i in term[1:]]))
        return node

    rng = np.random.default_rng([seed, 802])
    if mode == "balanced":
        trees = []
        for _ in range(4):
            a, b, c, d = rng.permutation(4)
            ops = [MaxNode if side else MinNode for side in rng.integers(0, 2, 3)]
            left = combine(leaf(a), leaf(b), ops[0])
            right = combine(leaf(c), leaf(d), ops[1])
            trees.append(Tree(combine(left, right, ops[2])))
        return trees

    tables, codes = family()
    if mode == "one_per_arity":
        arity = arities(tables)
        indices = [rng.choice(np.flatnonzero(arity == d)) for d in range(1, 5)]
    elif mode == "uniform_family":
        indices = rng.choice(len(tables), 4, replace=False)
    else:
        raise ValueError(f"unknown initialization: {mode}")

    programs = small_lattice_programs(4)
    trees = []
    for index in indices:
        terms = programs[index]
        root = conjunction(terms[0])
        if len(terms) > 1:
            root.add_child(MaxNode([conjunction(term) for term in terms[1:]]))
        tree = Tree(root)
        assert int((tree.predict().ravel() > 0.5) @ POWERS) == codes[index]
        trees.append(tree)
    return trees


class PrefixArchive(FitnessArchive):
    def __init__(self, fit):
        super().__init__(fit)
        self.snapshots = {}
        self.start = time.perf_counter()

    def __call__(self, score, labels):
        result = super().__call__(score, labels)
        if self.calls in BUDGETS:
            self.snapshots[self.calls] = (
                tuple(self.errors),
                time.perf_counter() - self.start,
            )
        return result


def prefixes(p, paths, label_file, seed, mode):
    from okapi import Okapi
    from okapi.operators import MIN, MAX

    archive = PrefixArchive(p.fit)
    gp = Okapi(
        paths,
        label_file,
        4,
        2,
        2,
        objective_functions=(archive,),
        backend="numpy",
        seed=seed,
        allowed_ops=(MIN, MAX),
        postprocessing_function=np.asarray,
    )
    if mode != "singletons":
        gp.population = starters(mode, paths, seed)
    initial_nodes = np.mean([tree.nodes_count for tree in gp.population])
    while archive.calls < max(BUDGETS):
        gp.run_iteration()
    return archive.snapshots, initial_nodes


def panel(job):
    task, data_seed, search_seeds = job
    tables, codes = family()
    code_to_ix = {int(c): i for i, c in enumerate(codes)}
    rows = []
    with TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        paths = [tmp / f"provider{i}.npy" for i in range(4)]
        for i, path in enumerate(paths):
            np.save(path, SCORES[:, i : i + 1])
        label_file = tmp / "labels.npy"
        for n in (32, 128):
            p = Problem(task, data_seed, n, 128, 128, 4096)
            np.save(
                label_file,
                p.fit.pos[:, None] / np.maximum(1, (p.fit.pos + p.fit.neg)[:, None]),
            )
            fit = p.fit.errors(tables)
            val = p.val.errors(tables)
            for seed in search_seeds:
                for mode in (*MODES, "uniform_draw"):
                    if mode == "uniform_draw":
                        start = time.perf_counter()
                        perm = np.random.default_rng([seed, 719]).permutation(codes)
                        samples = {}
                        scored = 0
                        for budget in BUDGETS:
                            selected = perm[:budget]
                            # Evaluate the same binary loss on the actual proposal prefix.
                            p.fit.errors(
                                np.stack([_decisions(c) for c in selected[scored:]])
                            )
                            scored = len(selected)
                            samples[budget] = (
                                tuple(selected),
                                time.perf_counter() - start,
                            )
                        init_nodes = None
                    else:
                        samples, init_nodes = prefixes(p, paths, label_file, seed, mode)
                    for budget, (visited, secs) in samples.items():
                        indices = np.array([code_to_ix[c] for c in visited])
                        best = indices[np.argmin(fit[indices])]
                        valid = min(indices, key=lambda i: (val[i], fit[i]))
                        for selector, index in [("fit", best), ("validation", valid)]:
                            rows.append(
                                dict(
                                    task=task,
                                    n_fit=n,
                                    data_seed=data_seed,
                                    gp_seed=seed,
                                    mode=mode,
                                    budget=budget,
                                    selector=selector,
                                    unique=len(indices),
                                    calls=min(budget, len(codes))
                                    if mode == "uniform_draw"
                                    else budget,
                                    initial_nodes=init_nodes,
                                    seconds=secs,
                                    **p.metrics(codes[index]),
                                )
                            )
    return rows


def run(out, data_seeds=range(3200, 3212), search_seeds=(91, 97), workers=1):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(t, d, search_seeds) for t, d in product(TASKS, data_seeds)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = [r for batch in pool.map(panel, jobs) for r in batch]
    df = pd.DataFrame(rows)
    df["hit"] = df.delta_family_fit_errors.eq(0)
    df.to_csv(out / "runs.csv", index=False)
    cols = [
        "delta_family_fit_errors",
        "hit",
        "fit_loss",
        "iid_risk",
        "shift_risk",
        "unique",
        "calls",
        "seconds",
        "initial_nodes",
    ]
    keys = ["task", "n_fit", "mode", "budget", "selector"]
    df.groupby(keys + ["data_seed"])[cols].mean().groupby(keys).mean().to_csv(
        out / "means.csv"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--data-seeds", type=int, nargs="+", default=list(range(3200, 3212))
    )
    parser.add_argument("--gp-seeds", type=int, nargs="+", default=[91, 97])
    args = parser.parse_args()
    run(args.out, args.data_seeds, args.gp_seeds, args.workers)

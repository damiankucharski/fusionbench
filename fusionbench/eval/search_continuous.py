"""Audit native MIN/MAX GP against a certified Brier optimum on continuous provider scores."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from fusionbench.eval.lattice_brier_oracle import fit_lattice_brier
from fusionbench.methods.averaging import TopN
from fusionbench.methods.stacking import StackingLogit
from fusionbench.scenarios.gates import TwoPathways


def _split(seed, split, n, accuracy=0.9):
    scenario = TwoPathways(rep=2, accuracy=accuracy)
    p, y, latent = scenario._generate(n, np.random.default_rng([seed, split]))
    return p, y, scenario.bayes_oracle(p, latent)[:, 1]


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def structural_key(node):
    return node.code, tuple(structural_key(child) for child in node.children)


def searched(x, y, validation, vlabels, seed):
    from okapi import Okapi
    from okapi.operators import MIN, MAX

    class LoggedOkapi(Okapi):
        """Observe evaluations while delegating each fitness calculation to the engine."""

        def _calculate_fitnesses(self, trees=None):
            trees = self.population if trees is None else trees
            rows = []
            for tree in trees:
                score = super()._calculate_fitnesses([tree])
                if self.evaluations < 300:
                    self.evaluations += 1
                    key = structural_key(tree.root)
                    if key not in self.archive:
                        self.archive[key] = (1 - float(score[0, 0]), tree.copy())
                rows.append(score[0])
            return np.stack(rows)

    with TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        paths = [tmp / f"p{i}.npy" for i in range(x.shape[1])]
        for i, path in enumerate(paths):
            np.save(path, x[:, i : i + 1])
        gt = tmp / "labels.npy"
        np.save(gt, y)
        gp = LoggedOkapi(
            paths,
            gt,
            8,
            2,
            2,
            backend="numpy",
            seed=seed,
            allowed_ops=(MIN, MAX),
            postprocessing_function=np.asarray,
            objective_functions=(lambda p, y: 1 - brier(p.ravel(), y.ravel()),),
        )
        gp.evaluations = 0
        gp.archive = {}
        start = time.perf_counter()
        while gp.evaluations < 300:
            gp.run_iteration()
        elapsed = time.perf_counter() - start
        tensors = {path.name: validation[:, i : i + 1] for i, path in enumerate(paths)}
        items = list(gp.archive.values())
        fit_winner = min(items, key=lambda e: e[0])[1]
        val_winner = min(
            items,
            key=lambda e: (
                brier(
                    e[1].do_pred_on_another_tensors(current_tensors=tensors).ravel(),
                    vlabels,
                ),
                e[0],
            ),
        )[1]
        return fit_winner, val_winner, elapsed, len(items)


def panel(seed):
    train = _split(seed, 0, 128)
    val = _split(seed, 1, 128)
    iid = _split(seed, 2, 8192)
    shift = _split(seed, 3, 8192, 0.78)
    rows = []
    for n in (32, 128):
        fit = tuple(z[:n] for z in train)
        x, y = fit[0][:, :, 1], fit[1]
        start = time.perf_counter()
        exact = fit_lattice_brier(x, y)
        seconds = time.perf_counter() - start
        optimum = float(exact.minimum_loss)
        splits = (fit, val, iid, shift)
        pred = [exact.predict(p[:, :, 1]) for p, labels, o in splits]
        assert abs(brier(pred[0], y) - optimum) < 1e-14

        def record(
            method,
            probs,
            seedid=None,
            nodes=None,
            search_time=None,
            visited=None,
            top=None,
        ):
            loss = [brier(p, part[1]) for p, part in zip(probs, splits, strict=True)]
            rows.append(
                dict(
                    data_seed=seed,
                    n_fit=n,
                    method=method,
                    gp_seed=seedid,
                    fit=loss[0],
                    validation=loss[1],
                    iid=loss[2],
                    shift=loss[3],
                    fit_minus_lattice_optimum=loss[0] - optimum,
                    exact_fit_numerator=exact.minimum_loss.numerator,
                    exact_fit_denominator=exact.minimum_loss.denominator,
                    exact_terms=len(exact.terms()),
                    exact_literals=sum(map(len, exact.terms())),
                    fit_seconds=search_time,
                    nodes=nodes,
                    top_n=top.n_selected_ if top is not None else None,
                    top_ranking=json.dumps(np.asarray(top.ranking_).tolist())
                    if top is not None
                    else None,
                    top_prefix_scores=json.dumps(
                        np.asarray(top.prefix_scores_).tolist()
                    )
                    if top is not None
                    else None,
                    visited=visited,
                )
            )

        record("exact_lattice", pred, search_time=seconds)
        record("average", [p[:, :, 1].mean(1) for p, _, _ in splits])
        top = TopN(metric="brier").fit(fit[0], y, task="binary", n_classes=2)
        record("top_n", [top.predict(p)[:, 1] for p, _, _ in splits], top=top)
        if np.unique(y).size == 1:
            stack = [np.full(len(p), y[0]) for p, _, _ in splits]
        else:
            lr = StackingLogit().fit(fit[0], y, task="binary", n_classes=2)
            stack = [lr.predict(p)[:, 1] for p, _, _ in splits]
        record("stack_logit", stack)
        record("bayes", [part[2] for part in splits])
        for gp_seed in (131, 132, 133):
            a, b, elapsed, k = searched(x, y, val[0][:, :, 1], val[1], gp_seed)
            for name, tree in [("gp_fit", a), ("gp_validation", b)]:
                probs = [
                    tree.do_pred_on_another_tensors(
                        current_tensors={f"p{i}.npy": p[:, i, 1:2] for i in range(8)}
                    ).ravel()
                    for p, _, _ in splits
                ]
                assert brier(probs[0], y) >= optimum - 1e-13
                record(name, probs, gp_seed, tree.nodes_count, elapsed, k)
    return rows


def run(out, seeds=range(3400, 3408), workers=1):
    out = Path(out)
    out.mkdir(exist_ok=True, parents=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        df = pd.DataFrame([row for batch in pool.map(panel, seeds) for row in batch])
    df.to_csv(out / "runs.csv", index=False)
    names = [
        "fit",
        "validation",
        "iid",
        "shift",
        "fit_minus_lattice_optimum",
        "exact_terms",
        "exact_literals",
        "fit_seconds",
        "nodes",
        "visited",
    ]
    df.groupby(["n_fit", "method", "data_seed"])[names].mean().groupby(
        ["n_fit", "method"]
    ).mean().to_csv(out / "means.csv")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=1)
    args = p.parse_args()
    run(args.out, workers=args.workers)

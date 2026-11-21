"""Native OKAPI search and exact one-gate fit on matched known label laws."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

from fusionbench.eval.exact_switch import fit_gate
from fusionbench.eval.search_continuous import brier
from fusionbench.eval.structural_key_audit import EndOfBudget, training_files
from fusionbench.methods.averaging import TopN
from fusionbench.methods.stacking import StackingLogit
from fusionbench.scenarios.native_switch import draw_native_switch, farther


def serialize(node):
    """Persist exact native parameters and ordered children, including repeated operands."""
    result = {
        "type": type(node).__name__,
        "children": [serialize(c) for c in node.children],
    }
    if hasattr(node, "id"):
        result["id"] = node.id
    if hasattr(node, "weights"):
        result["weights"] = np.asarray(node.weights).tolist()
    if hasattr(node, "threshold"):
        result["threshold"] = float(node.threshold)
        result["close"] = bool(node.close)
    return result


def tensors(x):
    return {f"p{i}.npy": x[:, i : i + 1] for i in range(x.shape[1])}


def search(paths, gt, validation, validation_y, seed, use_thresholds):
    from okapi import Okapi
    from okapi.operators import (
        CLOSE_THRESHOLD,
        FAR_THRESHOLD,
        MAX,
        MEAN,
        MIN,
        WEIGHTED_MEAN,
    )

    ops = (MEAN, MIN, MAX, WEIGHTED_MEAN)
    if use_thresholds:
        ops += (FAR_THRESHOLD, CLOSE_THRESHOLD)
    archive = {}
    records = []
    calls = 0
    loss = None
    start = time.perf_counter()

    def objective(p, y):
        nonlocal loss
        loss = brier(p.ravel(), y.ravel())
        return 1 - loss

    class Observed(Okapi):
        def _calculate_fitnesses(self, trees=None):
            nonlocal calls
            results = []
            for tree in self.population if trees is None else trees:
                row = super()._calculate_fitnesses([tree])[0]
                calls += 1
                key = tree.structural_key()
                if key not in archive:
                    candidate = tree.copy()
                    v = candidate.do_pred_on_another_tensors(
                        current_tensors=validation
                    ).ravel()
                    archive[key] = loss, brier(v, validation_y), candidate
                if calls in (300, 1200):
                    fit = min(archive.values(), key=lambda e: e[0])[2]
                    val = min(archive.values(), key=lambda e: (e[1], e[0]))[2]
                    records.append(
                        (calls, len(archive), time.perf_counter() - start, fit, val)
                    )
                    if calls == 1200:
                        raise EndOfBudget
                results.append(row)
            return np.stack(results)

    gp = Observed(
        paths,
        gt,
        len(paths),
        2,
        2,
        backend="numpy",
        seed=seed,
        allowed_ops=ops,
        postprocessing_function=np.asarray,
        objective_functions=(objective,),
        reuse_survivor_fitness=False,
    )
    try:
        while True:
            gp.run_iteration()
    except EndOfBudget:
        pass
    assert calls == 1200 and len(records) == 2
    return records


def record_predictions(rows, seed, regime, parts, ys, qs, method, pred, **metadata):
    for (split, _, _), p, y, q in zip(parts, pred, ys, qs, strict=True):
        rows.append(
            dict(
                data_seed=seed,
                fit_regime=regime,
                method=method,
                split=split,
                observed_brier=brier(p, y),
                conditional_excess=brier(p, q),
                **metadata,
            )
        )


def panel(seed):
    from loguru import logger

    logger.disable("okapi")
    fit, val, test = [
        draw_native_switch(seed, i, n) for i, n in enumerate((512, 256, 8192))
    ]
    signal = [fit.parent, fit.child]
    support = np.zeros((63**2, 6))
    a, b = np.meshgrid(np.arange(1, 64) / 64, np.arange(1, 64) / 64)
    support[:, fit.parent], support[:, fit.child] = a.ravel(), b.ravel()
    teacher = farther(a.ravel(), b.ravel(), 27 / 64)
    rows = []
    for regime, mixing in (("switch", 0), ("linear", 1)):
        parts = [
            ("fit", fit, mixing),
            ("validation", val, mixing),
            ("switch_test", test, 0),
            ("weak_test", test, 0.5),
            ("linear_test", test, 1),
        ]
        xs = [d.preds[:, :, 1] for _, d, _ in parts]
        ys = [d.labels(mix) for _, d, mix in parts]
        qs = [d.probability(mix) for _, d, mix in parts]

        record = partial(record_predictions, rows, seed, regime, parts, ys, qs)
        record("known_q", qs)
        record("average", [x.mean(1) for x in xs])
        top = TopN("brier").fit(fit.preds, ys[0], task="binary", n_classes=2)
        record(
            "top_n",
            [top.predict(d.preds)[:, 1] for _, d, _ in parts],
            selected_n=top.n_selected_,
            ranking=json.dumps(top.ranking_.tolist()),
            prefix_scores=json.dumps(top.prefix_scores_.tolist()),
        )
        lr = StackingLogit().fit(fit.preds, ys[0], task="binary", n_classes=2)
        record("stack_logit", [lr.predict(d.preds)[:, 1] for _, d, _ in parts])
        for method, fixed in (
            ("gate_fixed_threshold", True),
            ("gate_free_threshold", False),
        ):
            start = time.perf_counter()
            gate = fit_gate(xs[0], ys[0], fixed_threshold=fixed)
            elapsed = time.perf_counter() - start
            gate_preds = [gate.predict(x) for x in xs]
            assert (
                brier(gate_preds[0], ys[0])
                == gate.loss_numerator / gate.loss_denominator
            )
            pair_match = {gate.parent, gate.child} == set(signal)
            record(
                method,
                gate_preds,
                fit_seconds=elapsed,
                program=json.dumps(asdict(gate)),
                nodes=3,
                roles_match=pair_match,
                signal_grid_match=np.mean(gate.predict(support) == teacher)
                if pair_match
                else None,
            )
        binding = [tensors(x) for x in xs[:3]]
        with training_files(xs[0], ys[0]) as (paths, gt):
            for gp_seed in (251, 257, 263):
                for method, enabled in (("gp_four", False), ("gp_six", True)):
                    runs = search(paths, gt, binding[1], ys[1], gp_seed, enabled)
                    for calls, archive, elapsed, fit_tree, val_tree in runs:
                        for selection, tree in (
                            ("fit", fit_tree),
                            ("validation", val_tree),
                        ):
                            program = serialize(tree.root)
                            pp = [
                                tree.do_pred_on_another_tensors(
                                    current_tensors=m
                                ).ravel()
                                for m in binding
                            ]
                            nodes = tree.root.get_nodes()
                            role_set = {
                                int(str(n.id)[1:-4]) for n in nodes if hasattr(n, "id")
                            }
                            uses_gate = any(hasattr(n, "threshold") for n in nodes)
                            pair_match = role_set == set(signal)
                            grid_match = None
                            if pair_match:
                                pred_grid = tree.do_pred_on_another_tensors(
                                    current_tensors=tensors(support)
                                ).ravel()
                                grid_match = float(np.mean(pred_grid == teacher))
                            record(
                                method,
                                [pp[0], pp[1], pp[2], pp[2], pp[2]],
                                gp_seed=gp_seed,
                                calls=calls,
                                selection=selection,
                                archive_trees=archive,
                                fit_seconds=elapsed,
                                nodes=tree.nodes_count,
                                uses_gate=uses_gate,
                                roles_match=pair_match,
                                signal_grid_match=grid_match,
                                program=json.dumps(program),
                            )
    return rows


def summarize(rows):
    keys = ["fit_regime", "method", "calls", "selection", "split"]
    clean = rows.copy()
    clean["calls"] = clean.calls.fillna(0)
    clean["selection"] = clean.selection.fillna("fit")
    metrics = [
        "observed_brier",
        "conditional_excess",
        "nodes",
        "archive_trees",
        "fit_seconds",
    ]
    result = clean.groupby([*keys, "data_seed"])[metrics].mean().groupby(keys).mean()
    by_seed = clean.groupby([*keys, "data_seed"])[
        ["observed_brier", "conditional_excess"]
    ].mean()
    effects = []

    def compare(regime, method1, method2, calls1, calls2, selection, split):
        p = by_seed.xs((regime, method1, calls1, selection, split), level=keys)
        q = by_seed.xs((regime, method2, calls2, selection, split), level=keys)
        for metric in p:
            d = p[metric] - q[metric]
            assert len(d) == 12
            center = float(d.mean())
            half = t.ppf(0.975, len(d) - 1) * d.sem()
            effects.append(
                {
                    "fit_regime": regime,
                    "method1": method1,
                    "method2": method2,
                    "calls": calls1,
                    "selection": selection,
                    "split": split,
                    "metric": metric,
                    "difference": center,
                    "low_95": center - half,
                    "high_95": center + half,
                }
            )

    for reg in ("switch", "linear"):
        for split in clean.split.unique():
            for selection in ("fit", "validation"):
                for calls in (300, 1200):
                    compare(reg, "gp_six", "gp_four", calls, calls, selection, split)
            for alternative in ("stack_logit", "gate_fixed_threshold"):
                compare(reg, "gate_free_threshold", alternative, 0, 0, "fit", split)
    return result.reset_index(), pd.DataFrame(effects)


def add_gate_gaps(rows):
    """Signed fit shortfall vs the empirical one-gate optimum, for each selected model."""
    keys = ["data_seed", "fit_regime"]
    fit = rows[rows.split == "fit"]
    gate = fit[fit.method == "gate_free_threshold"][keys + ["observed_brier"]].rename(
        columns={"observed_brier": "one_gate_fit_brier"}
    )
    selected = keys + ["method", "gp_seed", "calls", "selection"]
    fitted = fit.merge(gate, on=keys, validate="many_to_one")
    fitted["fit_minus_one_gate"] = (
        fitted["observed_brier"] - fitted["one_gate_fit_brier"]
    )
    return rows.merge(
        fitted[selected + ["fit_minus_one_gate"]], on=selected, validate="many_to_one"
    )


def run(out, workers):
    out.mkdir(exist_ok=True, parents=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = pd.DataFrame(
            [r for batch in pool.map(panel, range(3700, 3712)) for r in batch]
        )
    rows = add_gate_gaps(rows)
    rows.to_csv(out / "runs.csv.gz", index=False)
    means, effects = summarize(rows)
    means.to_csv(out / "means.csv", index=False)
    effects.to_csv(out / "paired.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    run(args.out, args.workers)

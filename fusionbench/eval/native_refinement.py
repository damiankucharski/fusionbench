"""Evaluate native parameter edits against cost-matched continuation of the same GP.

Kommenda et al. (2020) studied local fitting of continuous constants in symbolic
regression. This experiment changes a threshold in an OKAPI tree and compares
its held-out gain with further search at matched cost.
See kommenda2020parameter in docs/ensemble-study/literature/README.md.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

from fusionbench.eval.continued_gp import continued_gp
from fusionbench.eval.exact_switch import fit_gate
from fusionbench.eval.gate_events import fit_threshold
from fusionbench.eval.native_switch import record_predictions, serialize, tensors
from fusionbench.eval.search_continuous import brier
from fusionbench.eval.structural_key_audit import training_files
from fusionbench.methods.averaging import TopN
from fusionbench.methods.stacking import StackingLogit
from fusionbench.scenarios.native_switch import draw_native_switch


@dataclass
class Edit:
    tree: object
    forwards: int
    seconds: float
    candidates: int = 0
    events: int = 0
    row_events: int = 0
    solver_mode: str = ""
    index: int | None = None
    old_threshold: float | None = None
    new_threshold: float | None = None


def native_brier(p, y):
    return brier(np.ravel(p), np.ravel(y))


def refine(tree, providers, y):
    from okapi.interventions import operator_paths
    from okapi.refinement import refine_provider_assignments
    from okapi.threshold_refinement import refine_threshold_coordinate

    def threshold(source):
        positions = [
            i for i, node in enumerate(source.op_nodes) if hasattr(node, "threshold")
        ]
        if not positions:
            return Edit(source, 0, 0.0)
        i = positions[0]
        path = next(
            path for path, node in operator_paths(source) if node is source.op_nodes[i]
        )
        start = time.perf_counter()
        r = refine_threshold_coordinate(
            source, providers, y, operator_index=i, fit_threshold=fit_threshold
        )
        sol = r.solver_result
        return Edit(
            r.tree,
            r.full_tree_evaluations,
            time.perf_counter() - start,
            sol.n_thresholds,
            sol.n_events,
            sol.n_row_events,
            sol.method,
            i,
            float(source.op_nodes[i].threshold),
            float(dict(operator_paths(r.tree))[path].threshold),
        )

    start = time.perf_counter()
    op = refine_provider_assignments(
        tree,
        providers,
        y,
        native_brier,
        max_evaluations=1 + len(tree.value_nodes) * (len(providers) - 1),
    )
    operand = Edit(op.tree, op.evaluations, time.perf_counter() - start)
    gate = threshold(tree)
    combined = threshold(operand.tree)
    combined.seconds += operand.seconds
    combined.forwards += operand.forwards
    return {"operand": operand, "threshold": gate, "combined": combined}


def run_gp(seed, provider_paths, gt, parts, ys, record):
    arrays = [part[1].preds[:, :, 1] for part in parts[:3]]
    bindings = [tensors(x) for x in arrays]
    edits = {}

    def callback(fit_tree, val_tree):
        updates = refine(fit_tree, bindings[0], ys[0])
        edits.update(updates)
        return {
            cost
            for edit in updates.values()
            for cost in (edit.forwards, edit.forwards + edit.candidates)
        }

    checkpoints = continued_gp(provider_paths, gt, bindings[1], ys[1], seed, callback)
    paths = {}

    def predictions(tree):
        key = tree.structural_key()
        if key not in paths:
            pred = [
                tree.do_pred_on_another_tensors(current_tensors=b).ravel()
                for b in bindings
            ]
            paths[key] = [pred[0], pred[1], pred[2], pred[2], pred[2]]
        return paths[key]

    base = checkpoints[1200]
    incumbent = base.fit_tree
    raw = predictions(incumbent)
    original_val = brier(raw[1], ys[1])
    has_gate = any(hasattr(n, "threshold") for n in incumbent.op_nodes)

    def score_champion(method, arm, tree, calls, **metadata):
        p = predictions(tree)
        improve_val = brier(p[1], ys[1]) < original_val
        for rule, champ, pp in (
            ("fit", tree, p),
            (
                "validation_guard",
                tree if improve_val else incumbent,
                p if improve_val else raw,
            ),
        ):
            record(
                method,
                pp,
                gp_seed=seed,
                arm=arm,
                decision=rule,
                calls=calls,
                nodes=champ.nodes_count,
                initial_has_threshold=has_gate,
                validation_accepts=improve_val,
                changed=champ.structural_key() != incumbent.structural_key(),
                program=json.dumps(serialize(champ.root)),
                **metadata,
            )

    score_champion(
        "gp_initial",
        "none",
        incumbent,
        1200,
        seconds=base.elapsed_seconds,
        n_archive=base.n_trees,
    )
    for arm, edit in edits.items():
        metadata = {
            field: getattr(edit, field)
            for field in (
                "forwards",
                "candidates",
                "events",
                "row_events",
                "solver_mode",
                "index",
                "old_threshold",
                "new_threshold",
            )
        }
        score_champion(
            "refined", arm, edit.tree, 1200, seconds=edit.seconds, **metadata
        )
        assert brier(predictions(edit.tree)[0], ys[0]) <= brier(raw[0], ys[0]) + 2e-15
        for name, extra in (
            ("gp_matched_forwards", edit.forwards),
            ("gp_matched_candidates", edit.forwards + edit.candidates),
        ):
            cp = checkpoints[1200 + extra]
            score_champion(
                name,
                arm,
                cp.fit_tree,
                1200 + extra,
                seconds=cp.elapsed_seconds - base.elapsed_seconds,
                n_archive=cp.n_trees,
                **metadata,
            )


def panel(data_seed):
    from loguru import logger

    logger.disable("okapi")
    fit, val, test = [
        draw_native_switch(data_seed, i, n) for i, n in enumerate((512, 256, 8192))
    ]
    rows = []
    for regime, mix in (("switch", 0), ("linear", 1)):
        parts = [
            ("fit", fit, mix),
            ("validation", val, mix),
            ("switch_test", test, 0),
            ("weak_test", test, 0.5),
            ("linear_test", test, 1),
        ]
        ys = [d.labels(k) for _, d, k in parts]
        qs = [d.probability(k) for _, d, k in parts]
        xs = [d.preds[:, :, 1] for _, d, _ in parts]
        record = partial(record_predictions, rows, data_seed, regime, parts, ys, qs)
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
        stack = StackingLogit().fit(fit.preds, ys[0], task="binary", n_classes=2)
        record("stack_logit", [stack.predict(d.preds)[:, 1] for _, d, _ in parts])
        gate = fit_gate(xs[0], ys[0])
        record(
            "gate_free_threshold",
            [gate.predict(x) for x in xs],
            nodes=3,
            program=json.dumps(asdict(gate)),
        )
        with training_files(xs[0], ys[0]) as (paths, gt):
            for seed in (271, 277):
                run_gp(seed, paths, gt, parts, ys, record)
    return rows


def summarize(rows):
    numeric = [
        "observed_brier",
        "conditional_excess",
        "nodes",
        "calls",
        "forwards",
        "candidates",
        "events",
        "row_events",
        "seconds",
        "changed",
        "validation_accepts",
    ]
    work = rows.assign(
        arm=rows.arm.fillna("none"), decision=rows.decision.fillna("fit")
    )
    keys = ["fit_regime", "split", "decision", "arm", "method"]
    means = (
        work.groupby([*keys, "data_seed"])[numeric]
        .mean()
        .groupby(keys)
        .mean()
        .reset_index()
    )
    groups = ["fit_regime", "split", "decision", "arm"]
    results = []
    for group, refined in work[work.method == "refined"].groupby(groups):
        reg, split, decision, arm = group
        match = work[
            (work.fit_regime == reg)
            & (work.split == split)
            & (work.decision == decision)
        ]
        for baseline in ("gp_initial", "gp_matched_forwards", "gp_matched_candidates"):
            comparator = match[
                (match.method == baseline)
                & (match.arm == ("none" if baseline == "gp_initial" else arm))
            ]
            ix = ["data_seed", "gp_seed"]
            r = refined.set_index(ix)
            q = comparator.set_index(ix)
            for metric in ("observed_brier", "conditional_excess"):
                diff = (r[metric] - q[metric]).groupby("data_seed").mean()
                assert len(diff) == 12
                mean = diff.mean()
                half = t.ppf(0.975, len(diff) - 1) * diff.sem()
                results.append(
                    {
                        "fit_regime": reg,
                        "split": split,
                        "decision": decision,
                        "arm": arm,
                        "comparator": baseline,
                        "metric": metric,
                        "difference": mean,
                        "low_95": mean - half,
                        "high_95": mean + half,
                    }
                )
    return means, pd.DataFrame(results)


def run(out, workers):
    out.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = pd.DataFrame(
            [r for group in pool.map(panel, range(3800, 3812)) for r in group]
        )
    rows.to_csv(out / "runs.csv.gz", index=False)
    means, paired = summarize(rows)
    means.to_csv(out / "means.csv", index=False)
    paired.to_csv(out / "paired.csv", index=False)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--out", type=Path, required=True)
    cli.add_argument("--workers", type=int, default=1)
    args = cli.parse_args()
    run(args.out, args.workers)

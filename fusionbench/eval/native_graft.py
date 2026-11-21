"""Paired native gate insertion, archive-only proposal and ordinary search at matched expense.

The structure-versus-parameter contrast extends the native switch study. Kommenda
et al. (2020), doi:10.1007/s10710-019-09371-3, studied local constant fitting in
symbolic regression. Its scope and our separate structural hypothesis are
documented in docs/ensemble-study/literature/2026-11-21.md and the study report.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.stats import t

from fusionbench.eval.exact_switch import fit_gate
from fusionbench.eval.gate_events import fit_threshold
from fusionbench.eval.gate_proposal_gp import (
    ProposalPlan,
    probe_and_continue,
    replay_insertion,
)
from fusionbench.eval.native_switch import record_predictions, serialize, tensors
from fusionbench.eval.search_continuous import brier
from fusionbench.eval.structural_key_audit import training_files
from fusionbench.methods.averaging import TopN
from fusionbench.methods.stacking import StackingLogit
from fusionbench.scenarios.native_switch import draw_native_switch

ARMS = (
    ("first_fixed", False, False),
    ("all_fixed", True, False),
    ("first_tuned", False, True),
    ("all_tuned", True, True),
)


def measure_search(paths, gt, data_seed, gp_seed, splits, labels, record):
    from okapi.threshold_graft import propose_threshold_graft

    bindings = [tensors(part.preds[:, :, 1]) for _, part, _ in splits[:3]]
    fit_y, val_y = labels[:2]
    draw_seed = int(
        np.random.SeedSequence([data_seed, gp_seed, 937]).generate_state(1)[0]
    )
    proposals, timings = {}, {}

    def propose(incumbent, validation_champion, checkpoint):
        offsets = set()
        for arm, all_inputs, tune in ARMS:
            start = perf_counter()
            result = propose_threshold_graft(
                incumbent,
                bindings[0],
                fit_y,
                seed=draw_seed,
                search_operands=all_inputs,
                fit_threshold=fit_threshold if tune else None,
            )
            timings[arm] = perf_counter() - start
            proposals[arm] = result
            s = (len(bindings[0]) - 1) if all_inputs else 1
            assert result.full_tree_evaluations == 1 + (8 if tune else 2) * s
            assert result.n_proposals == 2 * s
            assert result.loss <= result.initial_loss
            f = result.full_tree_evaluations
            total = f + result.threshold_candidates
            offsets.update((f, total))
            if arm == "all_tuned":
                offsets.update((f + 300, total + 300))
        full = proposals["all_tuned"]
        assert (
            len(
                {
                    (p.site_path, p.anchor_id, p.operand_order)
                    for p in proposals.values()
                }
            )
            == 1
        )
        return ProposalPlan(
            tree=full.tree, fit_brier=full.loss, offsets=sorted(offsets)
        )

    probe = probe_and_continue(paths, gt, bindings[1], val_y, gp_seed, propose)
    active = replay_insertion(paths, gt, bindings[1], val_y, gp_seed, probe)
    base = probe.base
    full = proposals["all_tuned"]
    assert all(abs(r.initial_loss - base.fit_brier) < 2e-15 for r in proposals.values())
    if not active.inserted:
        assert (
            active.checkpoint.path_digest == probe.plain[base.calls + 300].path_digest
        )

    signature = lambda tree: tree.structural_key()
    init_key = signature(base.fit_tree)
    initially_gated = any(hasattr(n, "threshold") for n in base.fit_tree.op_nodes)
    # Hidden roles are read only after both optimizers and every proposal have finished.
    site = splits[0][1]
    hidden = {f"p{site.parent}.npy", f"p{site.child}.npy"}
    cache = {}

    def predictions(tree):
        key = signature(tree)
        if key not in cache:
            p = [
                tree.do_pred_on_another_tensors(current_tensors=b).ravel()
                for b in bindings
            ]
            cache[key] = [p[0], p[1], p[2], p[2], p[2]]
        return cache[key]

    def report(
        method, arm, trees, native_calls, *, checkpoint=None, propose_used=False
    ):
        result = proposals[arm] if arm != "none" else None
        chosen = None if result is None else result.selected_operand
        f = 0 if result is None else result.full_tree_evaluations
        npoints = 0 if result is None else result.threshold_candidates
        common = {
            "gp_seed": gp_seed,
            "arm": arm,
            "base_calls": base.calls,
            "base_generation": base.generation,
            "ordinary_calls": native_calls,
            "initial_has_gate": initially_gated,
            "total_forwards": native_calls + (f if propose_used else 0),
            "total_candidates": native_calls + (f + npoints if propose_used else 0),
        }
        if checkpoint is not None:
            common.update(
                observed_generation=checkpoint.generation,
                archive_trees=checkpoint.n_trees,
                path_sha256=checkpoint.path_digest,
                search_seconds=checkpoint.elapsed_seconds,
            )
        if result is not None:
            common.update(
                graft_seed=draw_seed,
                site_path=json.dumps(result.site_path),
                anchor=result.anchor_id,
                operand_order=json.dumps(result.operand_order),
                selected_operand=chosen,
                direction=result.selected_direction,
                threshold=result.selected_threshold,
                graft_changed=signature(result.tree) != init_key,
                anchor_is_signal=result.anchor_id in hidden,
                graft_signal_pair=({result.anchor_id, chosen} == hidden)
                if chosen is not None
                else None,
                grafts=result.n_proposals,
                proposal_seconds=timings[arm],
                proposal_forwards=f,
                threshold_candidates=npoints,
                events=result.solver_events,
                row_events=result.solver_row_events,
                graft_fit_loss=result.loss,
            )
        if method in ("active_tail", "archive_only_tail"):
            common.update(
                graft_status=active.status,
                full_graft_in_survivors=active.survived,
                full_graft_inserted=active.inserted,
                active_path_changed=active.checkpoint.path_digest
                != probe.plain[base.calls + 300].path_digest,
            )
        for selection, tree in zip(("fit", "validation"), trees, strict=True):
            key = signature(tree)
            record(
                method,
                predictions(tree),
                decision=selection,
                nodes=tree.nodes_count,
                program=json.dumps(serialize(tree.root)),
                winner_changed=key != init_key,
                winner_is_full_offer=key == signature(full.tree),
                **common,
            )

    def winners(cp):
        return cp.fit_tree, cp.validation_tree

    report("gp_initial", "none", winners(base), base.calls, checkpoint=base)
    for arm, _, _ in ARMS:
        result = proposals[arm]
        # An immediate proposal makes only the returned tree eligible, with the original archive retained.
        val_winner = base.validation_tree
        new_key = (brier(predictions(result.tree)[1], val_y), result.loss)
        current_key = (
            brier(predictions(val_winner)[1], val_y),
            brier(predictions(val_winner)[0], fit_y),
        )
        if new_key < current_key:
            val_winner = result.tree
        report(
            "immediate", arm, (result.tree, val_winner), base.calls, propose_used=True
        )
        for method, offset in (
            ("matched_forwards", result.full_tree_evaluations),
            (
                "matched_candidates",
                result.full_tree_evaluations + result.threshold_candidates,
            ),
        ):
            cp = probe.plain[base.calls + offset]
            report(method, arm, winners(cp), cp.calls, checkpoint=cp)
    tail = probe.plain[base.calls + 300]
    report("plain_tail", "none", winners(tail), tail.calls, checkpoint=tail)
    report(
        "archive_only_tail",
        "all_tuned",
        winners(probe.sham),
        probe.sham.calls,
        checkpoint=probe.sham,
        propose_used=True,
    )
    report(
        "active_tail",
        "all_tuned",
        winners(active.checkpoint),
        active.checkpoint.calls,
        checkpoint=active.checkpoint,
        propose_used=True,
    )
    for method, offset in (
        ("matched_forwards_tail", full.full_tree_evaluations),
        (
            "matched_candidates_tail",
            full.full_tree_evaluations + full.threshold_candidates,
        ),
    ):
        cp = probe.plain[base.calls + offset + 300]
        report(method, "all_tuned", winners(cp), cp.calls, checkpoint=cp)


def panel(data_seed):
    from loguru import logger

    logger.disable("okapi")
    fit, val, test = [
        draw_native_switch(data_seed, i, n) for i, n in enumerate((512, 256, 8192))
    ]
    rows = []
    for regime, mix in (("switch", 0), ("linear", 1)):
        splits = [
            ("fit", fit, mix),
            ("validation", val, mix),
            ("switch_test", test, 0),
            ("weak_test", test, 0.5),
            ("linear_test", test, 1),
        ]
        ys = [d.labels(k) for _, d, k in splits]
        qs = [d.probability(k) for _, d, k in splits]
        xs = [d.preds[:, :, 1] for _, d, _ in splits]
        record = partial(record_predictions, rows, data_seed, regime, splits, ys, qs)
        record("known_q", qs)
        record("average", [x.mean(1) for x in xs])
        top = TopN("brier").fit(fit.preds, ys[0], task="binary", n_classes=2)
        record(
            "top_n",
            [top.predict(d.preds)[:, 1] for _, d, _ in splits],
            selected_n=top.n_selected_,
            ranking=json.dumps(top.ranking_.tolist()),
            prefix_scores=json.dumps(top.prefix_scores_.tolist()),
        )
        stack = StackingLogit().fit(fit.preds, ys[0], task="binary", n_classes=2)
        record("stack_logit", [stack.predict(d.preds)[:, 1] for _, d, _ in splits])
        exact = fit_gate(xs[0], ys[0])
        record(
            "gate_free_threshold",
            [exact.predict(x) for x in xs],
            nodes=3,
            program=json.dumps(asdict(exact)),
        )
        with training_files(xs[0], ys[0]) as (paths, gt):
            for gp_seed in (293, 307):
                measure_search(paths, gt, data_seed, gp_seed, splits, ys, record)
    return rows


def summaries(rows):
    keys = ["fit_regime", "split", "decision", "arm", "method"]
    metrics = [
        "observed_brier",
        "conditional_excess",
        "nodes",
        "base_calls",
        "ordinary_calls",
        "total_forwards",
        "total_candidates",
        "threshold_candidates",
        "proposal_seconds",
        "search_seconds",
        "archive_trees",
    ]
    work = rows.assign(
        arm=rows.arm.fillna("none"), decision=rows.decision.fillna("fit")
    )
    means = (
        work.groupby([*keys, "data_seed"])[metrics]
        .mean()
        .groupby(keys)
        .mean()
        .reset_index()
    )
    contrasts = [
        ("active_tail", "all_tuned", "matched_candidates_tail", "all_tuned"),
        ("active_tail", "all_tuned", "matched_forwards_tail", "all_tuned"),
        ("active_tail", "all_tuned", "archive_only_tail", "all_tuned"),
        ("archive_only_tail", "all_tuned", "plain_tail", "none"),
        ("active_tail", "all_tuned", "gp_initial", "none"),
        ("immediate", "all_tuned", "immediate", "all_fixed"),
        ("immediate", "all_tuned", "immediate", "first_tuned"),
        ("immediate", "all_fixed", "immediate", "first_fixed"),
        ("immediate", "first_tuned", "immediate", "first_fixed"),
    ]
    for arm, _, _ in ARMS:
        contrasts.extend(
            [
                ("immediate", arm, "matched_candidates", arm),
                ("immediate", arm, "matched_forwards", arm),
            ]
        )
    result = []
    for (regime, split, decision), group in work.groupby(keys[:3]):
        groups = group.set_index(["method", "arm", "data_seed", "gp_seed"])
        for m1, a1, m2, a2 in contrasts:
            left = groups.xs((m1, a1), level=("method", "arm"))
            right = groups.xs((m2, a2), level=("method", "arm"))
            for metric in ("observed_brier", "conditional_excess"):
                paired = left[metric] - right[metric]
                assert len(paired) == 24 and not paired.isna().any()
                average = paired.groupby("data_seed").mean()
                center = average.mean()
                half = t.ppf(0.975, len(average) - 1) * average.sem()
                result.append(
                    {
                        "fit_regime": regime,
                        "split": split,
                        "decision": decision,
                        "method1": m1,
                        "arm1": a1,
                        "method2": m2,
                        "arm2": a2,
                        "metric": metric,
                        "difference": center,
                        "low_95": center - half,
                        "high_95": center + half,
                    }
                )
    return means, pd.DataFrame(result)


def run(out, workers):
    out.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = pd.DataFrame(
            [r for unit in pool.map(panel, range(3900, 3912)) for r in unit]
        )
    rows.to_csv(out / "runs.csv.gz", index=False)
    means, paired = summaries(rows)
    means.to_csv(out / "means.csv", index=False)
    paired.to_csv(out / "paired.csv", index=False)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--out", type=Path, required=True)
    cli.add_argument("--workers", type=int, default=1)
    args = cli.parse_args()
    run(args.out, args.workers)

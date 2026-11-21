"""Slot-matched tests of structure, threshold fitting and functionally neutral variation.

Beadle and Johnson's Boolean-equivalence control (2008) and Kommenda et al.'s
constant optimization (2020) motivate distinctions here. The identity-control
construction and its use after actual graft admission are experimental hypotheses.
See docs/ensemble-study/literature/README.md for the source scopes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.stats import t

from fusionbench.eval.exact_switch import fit_gate
from fusionbench.eval.gate_events import fit_threshold
from fusionbench.eval.matched_parent_gp import (
    ARMS,
    OFFERS,
    MatchedPlan,
    matched_parent_gp,
)
from fusionbench.eval.native_switch import serialize, tensors
from fusionbench.eval.search_continuous import brier
from fusionbench.eval.structural_key_audit import training_files
from fusionbench.methods.averaging import TopN
from fusionbench.methods.stacking import StackingLogit
from fusionbench.scenarios.native_switch import draw_native_switch


def data_panel(seed):
    draws = [draw_native_switch(seed, i, n) for i, n in enumerate((512, 256, 8192))]
    near = [replace(d, close=True, threshold=37 / 64) for d in draws]
    return {"far": (draws, 0.0), "close": (near, 0.0), "linear": (draws, 1.0)}


def regime_splits(panel, regime):
    draws, mix = panel[regime]
    return [
        ("fit", draws[0], mix),
        ("validation", draws[1], mix),
        *[(name + "_test", data[2], k) for name, (data, k) in panel.items()],
    ]


def fit_proposals(incumbent, sources, y, draw_seed, store, marker=True):
    from okapi.gate_lineage import gate_descendants, mark_gate_origin
    from okapi.threshold_graft import propose_threshold_graft, threshold_graft_variant

    start = perf_counter()
    full = propose_threshold_graft(
        incumbent,
        sources,
        y,
        seed=draw_seed,
        search_operands=True,
        fit_threshold=fit_threshold,
    )
    assert full.full_tree_evaluations == 41 and full.n_proposals == 10
    offers = {OFFERS[0]: (full.tree, full.loss)}
    improved = full.tree.structural_key() != incumbent.structural_key()
    extra = 0
    for arm in OFFERS[1:]:
        if improved:
            candidate = threshold_graft_variant(
                full.tree,
                full.site_path,
                reset_threshold=arm.endswith("half"),
                duplicate_anchor=arm.startswith("duplicate"),
            )
            prediction = candidate.predict().ravel()
            loss = brier(prediction, y)
            extra += 1
            if arm.startswith("duplicate"):
                assert loss == full.initial_loss
        else:
            candidate, loss = full.tree, full.loss
        offers[arm] = (candidate, loss)
    store.update(
        result=full, offers=offers, extra=extra, proposal_seconds=perf_counter() - start
    )

    def mark(tree, arm):
        return mark_gate_origin(tree, (*full.site_path, 0), token="offered")

    def tracked(tree):
        occurrences = gate_descendants(tree, token="offered")
        exact = sum(
            o.threshold == full.selected_threshold
            and o.direction == full.selected_direction
            for o in occurrences
        )
        return len(occurrences), exact

    charged = full.full_tree_evaluations + extra
    offsets = [300, charged + 300, charged + full.threshold_candidates + 300]
    return MatchedPlan(
        offers,
        offsets,
        mark if marker else None,
        tracked if marker else None,
        partial(gate_descendants, token="offered") if marker else None,
    )


def measure(paths, gt, data_seed, gp_seed, splits, ys, record, *, marker=True):
    from okapi.gate_lineage import gate_descendants

    # The three test label laws share one feature matrix.
    bindings = [tensors(d.preds[:, :, 1]) for _, d, _ in splits[:3]]
    draw_seed = int(
        np.random.SeedSequence([data_seed, gp_seed, 937]).generate_state(1)[0]
    )
    store = {}

    def callback(fit_tree, validation_tree, checkpoint):
        return fit_proposals(
            fit_tree, bindings[0], ys[0], draw_seed, store, marker=marker
        )

    res = matched_parent_gp(paths, gt, bindings[1], ys[1], gp_seed, callback)
    base = res.base
    result = store["result"]
    extra = store["extra"]
    f = result.full_tree_evaluations + extra
    tcount = result.threshold_candidates
    assert abs(result.initial_loss - base.fit_brier) < 2e-15
    admitted = res.admission.admitted
    start_states = {k: v.population_digest for k, v in res.states.items()}
    if not admitted:
        assert len(set(start_states.values())) == 1
        assert len({res.arms[k][base.calls + 300].path_digest for k in ARMS}) == 1
    if admitted:
        assert all(
            res.states[k].rng_after == res.admission.rng_after
            for k in (*OFFERS, "reorder")
        )
        assert res.states["archive_only"].rng_after == res.admission.rng_before
    hidden = {f"p{splits[0][1].parent}.npy", f"p{splits[0][1].child}.npy"}
    shared = {
        "gp_seed": gp_seed,
        "graft_seed": draw_seed,
        "base_calls": base.calls,
        "base_generation": base.generation,
        "original_has_gate": any(
            hasattr(n, "threshold") for n in base.fit_tree.op_nodes
        ),
        "original_nodes": base.fit_tree.nodes_count,
        "status": res.admission.status,
        "admitted": admitted,
        "slot": res.admission.slot,
        "displaced": res.admission.displaced,
        "parent_order": json.dumps(res.admission.parent_order),
        "rng_before": res.admission.rng_before,
        "rng_after_admission": res.admission.rng_after,
        "anchor": result.anchor_id,
        "site_path": json.dumps(result.site_path),
        "operand_order": json.dumps(result.operand_order),
        "chosen_partner": result.selected_operand,
        "direction": result.selected_direction,
        "fitted_threshold": result.selected_threshold,
        "signal_pair": {result.anchor_id, result.selected_operand} == hidden,
        "graft_forwards": result.full_tree_evaluations,
        "variant_forwards": extra,
        "threshold_candidates": tcount,
        "unique_events": result.solver_events,
        "row_events": result.solver_row_events,
        "proposal_seconds": store["proposal_seconds"],
    }
    cache = {}

    def predict(tree):
        k = tree.structural_key()
        if k not in cache:
            arr = [
                tree.do_pred_on_another_tensors(current_tensors=m).ravel()
                for m in bindings
            ]
            cache[k] = [arr[0], arr[1], arr[2], arr[2], arr[2]]
        return cache[k]

    def score(method, tree, decision, **meta):
        record(
            method,
            predict(tree),
            decision=decision,
            nodes=tree.nodes_count,
            program=json.dumps(serialize(tree.root)),
            winner_occurrences=json.dumps(
                [asdict(o) for o in gate_descendants(tree, token="offered")]
                if marker
                else []
            ),
            **shared,
            **meta,
        )

    def scores(method, cp, **meta):
        for name, tree in (("fit", cp.fit_tree), ("validation", cp.validation_tree)):
            score(
                method,
                tree,
                name,
                calls=cp.calls,
                search_seconds=cp.elapsed_seconds,
                archive_trees=cp.n_trees,
                path_sha256=cp.path_digest,
                **meta,
            )

    scores("gp_initial", base)
    for arm, (tree, loss) in store["offers"].items():
        assert brier(predict(tree)[0], ys[0]) == loss
        score(
            "immediate_" + arm,
            tree,
            "fit",
            calls=base.calls,
            free_admission=res.admission.marginal[arm],
        )
    for arm in ARMS:
        state = asdict(res.states[arm])
        lin = asdict(res.lineage[arm])
        assert lin["marked_evaluations"] <= 300
        if not admitted or arm in ("archive_only", "reorder"):
            assert lin["marked_evaluations"] == 0
        # Lineage flags refer throughout to the fit-selected winning program.
        scores(
            arm,
            res.arms[arm][base.calls + 300],
            free_admission=res.admission.marginal.get(arm),
            **{
                f"state_{k}": json.dumps(v) if isinstance(v, tuple) else v
                for k, v in state.items()
            },
            **{
                f"lineage_{k}": json.dumps(v) if isinstance(v, tuple) else v
                for k, v in lin.items()
            },
        )
    for name, offset in (
        ("plain_tail", 300),
        ("plain_forward_tail", f + 300),
        ("plain_candidate_tail", f + tcount + 300),
    ):
        scores(name, res.plain[base.calls + offset])


def record_predictions(rows, seed, regime, parts, ys, qs, method, pred, **metadata):
    for (split, _, _), p, y, q in zip(parts, pred, ys, qs, strict=True):
        vector = np.asarray(p, dtype="<f8").ravel()
        rows.append(
            dict(
                data_seed=seed,
                fit_regime=regime,
                split=split,
                method=method,
                observed_brier=brier(vector, y),
                conditional_excess=brier(vector, q),
                prediction_sha256=hashlib.sha256(vector.tobytes()).hexdigest(),
                **metadata,
            )
        )


def panel(seed, *, marker=True):
    from loguru import logger

    logger.disable("okapi")
    data = data_panel(seed)
    rows = []
    for regime in data:
        parts = regime_splits(data, regime)
        fit = parts[0][1]
        p = [d.preds for _, d, _ in parts]
        x = [m[:, :, 1] for m in p]
        ys = [d.labels(mix) for _, d, mix in parts]
        qs = [d.probability(mix) for _, d, mix in parts]
        record = partial(record_predictions, rows, seed, regime, parts, ys, qs)
        record("known_q", qs)
        record("average", [m.mean(1) for m in x])
        top = TopN("brier").fit(fit.preds, ys[0], task="binary", n_classes=2)
        record(
            "top_n",
            [top.predict(m)[:, 1] for m in p],
            n_selected=top.n_selected_,
            ranking=json.dumps(top.ranking_.tolist()),
            prefix_scores=json.dumps(top.prefix_scores_.tolist()),
        )
        lr = StackingLogit().fit(fit.preds, ys[0], task="binary", n_classes=2)
        record("stack_logit", [lr.predict(m)[:, 1] for m in p])
        exact = fit_gate(x[0], ys[0])
        record(
            "one_gate",
            [exact.predict(m) for m in x],
            nodes=3,
            program=json.dumps(asdict(exact)),
        )
        with training_files(x[0], ys[0]) as (paths, gt):
            for gp_seed in (311, 313):
                measure(paths, gt, seed, gp_seed, parts, ys, record, marker=marker)
    return rows


def summarize(rows):
    work = rows.assign(decision=rows.decision.fillna("fit"))
    keys = ["fit_regime", "split", "decision"]
    metrics = [
        "observed_brier",
        "conditional_excess",
        "nodes",
        "admitted",
        "calls",
        "threshold_candidates",
        "lineage_marked_evaluations",
        "lineage_marked_unique",
        "lineage_winner_ever_marked",
        "lineage_winner_first_marked",
    ]
    means = (
        work.groupby([*keys, "method", "data_seed"])[metrics]
        .mean()
        .groupby([*keys, "method"])
        .mean()
        .reset_index()
    )
    pairs = [
        (OFFERS[0], OFFERS[2]),
        (OFFERS[0], OFFERS[1]),
        (OFFERS[0], OFFERS[3]),
        (OFFERS[2], OFFERS[3]),
        (OFFERS[1], OFFERS[3]),
        (OFFERS[2], "reorder"),
        (OFFERS[3], "reorder"),
        ("reorder", "archive_only"),
        (OFFERS[0], "archive_only"),
        (OFFERS[0], "plain_candidate_tail"),
        (OFFERS[0], "plain_forward_tail"),
    ]
    out = []
    for (regime, split, decision), group in work.groupby(keys):
        g = group.set_index(["method", "data_seed", "gp_seed"])
        for a, b in pairs:
            left, right = g.xs(a), g.xs(b)
            for subset, index in (
                ("all", left.index),
                ("admitted", left[left.admitted == True].index),
            ):
                for metric in ("observed_brier", "conditional_excess"):
                    delta = (left[metric] - right[metric]).loc[index]
                    d = delta.groupby("data_seed").mean()
                    center = d.mean()
                    half = t.ppf(0.975, len(d) - 1) * d.sem() if len(d) > 1 else np.nan
                    out.append(
                        {
                            "fit_regime": regime,
                            "split": split,
                            "decision": decision,
                            "arm1": a,
                            "arm2": b,
                            "subset": subset,
                            "metric": metric,
                            "pairs": len(delta),
                            "banks": len(d),
                            "difference": center,
                            "low_95": center - half,
                            "high_95": center + half,
                        }
                    )
    return means, pd.DataFrame(out)


def run(out, workers, *, marker=True):
    out.mkdir(exist_ok=True, parents=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        frame = pd.DataFrame(
            [
                r
                for batch in pool.map(partial(panel, marker=marker), range(4100, 4116))
                for r in batch
            ]
        )
    frame.to_csv(out / "runs.csv.gz", index=False)
    means, contrasts = summarize(frame)
    means.to_csv(out / "means.csv", index=False)
    contrasts.to_csv(out / "paired.csv", index=False)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--out", type=Path, required=True)
    cli.add_argument("--workers", type=int, default=1)
    cli.add_argument("--without-lineage", action="store_true")
    opts = cli.parse_args()
    run(opts.out, opts.workers, marker=not opts.without_lineage)

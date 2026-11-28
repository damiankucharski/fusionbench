"""Independent paired replication of joint pooled/fold terminal selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from runpy import run_path
from time import perf_counter

import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits

from fusionbench.scenarios.checkpoint_bank import (
    generate_checkpoint_bank,
    generate_scene_pool,
    scene_sample_indices,
    select_checkpoints,
    signal_provider_index,
)

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "fold-promotion"
previous = run_path(str(PREVIOUS / "run.py"))
fit_search, score_search, inputs, digest, score_references = (
    previous[key]
    for key in ("fit_search", "score_search", "inputs", "digest", "score_references")
)
SEEDS = range(288100, 288148)
LAWS = ("equal", "signal")
SIZES = (24, 192)
PATHS = ("pooled", "fold")
METHODS = (
    "pooled_final",
    "anchor",
    *(f"{p}_{s}" for p in PATHS for s in ("pooled", "fold", "veto", "joint")),
    "mean",
    "top_n",
    "prob_lr",
    "regularized",
)
IDS = ["seed", "law", "n_scenes"]
COST_KEYS = ("objective_evaluations", "offspring_proposals", "unique_evaluated")


def score_bank(seed):
    """Fit both independent searches before exposing either path to audit data."""
    started = perf_counter()
    scores, traces = [], []
    search_seed = seed + 1_000_000
    manifest = {
        "seed": seed,
        "search_seed": search_seed,
        "group_seed": seed + 3_000_000,
        "cells": [],
    }
    for law in LAWS:
        selection = generate_checkpoint_bank(seed, law=law)["A"]
        slots = select_checkpoints(selection, 8)
        signal = signal_provider_index(seed) if law == "signal" else None
        pool = generate_scene_pool(
            seed, law=law, scene_error_share=1.0, n_fit_scenes=192, n_audit_scenes=512
        )
        fit, audit = pool["F"], pool["I"]
        fit_pool, audit_x = inputs(fit, slots), inputs(audit, slots)
        for n_scenes in SIZES:
            ids = scene_sample_indices(n_scenes, 8)
            x, y, groups = fit_pool[ids], fit.y[ids], fit.scene[ids]
            cell = {
                "seed": seed,
                "law": law,
                "scene_error_share": 1.0,
                "n_scenes": n_scenes,
                "rows_per_scene": 8,
                "n_rows": len(y),
            }
            fitted = {
                path: fit_search(
                    x,
                    y,
                    groups=groups,
                    data_seed=seed,
                    search_seed=search_seed,
                    path=path,
                    joint_study=True,
                )
                for path in PATHS
            }
            for key in ("anchor", "founder_digest", "fold_digest", "fold_counts"):
                if fitted["fold"][key] != fitted["pooled"][key]:
                    raise RuntimeError(
                        f"Mismatched {key} on paired native search paths"
                    )
            meta = {
                **cell,
                "slot_ids": slots.tolist(),
                "signal_provider": signal,
                "selection_A_x_sha256": digest(selection.P),
                "selection_A_y_sha256": digest(selection.y),
                "fit_x_sha256": digest(x),
                "fit_y_sha256": digest(y),
                "fit_q_sha256": digest(fit.posterior[ids]),
                "group_sha256": digest(groups),
                "audit_x_sha256": digest(audit_x),
                "audit_y_sha256": digest(audit.y),
                "audit_q_sha256": digest(audit.posterior),
                "audit_scene_sha256": digest(audit.scene),
                "paths": {},
            }
            for path, result in fitted.items():
                scored, trace = score_search(
                    result,
                    audit_x,
                    audit.y,
                    fit_q=fit.posterior[ids],
                    audit_q=audit.posterior,
                    signal_provider=signal,
                )
                scores.extend(
                    {**cell, **row, "n_providers": row["unique_providers"]}
                    for row in scored
                )
                traces.extend({**cell, **row} for row in trace)
                meta["paths"][path] = {
                    **{
                        k: scored[0][k]
                        for k in (
                            "founder_digest",
                            "anchor_digest",
                            "fold_digest",
                            "path_digest",
                            *COST_KEYS,
                        )
                    },
                    "fold_row_counts": result["fold_counts"],
                    "final_menu_count": len(
                        dict.fromkeys(
                            (result["anchor"], *result["snapshots"][-1]["keys"])
                        )
                    ),
                }
            results = score_references(
                x,
                y,
                audit_x,
                audit.y,
                seed=search_seed,
                audit_q=audit.posterior,
                signal_provider=signal,
            )
            scores.extend(
                {**cell, "method": name, **row, "unique_providers": row["n_providers"]}
                for name, row in results.items()
            )
            manifest["cells"].append(meta)
    manifest["worker_seconds"] = perf_counter() - started
    return scores, traces, manifest


def interval(differences, draws):
    diff = np.asarray(differences, dtype=float)
    lo, hi = np.quantile(diff[draws].mean(axis=1), [0.025, 0.975])
    return {"estimate": diff.mean(), "ci_low": lo, "ci_high": hi}


def analyze(out):
    """Compute paired bank intervals and genotype-level replacement accounting."""
    scores = pd.read_csv(out / "r1-scores.csv")
    draws = np.random.default_rng(97826).integers(len(SEEDS), size=(10_000, len(SEEDS)))
    expected = pd.MultiIndex.from_product(
        [SEEDS, sorted(METHODS)], names=["seed", "method"]
    )
    levels, contrasts, events = [], [], []
    comparisons = [
        (f"{p}_{lhs}", f"{p}_{rhs}")
        for p in PATHS
        for lhs, rhs in (
            ("joint", "fold"),
            ("veto", "fold"),
            ("joint", "veto"),
            ("joint", "pooled"),
        )
    ]
    comparisons += [
        ("fold_joint", ref)
        for ref in (
            "pooled_joint",
            "pooled_final",
            "anchor",
            "mean",
            "top_n",
            "prob_lr",
            "regularized",
        )
    ]
    if set(
        scores[["law", "n_scenes"]].drop_duplicates().itertuples(index=False, name=None)
    ) != {(law, n) for law in LAWS for n in SIZES}:
        raise ValueError("Incomplete design grid")
    for (law, n_scenes), group in scores.groupby(["law", "n_scenes"]):
        cell = {"law": law, "n_scenes": n_scenes, "rows_per_scene": 8}
        ordered = group.set_index(["seed", "method"]).sort_index()
        if not ordered.index.equals(expected):
            raise ValueError("Missing or duplicated method/seed")
        for metric in ("audit_excess", "audit_brier", "fit_brier", "n_providers"):
            frame = ordered[metric].unstack()
            if not np.isfinite(frame.to_numpy()).all():
                raise ValueError(f"Nonfinite {metric}")
            for method in METHODS:
                levels.append(
                    {
                        **cell,
                        "method": method,
                        "metric": metric,
                        "n_banks": len(SEEDS),
                        **interval(frame[method], draws),
                    }
                )
            for method, reference in comparisons:
                change = (frame[method] - frame[reference]).to_numpy()
                contrasts.append(
                    {
                        **cell,
                        "method": method,
                        "reference": reference,
                        "metric": metric,
                        "n_banks": len(SEEDS),
                        **interval(change, draws),
                        "n_lower": int(np.sum(change < -1e-12)),
                        "n_equal": int(np.sum(np.abs(change) <= 1e-12)),
                        "n_higher": int(np.sum(change > 1e-12)),
                    }
                )
        for seed in SEEDS:
            rows = ordered.loc[seed]
            anchor = rows.loc["anchor"]
            for path in PATHS:
                fold, veto, joint = (
                    rows.loc[f"{path}_{name}"] for name in ("fold", "veto", "joint")
                )
                anchor_id = anchor.selected_digest
                blocked = fold.selected_digest != veto.selected_digest
                rescue = blocked and joint.selected_digest != anchor_id
                if (blocked and veto.selected_digest != anchor_id) or (
                    not blocked and joint.selected_digest != fold.selected_digest
                ):
                    raise RuntimeError(
                        "Inconsistent veto and joint deployment decisions on identical finalists"
                    )
                for record in (veto, joint):
                    if record.selected_digest != anchor_id and not (
                        record.pooled_gain > 0 and record.robust_gain > 0
                    ):
                        raise RuntimeError(
                            "A joint-gated nonanchor lacks a positive pooled or fold gain"
                        )
                if any(
                    fold[k] != joint[k] or fold[k] != veto[k]
                    for k in (*COST_KEYS, "path_digest")
                ):
                    raise RuntimeError("Comparisons must share a single fitted search")
                events.append(
                    {
                        **cell,
                        "seed": seed,
                        "path": path,
                        "blocked": bool(blocked),
                        "rescued": bool(rescue),
                        "fallback_after_block": bool(blocked and not rescue),
                        "anchor_exact": anchor.selected_is_exact,
                        "fold_exact": fold.selected_is_exact,
                        "veto_exact": veto.selected_is_exact,
                        "joint_exact": joint.selected_is_exact,
                        "fold_chose_anchor": fold.selected_digest == anchor_id,
                        "joint_chose_anchor": joint.selected_digest == anchor_id,
                        "veto_chose_anchor": veto.selected_digest == anchor_id,
                        "fold_pooled_gain": fold.pooled_gain,
                        "fold_robust_gain": fold.robust_gain,
                        "joint_pooled_gain": joint.pooled_gain,
                        "joint_robust_gain": joint.robust_gain,
                        "joint_minus_fold_audit_excess": joint.audit_excess
                        - fold.audit_excess,
                        "veto_minus_fold_audit_excess": veto.audit_excess
                        - fold.audit_excess,
                        "joint_minus_veto_audit_excess": joint.audit_excess
                        - veto.audit_excess,
                        "joint_minus_fold_fit_brier": joint.fit_brier - fold.fit_brier,
                        "joint_minus_anchor_audit_excess": joint.audit_excess
                        - anchor.audit_excess,
                        **{k: int(joint[k]) for k in COST_KEYS},
                    }
                )
    for filename, table in (
        ("r1-summary.csv", levels),
        ("r1-contrasts.csv", contrasts),
        ("r1-events.csv", events),
    ):
        pd.DataFrame(table).to_csv(out / filename, index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2, choices=(1, 2))
    parser.add_argument("--out", type=Path, default=HERE / "results")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    if args.analyze_only:
        analyze(args.out)
        return
    started = perf_counter()
    utc = datetime.now(UTC).isoformat()
    batches = {}
    with (
        threadpool_limits(limits=1),
        ProcessPoolExecutor(max_workers=args.workers) as pool,
    ):
        tasks = {pool.submit(score_bank, seed): seed for seed in SEEDS}
        for task in as_completed(tasks):
            seed = tasks[task]
            batches[seed] = task.result()
            print(f"Completed {len(batches)}/{len(SEEDS)} banks.", flush=True)
    frame = pd.DataFrame(row for seed in SEEDS for row in batches[seed][0])
    trace = pd.DataFrame(row for seed in SEEDS for row in batches[seed][1])
    cells = len(SEEDS) * len(LAWS) * len(SIZES)
    if (
        len(frame) != cells * len(METHODS)
        or frame.duplicated(IDS + ["method"]).any()
        or len(trace) != cells * len(PATHS) * 17
        or trace.duplicated(IDS + ["path", "generation"]).any()
    ):
        raise RuntimeError("Incomplete search or decision grid")
    args.out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out / "r1-scores.csv", index=False)
    trace.to_csv(
        args.out / "r1-trajectories.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    from okapi import fold_gain

    manifest = {
        "started_utc": utc,
        "elapsed_seconds": perf_counter() - started,
        "workers": args.workers,
        "numeric_threads_per_worker": 1,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "source_sha256": {
            name: hashlib.sha256(source.read_bytes()).hexdigest()
            for name, source in (
                ("protocol", HERE / "protocol.md"),
                ("runner", Path(__file__)),
                ("search", PREVIOUS / "search.py"),
                ("okapi.fold_gain", Path(fold_gain.__file__)),
            )
        },
        "score_rows": len(frame),
        "trace_rows": len(trace),
        "actual_searches": cells * 2,
        "banks": [batches[s][2] for s in SEEDS],
    }
    (args.out / "r1-run.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )
    analyze(args.out)
    print(
        f"Wrote {len(frame)} scores and {len(trace)} native generation records in {manifest['elapsed_seconds']:.1f}s.",
        flush=True,
    )


if __name__ == "__main__":
    main()

"""Reproduce the staged checkpoint experiment comparing pooled and four-fold promotion."""

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
search = run_path(str(HERE / "search.py"))
fit_search, score_search = search["fit_search"], search["score_search"]
previous = run_path(str(HERE.parent / "scene-count" / "run.py"))
inputs, digest, score_references = (
    previous[key] for key in ("inputs", "digest", "score_references")
)

SEEDS = range(287200, 287232)
DESIGNS = ((0.0, 24), (1.0, 24), (1.0, 192))
LAWS = ("equal", "signal")
METHODS = (
    "pooled_final",
    "pooled_pooled",
    "pooled_fold",
    "fold_pooled",
    "fold_fold",
    "anchor",
    "mean",
    "top_n",
    "prob_lr",
    "regularized",
)


def score_bank(seed):
    rows, traces = [], []
    search_seed = seed + 1_000_000
    info = {
        "seed": seed,
        "search_seed": search_seed,
        "group_seed": seed + 3_000_000,
        "cells": [],
    }
    for law in LAWS:
        selection = generate_checkpoint_bank(seed, law=law)["A"]
        slots = select_checkpoints(selection, 8)
        signal = signal_provider_index(seed) if law == "signal" else None
        for share in (0.0, 1.0):
            pool = generate_scene_pool(
                seed,
                law=law,
                n_fit_scenes=192,
                n_audit_scenes=512,
                scene_error_share=share,
            )
            fit, audit = pool["F"], pool["I"]
            fit_pool, audit_x = inputs(fit, slots), inputs(audit, slots)
            for design_share, scenes in DESIGNS:
                if design_share != share:
                    continue
                ids = scene_sample_indices(scenes, 8)
                x, y, groups = fit_pool[ids], fit.y[ids], fit.scene[ids]
                cell = {
                    "seed": seed,
                    "law": law,
                    "scene_error_share": share,
                    "n_scenes": scenes,
                    "rows_per_scene": 8,
                    "n_rows": len(y),
                }
                details = {
                    **cell,
                    "slot_ids": slots.tolist(),
                    "signal_provider": signal,
                    "selection_A_x_sha256": digest(selection.P),
                    "selection_A_y_sha256": digest(selection.y),
                    "fit_x_sha256": digest(x),
                    "fit_y_sha256": digest(y),
                    "group_sha256": digest(groups),
                    "audit_x_sha256": digest(audit_x),
                    "audit_y_sha256": digest(audit.y),
                    "audit_q_sha256": digest(audit.posterior),
                    "audit_scene_sha256": digest(audit.scene),
                    "paths": {},
                }
                # Freeze both complete searches and all five deployment policies before
                # providing any audit rows to either search's diagnostic evaluator.
                fits = {
                    path: fit_search(
                        x,
                        y,
                        groups=groups,
                        search_seed=search_seed,
                        data_seed=seed,
                        path=path,
                    )
                    for path in ("pooled", "fold")
                }
                for invariant in (
                    "anchor",
                    "founder_digest",
                    "fold_digest",
                    "fold_counts",
                ):
                    assert fits["pooled"][invariant] == fits["fold"][invariant], (
                        invariant
                    )
                for path, fitted in fits.items():
                    scored, trace = score_search(
                        fitted,
                        audit_x,
                        audit.y,
                        audit_q=audit.posterior,
                        signal_provider=signal,
                    )
                    rows.extend(
                        {**cell, **r, "n_providers": r["unique_providers"]}
                        for r in scored
                    )
                    traces.extend({**cell, **r} for r in trace)
                    details["paths"][path] = {
                        **{
                            k: scored[0][k]
                            for k in (
                                "founder_digest",
                                "anchor_digest",
                                "fold_digest",
                                "path_digest",
                                "objective_evaluations",
                                "offspring_proposals",
                                "unique_evaluated",
                            )
                        },
                        "fold_row_counts": fitted["fold_counts"],
                    }
                refs = score_references(
                    x,
                    y,
                    audit_x,
                    audit.y,
                    seed=search_seed,
                    audit_q=audit.posterior,
                    signal_provider=signal,
                )
                rows.extend(
                    {
                        **cell,
                        "method": name,
                        **row,
                        "unique_providers": row["n_providers"],
                    }
                    for name, row in refs.items()
                )
                info["cells"].append(details)
    return rows, traces, info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--out", type=Path, default=HERE / "results")
    args = parser.parse_args()
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
            print(f"Completed {len(batches)}/{len(SEEDS)}.", flush=True)
    frame = pd.DataFrame(row for seed in SEEDS for row in batches[seed][0])
    trace = pd.DataFrame(row for seed in SEEDS for row in batches[seed][1])
    key = ["seed", "law", "scene_error_share", "n_scenes"]
    cells = len(SEEDS) * len(LAWS) * len(DESIGNS)
    if (
        len(frame) != cells * len(METHODS)
        or frame.duplicated(key + ["method"]).any()
        or not (frame.groupby(key).method.apply(set) == set(METHODS)).all()
        or not np.isfinite(frame[["fit_brier", "audit_brier", "audit_excess"]])
        .all()
        .all()
        or len(trace) != cells * 2 * 17
        or trace.duplicated(key + ["path", "generation"]).any()
    ):
        raise RuntimeError("Incomplete score or trajectory grid")
    args.out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out / "scores.csv", index=False)
    trace.to_csv(
        args.out / "trajectories.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    from okapi import fold_gain

    sources = {
        label: hashlib.sha256(path.read_bytes()).hexdigest()
        for label, path in (
            ("protocol", HERE / "protocol.md"),
            ("search", HERE / "search.py"),
            ("runner", Path(__file__)),
            ("okapi.fold_gain", Path(fold_gain.__file__)),
        )
    }
    manifest = {
        "started_utc": utc,
        "workers": args.workers,
        "numeric_threads_per_worker": 1,
        "elapsed_seconds": perf_counter() - started,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "protocol_revision": "1110cbf",
        "source_sha256": sources,
        "score_rows": len(frame),
        "trace_rows": len(trace),
        "actual_searches": 2 * cells,
        "banks": [batches[seed][2] for seed in SEEDS],
    }
    (args.out / "run.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )
    print(
        f"Wrote {len(frame)} scores and {len(trace)} stages in {manifest['elapsed_seconds']:.1f}s.",
        flush=True,
    )


if __name__ == "__main__":
    main()

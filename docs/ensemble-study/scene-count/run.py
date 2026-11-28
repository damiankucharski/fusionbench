"""Run the scene-count experiment on four allocations of independent fusion-fit data."""

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
    generate_scene_pool,
    scene_sample_indices,
    select_checkpoints,
    signal_provider_index,
)

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "seeded-checkpoint"
run_seeded = run_path(str(SOURCE / "native_seed.py"))["run_seeded"]
score_references = run_path(str(SOURCE / "references.py"))["score_references"]

SEEDS = range(286100, 286132)
LAWS = ("equal", "signal")
DESIGNS = ((24, 8), (192, 1), (192, 8), (1536, 1))
BLOCK_FIELDS = ("P", "y", "scene", "posterior")


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def inputs(block, slots):
    positive = block.P[:, np.arange(len(slots)), slots]
    return np.stack((1.0 - positive, positive), axis=-1)


def score_bank(seed):
    """Freeze A-selected columns, fit on F and evaluate each frozen rule on I."""
    started = perf_counter()
    search_seed = seed + 1_000_000
    scores, stages = [], []
    bank_info = {"seed": seed, "search_seed": search_seed, "laws": {}}
    for law in LAWS:
        blocks = generate_scene_pool(
            seed, law=law, n_fit_scenes=1536, n_audit_scenes=512
        )
        selection, fit, audit = (blocks[key] for key in ("A", "F", "I"))
        slots = select_checkpoints(selection, 8)
        fit_pool, audit_x = inputs(fit, slots), inputs(audit, slots)
        signal = signal_provider_index(seed) if law == "signal" else None
        metadata = {
            "slots": slots.tolist(),
            "signal_provider": signal,
            "blocks": {
                name: {key: digest(getattr(block, key)) for key in BLOCK_FIELDS}
                for name, block in blocks.items()
            },
            "audit_x_sha256": digest(audit_x),
            "allocations": [],
        }
        bank_info["laws"][law] = metadata
        for n_scenes, rows_per_scene in DESIGNS:
            indices = scene_sample_indices(n_scenes, rows_per_scene)
            x, y = fit_pool[indices], fit.y[indices]
            n_rows = n_scenes * rows_per_scene
            sampled_scene, counts = np.unique(fit.scene[indices], return_counts=True)
            if (
                len(y) != n_rows
                or len(sampled_scene) != n_scenes
                or not np.all(counts == rows_per_scene)
            ):
                raise RuntimeError(
                    "Fusion-fit subset differs from the declared scene allocation"
                )
            cell = {
                "seed": seed,
                "law": law,
                "n_scenes": n_scenes,
                "rows_per_scene": rows_per_scene,
                "n_rows": n_rows,
            }
            options = {
                "seed": search_seed,
                "audit_q": audit.posterior,
                "signal_provider": signal,
            }
            args = x, y, audit_x, audit.y
            metrics, details = run_seeded(*args, arm="seeded", **options)
            scores.append(
                {
                    **cell,
                    "method": "seeded",
                    "fit_brier": metrics["fit_selected"],
                    "audit_brier": metrics["audit_selected"],
                    "audit_excess": metrics["audit_selected_excess"],
                    "n_providers": metrics["selected_provider_count"],
                    "selected_is_mean": metrics["selected_mean_equiv"],
                    "selected_is_signal": metrics["selected_signal_equiv"],
                    "fit_f1": metrics["fit_f1"],
                    "audit_f1": metrics["audit_f1"],
                }
            )
            stages.extend(
                {**cell, "method": "seeded", **row} for row in details["trajectory"]
            )
            references = score_references(*args, **options)
            scores.extend(
                {**cell, "method": method, **row} for method, row in references.items()
            )
            metadata["allocations"].append(
                {
                    "n_scenes": n_scenes,
                    "rows_per_scene": rows_per_scene,
                    "n_rows": n_rows,
                    "indices_sha256": digest(indices),
                    "scene_ids_sha256": digest(fit.scene[indices]),
                    "fit_x_sha256": digest(x),
                    "fit_y_sha256": digest(y),
                    "selected_program_sha256": metrics["selected_program_sha256"],
                    "trajectory_sha256": metrics["trajectory_sha256"],
                }
            )
    bank_info["worker_seconds"] = perf_counter() - started
    return scores, stages, bank_info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3, choices=range(1, 4))
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    start_utc = datetime.now(UTC).isoformat()
    started = perf_counter()
    batches = {}
    with (
        threadpool_limits(limits=1),
        ProcessPoolExecutor(max_workers=args.workers) as pool,
    ):
        futures = {pool.submit(score_bank, seed): seed for seed in SEEDS}
        for task in as_completed(futures):
            seed = futures[task]
            batches[seed] = task.result()
            print(f"Completed bank {seed} ({len(batches)}/{len(SEEDS)}).", flush=True)
    elapsed = perf_counter() - started
    args.out.mkdir(parents=True, exist_ok=True)
    scores = pd.DataFrame(row for s in SEEDS for row in batches[s][0])
    stages = pd.DataFrame(row for s in SEEDS for row in batches[s][1])
    scores.to_csv(args.out / "scores.csv", index=False)
    stages.to_csv(
        args.out / "trajectories.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    manifest = {
        "started_utc": start_utc,
        "experiment_wall_seconds": elapsed,
        "cpu_workers": args.workers,
        "numerical_threads_per_worker": 1,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "okapi_revision": "1d7ad2d1660627585329bc584a9c4634915e8bb1",
        "seeded_study_source_revision": "6120f52802111a48ab5008266eaa94ec30fc80a0",
        "protocol_revision": "541b4c3",
        "protocol_sha256": hashlib.sha256(
            (ROOT / "protocol.md").read_bytes()
        ).hexdigest(),
        "blocks": {"A": "checkpoint selection", "F": "fusion fit", "I": "audit"},
        "banks": [batches[s][2] for s in SEEDS],
    }
    (args.out / "run.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )
    print(
        f"Wrote {len(scores)} scores and {len(stages)} trajectory rows in {elapsed:.1f}s.",
        flush=True,
    )


if __name__ == "__main__":
    main()

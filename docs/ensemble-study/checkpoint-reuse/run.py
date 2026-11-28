"""Run the paired checkpoint selection experiment. See protocol.md for estimands."""

import argparse
import hashlib
import json
import platform
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import sklearn
from controls import score_controls
from native import discover_and_score
from regularized import score_regularized
from threadpoolctl import threadpool_limits

from fusionbench.scenarios.checkpoint_bank import (
    generate_checkpoint_bank,
    select_checkpoints,
)


def as_bank(block, slots):
    positive = block.P[:, np.arange(len(slots)), slots]
    return np.stack((1.0 - positive, positive), axis=-1)


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def provider_f1(block, slots):
    predicted = block.P[:, np.arange(len(slots)), slots] > 0.5
    counts = predicted.sum(axis=0) + block.y.sum()
    return np.divide(
        2 * (predicted * block.y[:, None]).sum(axis=0),
        counts,
        out=np.zeros(len(slots)),
        where=counts > 0,
    )


def score_cell(fit_x, fit_y, audit_x, audit_y, search_seed):
    rows = score_controls(fit_x, fit_y, audit_x, audit_y)
    rows["regularized"] = score_regularized(
        fit_x, fit_y, audit_x, audit_y, seed=search_seed
    )
    stages = []
    for objective in ("brier", "f1"):
        method = "gp_" + objective
        row, detail = discover_and_score(
            fit_x, fit_y, audit_x, audit_y, seed=search_seed, objective=objective
        )
        rows[method] = {
            **row,
            "fit_prediction_sha256": digest(detail["fit_selected"]),
            "audit_prediction_sha256": digest(detail["audit_selected"]),
        }
        stages.extend({"method": method, **entry} for entry in detail["trajectory"])
    return rows, stages


def run_bank(seed):
    blocks = generate_checkpoint_bank(seed)
    scores, checkpoints, trajectories = [], [], []
    manifest = {
        name: {
            "probabilities": digest(b.P),
            "labels": digest(b.y),
            "scenes": digest(b.scene),
            "positive_rows": int(b.y.sum()),
            "positive_scenes": len(np.unique(b.scene[b.y == 1])),
        }
        for name, b in blocks.items()
    }
    # K=1 yields identical fits under both checkpoint selectors. Reuse exact repeated
    # fits within a bank, keeping both crossover rows for the paired analysis.
    cache = {}
    for k in (1, 8):
        for selection, other in (("A", "B"), ("B", "A")):
            slots = select_checkpoints(blocks[selection], k)
            selection_f1, opposite_f1, audit_f1 = (
                provider_f1(blocks[n], slots) for n in (selection, other, "I")
            )
            for provider, slot in enumerate(slots):
                checkpoints.append(
                    {
                        "seed": seed,
                        "k": k,
                        "checkpoint_block": selection,
                        "provider": provider,
                        "slot": int(slot),
                        "selection_f1": selection_f1[provider],
                        "opposite_f1": opposite_f1[provider],
                        "audit_f1": audit_f1[provider],
                    }
                )
            audit_x = as_bank(blocks["I"], slots)
            for fit in (selection, other):
                key = (tuple(slots), fit)
                if key not in cache:
                    cache[key] = score_cell(
                        as_bank(blocks[fit], slots),
                        blocks[fit].y,
                        audit_x,
                        blocks["I"].y,
                        seed + 1000000,
                    )
                rows, stages = cache[key]
                cell = {
                    "seed": seed,
                    "k": k,
                    "checkpoint_block": selection,
                    "fit_block": fit,
                }
                scores.extend(
                    {**cell, "method": method, **row} for method, row in rows.items()
                )
                trajectories.extend({**cell, **entry} for entry in stages)
    return (
        scores,
        checkpoints,
        trajectories,
        {"seed": seed, "blocks": manifest, "unique_fits_per_procedure": len(cache)},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("results"))
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    seeds = range(284100, 284132)
    started = datetime.now(UTC).isoformat()
    tic = perf_counter()
    with (
        threadpool_limits(limits=1),
        ProcessPoolExecutor(max_workers=args.workers) as pool,
    ):
        results = list(pool.map(run_bank, seeds))
    elapsed = perf_counter() - tic
    args.out.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(("scores.csv", "checkpoints.csv", "trajectories.csv.gz")):
        table = pd.DataFrame(row for batch in results for row in batch[i])
        table.to_csv(
            args.out / name,
            index=False,
            compression={"method": "gzip", "mtime": 0} if name.endswith("gz") else None,
        )
    metadata = {
        "started_utc": started,
        "wall_seconds": elapsed,
        "protocol_commit": "1ef8a34",
        "okapi_revision": "1d7ad2d1660627585329bc584a9c4634915e8bb1",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "workers": args.workers,
        "numeric_threads_per_worker": 1,
        "banks": [b[3] for b in results],
    }
    (args.out / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"{len(results)} banks in {elapsed:.2f} seconds", flush=True)


if __name__ == "__main__":
    main()

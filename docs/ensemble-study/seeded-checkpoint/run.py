"""Run the prespecified mean-founder comparisons on the paired checkpoint laws."""

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
from native_seed import run_seeded
from references import score_references
from threadpoolctl import threadpool_limits

from fusionbench.scenarios.checkpoint_bank import (
    generate_checkpoint_bank,
    select_checkpoints,
    signal_provider_index,
)

SEEDS = range(285100, 285132)
ARMS = ("stock", "sham", "seeded", "carry")
DATASETS = ("A", "B", "I")


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def inputs(block, slots):
    positive = block.P[:, np.arange(len(slots)), slots]
    return np.stack((1.0 - positive, positive), axis=2)


def deployment(metrics, detail, fit_x, audit_x, signal, *, protected):
    prefix = "protected_" if protected else ""
    bank_predictions = [
        detail[prefix + split + "_selected"] for split in ("fit", "audit")
    ]
    banks = (fit_x[:, :, 1], audit_x[:, :, 1])
    equivalent_mean = all(
        np.allclose(p, x.mean(axis=1), atol=1e-12, rtol=0)
        for p, x in zip(bank_predictions, banks, strict=True)
    )
    equivalent_signal = (
        None
        if signal is None
        else all(
            np.allclose(p, x[:, signal], atol=1e-12, rtol=0)
            for p, x in zip(bank_predictions, banks, strict=True)
        )
    )
    return {
        "fit_brier": metrics[prefix + "fit_selected"],
        "audit_brier": metrics[prefix + "audit_selected"],
        "fit_f1": metrics[prefix + "fit_f1"],
        "audit_f1": metrics[prefix + "audit_f1"],
        "audit_excess": metrics[prefix + "audit_selected_excess"],
        "n_providers": metrics[prefix + "selected_provider_count"],
        "selected_is_mean": equivalent_mean,
        "selected_is_signal": equivalent_signal,
        "selected_program_sha256": metrics[prefix + "selected_program_sha256"],
        "fit_prediction_sha256": digest(bank_predictions[0]),
        "audit_prediction_sha256": digest(bank_predictions[1]),
        **{
            key: metrics[key]
            for key in (
                "audit_best_seen_excess",
                "audit_best_final_excess",
                "n_seen",
                "n_survivors",
                "score_calls",
                "proposals",
                "carry_reinsertions",
                "trajectory_sha256",
                "initial_population_sha256",
            )
        },
    }


def score_bank(seed):
    tic = perf_counter()
    sham = int(np.random.default_rng(np.random.SeedSequence([seed, 0, 2])).integers(8))
    search_seed = seed + 1000000
    scores, stages, checkpoints = [], [], []
    manifest = {}
    for law in ("equal", "signal"):
        blocks = generate_checkpoint_bank(seed, law=law)
        signal = signal_provider_index(seed) if law == "signal" else None
        manifest[law] = {
            name: {
                key: digest(getattr(b, key)) for key in ("P", "y", "scene", "posterior")
            }
            for name, b in blocks.items()
        }
        for source, other in (("A", "B"), ("B", "A")):
            slots = select_checkpoints(blocks[source], 8)
            checkpoints.extend(
                {
                    "seed": seed,
                    "law": law,
                    "checkpoint_block": source,
                    "provider": m,
                    "slot": int(s),
                    "signal_provider": signal,
                    "sham_provider": sham,
                }
                for m, s in enumerate(slots)
            )
            frozen = {n: inputs(blocks[n], slots) for n in DATASETS}
            audit = blocks["I"]
            for fit in (source, other):
                cell = {
                    "seed": seed,
                    "law": law,
                    "checkpoint_block": source,
                    "fit_block": fit,
                }
                args = (frozen[fit], blocks[fit].y, frozen["I"], audit.y)
                kwargs = {
                    "seed": search_seed,
                    "audit_q": audit.posterior,
                    "signal_provider": signal,
                }
                refs = score_references(*args, **kwargs)
                scores.extend({**cell, "method": m, **r} for m, r in refs.items())
                for arm in ARMS:
                    metrics, detail = run_seeded(
                        *args, **kwargs, arm=arm, sham_provider=sham
                    )
                    raw = deployment(
                        metrics,
                        detail,
                        frozen[fit],
                        frozen["I"],
                        signal,
                        protected=False,
                    )
                    scores.append({**cell, "method": arm, **raw})
                    stages.extend(
                        {**cell, "method": arm, **row} for row in detail["trajectory"]
                    )
                    if arm == "seeded":
                        protected = deployment(
                            metrics,
                            detail,
                            frozen[fit],
                            frozen["I"],
                            signal,
                            protected=True,
                        )
                        scores.append(
                            {
                                **cell,
                                "method": "protected",
                                "used_mean_finalist": metrics["protected_used_mean"],
                                **protected,
                            }
                        )
    return (
        scores,
        stages,
        checkpoints,
        {"seed": seed, "blocks": manifest, "worker_seconds": perf_counter() - tic},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("results"))
    args = parser.parse_args()
    start = datetime.now(UTC).isoformat()
    tic = perf_counter()
    with (
        threadpool_limits(limits=1),
        ProcessPoolExecutor(max_workers=args.workers) as pool,
    ):
        results = list(pool.map(score_bank, SEEDS))
    seconds = perf_counter() - tic
    args.out.mkdir(parents=True, exist_ok=True)
    for i, filename in enumerate(
        ("scores.csv", "trajectories.csv.gz", "checkpoints.csv")
    ):
        compression = (
            {"method": "gzip", "mtime": 0} if filename.endswith("gz") else None
        )
        df = pd.DataFrame(row for batch in results for row in batch[i])
        df.to_csv(args.out / filename, index=False, compression=compression)
    manifest = {
        "started_utc": start,
        "experiment_wall_seconds": seconds,
        "cpu_workers": args.workers,
        "numerical_threads_per_worker": 1,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "protocol": "8243ddc",
        "okapi_revision": "1d7ad2d1660627585329bc584a9c4634915e8bb1",
        "banks": [b[3] for b in results],
    }
    (args.out / "run.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Completed {len(results)} paired banks in {seconds:.2f} s.")


if __name__ == "__main__":
    main()

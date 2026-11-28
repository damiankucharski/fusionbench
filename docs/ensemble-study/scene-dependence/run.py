"""Vary within-scene provider error sharing with fixed slots and matched fit budgets."""

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
    ROWS_PER_SCENE,
    generate_checkpoint_bank,
    generate_scene_pool,
    scene_sample_indices,
    select_checkpoints,
    signal_provider_index,
)

ROOT = Path(__file__).resolve().parent
previous = run_path(str(ROOT.parent / "scene-count" / "run.py"))
digest, inputs, run_seeded, score_references = (
    previous[k] for k in ("digest", "inputs", "run_seeded", "score_references")
)
fixed = run_path(str(ROOT.parent / "scene-count" / "fixed_comparison.py"))
loss_difference, ratio = fixed["loss_difference"], fixed["ratio"]
SEEDS = range(287100, 287132)
LAWS = ("equal", "signal")
SHARES = (0.0, 0.5, 1.0)
DESIGNS = ((24, 8), (192, 1))
BLOCK_FIELDS = ("P", "y", "scene", "posterior")


def variance_components(fit, audit, slots, singleton, allocations):
    """Compare a prespecified singleton with the arithmetic mean on all F scenes.

    Within is the pooled sample variance within eight-row scenes. Between is the
    sample variance of scene means. Their ratio of sampling variances gives the
    design effect at equal row count. The singleton has the same anonymous index
    in both laws, fixed independently of empirical performance on F or I.
    """
    delta = loss_difference(fit, slots, singleton, fit.y)
    by_scene = delta.reshape(-1, ROWS_PER_SCENE)
    within = float(by_scene.var(axis=1, ddof=1).mean())
    between = float(by_scene.mean(axis=1).var(ddof=1))
    design_effect = float(ratio(between, within))
    return {
        "n_pool_scenes": len(by_scene),
        "singleton": singleton,
        "within": within,
        "between_means": between,
        "variance_ratio": design_effect,
        "loss_difference_intraclass_correlation": (design_effect - 1)
        / (ROWS_PER_SCENE - 1),
        "audit_conditional_difference": float(
            loss_difference(audit, slots, singleton, audit.posterior).mean()
        ),
        **{
            f"fit_difference_{g}x{r}": float(delta[indices].mean())
            for (g, r), indices in allocations.items()
        },
    }


def score_bank(seed):
    started = perf_counter()
    scores, trajectories, components = [], [], []
    search_seed = seed + 1_000_000
    singleton = signal_provider_index(seed)
    allocations = {(g, r): scene_sample_indices(g, r) for g, r in DESIGNS}
    info = {"seed": seed, "search_seed": search_seed, "laws": {}}
    equal_slots = None
    for law in LAWS:
        # Select once on historical A. Reselecting after changing the residuals in
        # a provider's checkpoints would mix changed checkpoint choice into the
        # effect of dependence. The planted provider's identical slots tie at 0.
        selection = generate_checkpoint_bank(seed, law=law)["A"]
        slots = select_checkpoints(selection, 8)
        signal = singleton if law == "signal" else None
        if law == "equal":
            equal_slots = slots
        elif slots[singleton] != 0 or not np.array_equal(
            np.delete(equal_slots, singleton), np.delete(slots, singleton)
        ):
            raise RuntimeError(
                "Default-A checkpoint choices differ outside the planted provider"
            )
        law_info = {
            "slots": slots.tolist(),
            "signal_provider": signal,
            "selection_A": {
                key: digest(getattr(selection, key)) for key in BLOCK_FIELDS
            },
            "states": [],
        }
        info["laws"][law] = law_info
        for share in SHARES:
            blocks = generate_scene_pool(
                seed,
                law=law,
                n_fit_scenes=1536,
                n_audit_scenes=512,
                scene_error_share=share,
            )
            fit, audit = blocks["F"], blocks["I"]
            fit_pool, audit_x = inputs(fit, slots), inputs(audit, slots)
            state = {"seed": seed, "law": law, "scene_error_share": share}
            state_info = {
                "scene_error_share": share,
                "blocks": {
                    name: {key: digest(getattr(block, key)) for key in BLOCK_FIELDS}
                    for name, block in (("F", fit), ("I", audit))
                },
                "audit_x_sha256": digest(audit_x),
                "allocations": [],
            }
            law_info["states"].append(state_info)
            components.append(
                {
                    **state,
                    **variance_components(fit, audit, slots, singleton, allocations),
                }
            )
            for (g, r), indices in allocations.items():
                x, y = fit_pool[indices], fit.y[indices]
                n_rows = g * r
                scenes, counts = np.unique(fit.scene[indices], return_counts=True)
                if len(y) != n_rows or len(scenes) != g or not np.all(counts == r):
                    raise RuntimeError(
                        "F subset differs from the frozen scene allocation"
                    )
                cell = {**state, "n_scenes": g, "rows_per_scene": r, "n_rows": n_rows}
                args = x, y, audit_x, audit.y
                options = {
                    "seed": search_seed,
                    "audit_q": audit.posterior,
                    "signal_provider": signal,
                }
                metrics, detail = run_seeded(*args, arm="seeded", **options)
                scores.append(
                    {
                        **cell,
                        "method": "seeded",
                        "fit_brier": metrics["fit_selected"],
                        "audit_brier": metrics["audit_selected"],
                        "audit_excess": metrics["audit_selected_excess"],
                        "fit_f1": metrics["fit_f1"],
                        "audit_f1": metrics["audit_f1"],
                        "selected_is_mean": metrics["selected_mean_equiv"],
                        "selected_is_signal": metrics["selected_signal_equiv"],
                        "n_providers": metrics["selected_provider_count"],
                    }
                )
                trajectories.extend(
                    {**cell, "method": "seeded", **row} for row in detail["trajectory"]
                )
                refs = score_references(*args, **options)
                scores.extend(
                    {**cell, "method": method, **row} for method, row in refs.items()
                )
                state_info["allocations"].append(
                    {
                        "n_scenes": g,
                        "rows_per_scene": r,
                        "n_rows": n_rows,
                        "indices_sha256": digest(indices),
                        "scene_ids_sha256": digest(fit.scene[indices]),
                        "fit_x_sha256": digest(x),
                        "fit_y_sha256": digest(y),
                        "selected_program_sha256": metrics["selected_program_sha256"],
                        "trajectory_sha256": metrics["trajectory_sha256"],
                    }
                )
    info["worker_seconds"] = perf_counter() - started
    return scores, trajectories, components, info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2, choices=(1, 2))
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    start_utc = datetime.now(UTC).isoformat()
    started = perf_counter()
    batches = {}
    with (
        threadpool_limits(limits=1),
        ProcessPoolExecutor(max_workers=args.workers) as pool,
    ):
        tasks = {pool.submit(score_bank, seed): seed for seed in SEEDS}
        for job in as_completed(tasks):
            seed = tasks[job]
            batches[seed] = job.result()
            print(f"Completed bank {seed} ({len(batches)}/{len(SEEDS)}).", flush=True)
    elapsed = perf_counter() - started
    args.out.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(
        ("scores.csv", "trajectories.csv.gz", "variance-components.csv")
    ):
        rows = pd.DataFrame(row for seed in SEEDS for row in batches[seed][i])
        compression = {"method": "gzip", "mtime": 0} if name.endswith(".gz") else None
        rows.to_csv(args.out / name, index=False, compression=compression)
    manifest = {
        "started_utc": start_utc,
        "experiment_wall_seconds": elapsed,
        "cpu_workers": args.workers,
        "numerical_threads_per_worker": 1,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "okapi_revision": "1d7ad2d1660627585329bc584a9c4634915e8bb1",
        "study_source_revision": "8bf95a0ab613c953ea35979990f340d8665cea97",
        "protocol_sha256": hashlib.sha256(
            (ROOT / "protocol.md").read_bytes()
        ).hexdigest(),
        "banks": [batches[seed][3] for seed in SEEDS],
    }
    (args.out / "run.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )
    print(f"Finished {len(SEEDS)} paired banks in {elapsed:.1f}s.", flush=True)


if __name__ == "__main__":
    main()

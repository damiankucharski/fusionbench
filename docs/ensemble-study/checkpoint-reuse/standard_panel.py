"""Replay the existing standard-v3 evaluation on its original seed 1729."""

import argparse
import hashlib
import json
import platform
import runpy
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits

from fusionbench.transfer import load_preset

HERE = Path(__file__).resolve().parent
REFERENCE = HERE.parent / "scenario-coverage" / "standard-v3"
SEED = 1729
KEY = ["scenario_id", "data_seed", "search_seed", "method"]
TOLERANCE = 1e-12
# Use the exact archived dataset generator, method factories and evaluation loop.
# Its main() is never called, so the archived results stay untouched.
DRIVER = runpy.run_path(str(REFERENCE / "evaluate.py"))


def replay(scenario):
    with threadpool_limits(limits=1):
        return DRIVER["evaluate_bank"]((scenario, SEED))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    scenarios = tuple(load_preset("standard-v3")["scenarios"])
    methods = tuple(DRIVER["METHODS"])
    expected = {(s, SEED, 0, m) for s in scenarios for m in methods}
    if len(scenarios) != 24 or len(methods) != 4 or len(expected) != 96:
        raise RuntimeError(
            "The standard-v3 reference requires exactly 24 scenarios and 4 methods"
        )

    source = REFERENCE / "scores.csv"
    historical = pd.read_csv(source, float_precision="round_trip").query(
        "data_seed == @SEED"
    )
    started = perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as workers:
        reproduced = pd.concat(workers.map(replay, scenarios), ignore_index=True)
    wall_seconds = perf_counter() - started
    for frame in (historical, reproduced):
        if set(frame[KEY].itertuples(index=False, name=None)) != expected:
            raise RuntimeError("Incomplete or unexpected reference rows")
    historical = historical.rename(
        columns={
            "fit_brier": "archived_fit_brier",
            "test_brier": "archived_test_brier",
            "fit_time_s": "archived_fit_time_s",
        }
    )
    scores = reproduced.merge(historical, on=KEY, validate="one_to_one")
    for split in ("fit", "test"):
        scores[f"abs_{split}_error"] = (
            scores[f"{split}_brier"] - scores[f"archived_{split}_brier"]
        ).abs()
    errors = scores[["abs_fit_error", "abs_test_error"]].to_numpy()
    if not np.isfinite(errors).all():
        raise RuntimeError("A reference or reproduced score is not finite")

    maximum = float(scores.abs_test_error.max())
    report = {
        "purpose": "Regression replay of published seed 1729 from the four-seed standard-v3 evaluation",
        "reference_evaluation_commit": "a218ee1071009a5fa3076bb49d3f40e06e6a8d6c",
        "reference_scores_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "data_seed": SEED,
        "search_seed": 0,
        "rows_per_split": 512,
        "scenarios": len(scenarios),
        "methods": len(methods),
        "comparisons": len(scores),
        "workers": args.workers,
        "numeric_threads_per_worker": 1,
        "evaluation_wall_seconds": wall_seconds,
        "absolute_tolerance": TOLERANCE,
        "max_abs_test_brier_error": maximum,
        "max_abs_fit_brier_error": float(scores.abs_fit_error.max()),
        "test_scores_within_tolerance": int((scores.abs_test_error <= TOLERANCE).sum()),
        "methods_summary": scores.groupby("method", sort=False)
        .agg(
            test_brier=("test_brier", "mean"),
            archived_test_brier=("archived_test_brier", "mean"),
            max_abs_test_brier_error=("abs_test_error", "max"),
            summed_fit_time_s=("fit_time_s", "sum"),
        )
        .to_dict("index"),
    }
    output = HERE / "results"
    output.mkdir(exist_ok=True)
    scores.to_csv(output / "standard-panel.csv", index=False)
    (output / "standard-panel-timing.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(report["methods_summary"], indent=2), flush=True)
    if maximum > TOLERANCE:
        raise RuntimeError(f"Test Brier regression exceeds {TOLERANCE:g}: {maximum:g}")


if __name__ == "__main__":
    main()

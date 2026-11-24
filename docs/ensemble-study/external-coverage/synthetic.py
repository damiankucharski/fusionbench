"""Generate matched multiclass banks and score the frozen methods on synthetic tests."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from common import evaluate_methods, profile_fit
from fusionbench import Dataset, get_scenario

STRATA = {"digits": (10, 449, 3_160_000), "wine": (3, 44, 3_170_000)}
PANEL = "multiclass_provider_panel"
SETTINGS = {
    "committee_60": ("multiclass_committee", {"k": 3, "accuracy": 0.60}),
    "committee_95": ("multiclass_committee", {"k": 3, "accuracy": 0.95}),
    "saturated_50": ("saturated", {"k": 3, "accuracy": 0.95, "corr": 0.50}),
    "saturated_95": ("saturated", {"k": 3, "accuracy": 0.95, "corr": 0.95}),
    "panel_A": (PANEL, {"separation": 2.5, "votes": [0, 0]}),
    "panel_B": (PANEL, {"separation": 6.0, "votes": [31, 7]}),
    "panel_C": (PANEL, {"separation": 6.0, "votes": [0, 0]}),
    "panel_D": (PANEL, {"separation": 2.5, "votes": [31, 7]}),
    "panel_E": (PANEL, {"separation": 6.0, "votes": [31, 7], "confusion_bias": 3.0}),
}


def run_bank(job):
    stratum, config, replicate = job
    n_classes, n_fit, root = STRATA[stratum]
    seed = root + replicate
    scenario_id, parameters = SETTINGS[config]
    scenario = get_scenario(scenario_id, n_classes=n_classes, **parameters)
    fit, test = [
        scenario.sample(size, np.random.default_rng([seed, stream]))
        for stream, size in enumerate((n_fit, 2048))
    ]
    dataset = Dataset(
        name=config,
        task="multiclass",
        n_classes=n_classes,
        model_ids=scenario.model_ids(),
        fit_preds=fit[0].astype(np.float32),
        fit_y=fit[1],
        test_preds=test[0].astype(np.float32),
        test_y=test[1],
    )
    key = {"stratum": stratum, "config": config, "replicate": replicate, "seed": seed}
    profile = {**key, **profile_fit(dataset.fit_preds, dataset.fit_y)}
    scores = [{**key, **row} for row in evaluate_methods(dataset, seed=seed)]
    return profile, scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    start = perf_counter()
    jobs = [(s, c, r) for s in STRATA for c in SETTINGS for r in range(8)]
    profiles, scores = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for profile, rows in pool.map(run_bank, jobs):
            profiles.append(profile)
            scores.extend(rows)
    dest = args.out
    dest.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(scores).to_csv(dest / "scores.csv", index=False)
    pd.DataFrame(profiles).to_csv(dest / "profiles.csv", index=False)
    (dest / "settings.json").write_text(
        json.dumps(
            {
                "strata_classes_fit_rows_seed_root": STRATA,
                "scenarios": SETTINGS,
                "replicates": 8,
                "test_rows": 2048,
                "rng_streams": {"fit": 0, "test": 1},
                "workers": args.workers,
                "elapsed_seconds": perf_counter() - start,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

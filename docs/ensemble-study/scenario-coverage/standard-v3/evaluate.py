"""Reproduce the standard-v3 reference evaluation from the source checkout."""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

import pandas as pd

from fusionbench import generate_dataset
from fusionbench.eval.runner import evaluate_dataset
from fusionbench.methods.registry import FACTORIES
from fusionbench.transfer import load_preset

METHODS = ("okapi_native_pattern_l2", "simple_average", "top_n", "stacking_lr")
SEEDS = (1729, 1730, 1731, 1732)


def evaluate_bank(job):
    scenario_id, seed = job
    bank = generate_dataset(scenario_id, n_per_split=512, seed=seed)
    methods = [FACTORIES[name]() for name in METHODS]
    scores = evaluate_dataset(bank, methods)
    if scores["error"].ne("").any():
        raise RuntimeError(scores.loc[scores.error != "", ["method", "error"]].to_dict("records"))
    scores = scores[scores.method.isin(METHODS)].copy()
    return scores.assign(scenario_id=scenario_id, data_seed=seed, search_seed=0)[
        ["scenario_id", "data_seed", "search_seed", "method", "fit_brier", "test_brier", "fit_time_s"]
    ]


def main():
    start = perf_counter()
    scenarios = load_preset("standard-v3")["scenarios"]
    jobs = [(name, seed) for name in scenarios for seed in SEEDS]
    with ProcessPoolExecutor(max_workers=3) as pool:
        scores = pd.concat(pool.map(evaluate_bank, jobs), ignore_index=True)
    if len(scores) != len(jobs) * len(METHODS) or scores.test_brier.isna().any():
        raise RuntimeError("Incomplete evaluation")
    target = Path(__file__).resolve().parent
    scores.to_csv(target / "scores.csv", index=False)
    means = scores.pivot_table(index="scenario_id", columns="method", values="test_brier")
    means.to_csv(target / "means.csv")
    print(means.mean(), f"\nElapsed seconds: {perf_counter()-start:.2f}")


if __name__ == "__main__":
    main()

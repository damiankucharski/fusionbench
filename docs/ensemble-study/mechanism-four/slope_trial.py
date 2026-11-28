"""Run the registered synthetic stress test on the response slope of base providers."""

import argparse
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from calibrate_gp import fit_calibrated
from run_four import fit_models, score_models
from slope_banks import SLOPES, banks_slope
from stats_four import summarize, weights
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent / "results"
METHODS = {
    "mean",
    "cal_mean",
    "lr",
    "top_n",
    "full",
    "previous8",
    "pull_mean",
    "cal_pull",
}


def fit_one(bank):
    with threadpool_limits(limits=1):
        start = perf_counter()
        models = fit_models(4, bank.x, bank.y, groups=bank.groups, chosen="cal_pull")
        models["cal_mean"] = fit_calibrated(
            models["mean"], bank.x, bank.y, groups=bank.groups
        )
        seconds = perf_counter() - start
        if set(models) != METHODS:
            raise RuntimeError("Unexpected policies for provider-slope stress trial")
        return score_models(5, bank, models), seconds


def run(workers):
    start = perf_counter()
    rows = []
    fit_seconds = 0
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=mp.get_context("spawn")
    ) as pool:
        for i, (batch, seconds) in enumerate(
            pool.map(fit_one, banks_slope(), chunksize=2), 1
        ):
            rows.extend(batch)
            fit_seconds += seconds
            if i % 50 == 0:
                print(f"Round 5: {i}/426 bank conditions", flush=True)
    table = pd.DataFrame(rows)
    key = ["family", "cohort", "seed", "slope", "method"]
    assert (
        not table.duplicated(key).any() and i == 426 and len(rows) == i * len(METHODS)
    )
    table.to_csv(OUT / "r5-scores.csv", index=False)
    (OUT / "r5-time.json").write_text(
        json.dumps(
            {
                "round": 5,
                "banks": i,
                "scores": len(rows),
                "workers": workers,
                "threads_per_worker": 1,
                "seconds": perf_counter() - start,
                "sum_fit_seconds": fit_seconds,
            },
            indent=2,
        )
        + "\n"
    )


def summarize_slope(table):
    facet = table.query("cohort=='factorial'")
    other = table.query("cohort!='factorial'")
    output = {"levels": [], "contrasts": []}
    for slope in SLOPES:
        subset = facet[facet.slope == slope]
        levels, contrasts, families = summarize(pd.concat([other, subset]), 5)
        for name, data in (("levels", levels), ("contrasts", contrasts)):
            take = data.group.isin(["factorial", *subset.family.unique()])
            if slope == SLOPES[0]:
                output[name].append(data[~take].assign(slope=np.nan))
            output[name].append(data[take].assign(slope=slope))
        if slope == SLOPES[0]:
            families.to_csv(OUT / "r5-families.csv", index=False)
    for name, frames in output.items():
        pd.concat(frames, ignore_index=True).to_csv(OUT / f"r5-{name}.csv", index=False)

    # Differences always compare the same seed and retain all eight cells.
    comparisons = [
        ("cal_pull", "pull_mean"),
        ("cal_mean", "mean"),
        ("cal_pull", "cal_mean"),
        ("pull_mean", "lr"),
        ("mean", "lr"),
        ("cal_pull", "lr"),
        ("cal_mean", "lr"),
    ]
    rows = []
    for metric in ("expected_brier", "brier", "ap", "macro_f1"):
        cells = facet.pivot(
            index=["seed", "family", "slope"], columns="method", values=metric
        )
        assert len(cells) == 12 * 8 * 3 and not cells.isna().any().any()
        for method, reference in comparisons:
            diffs = (
                (cells[method] - cells[reference])
                .groupby(["seed", "slope"])
                .mean()
                .unstack()
            )
            for slope in SLOPES[1:]:
                contrast = (diffs[slope] - diffs[SLOPES[0]]).to_numpy()
                boot = weights(1, len(contrast), 5) @ contrast
                lo, hi = np.quantile(boot, [0.025, 0.975])
                rows.append(
                    {
                        "metric": metric,
                        "method": method,
                        "reference": reference,
                        "slope": slope,
                        "reference_slope": SLOPES[0],
                        "delta_at_reference_slope": diffs[SLOPES[0]].mean(),
                        "delta_at_slope": diffs[slope].mean(),
                        "difference_in_differences": contrast.mean(),
                        "lo": lo,
                        "hi": hi,
                        "paired_seeds": len(contrast),
                    }
                )
    pd.DataFrame(rows).to_csv(OUT / "r5-slope-effects.csv", index=False)
    calibration = (
        facet[facet.method.isin(["cal_mean", "cal_pull"])]
        .groupby(["slope", "method", "beta"], as_index=False)
        .size()
    )
    calibration.to_csv(OUT / "r5-beta-frequencies.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    if not args.summarize_only:
        run(args.workers)
    with threadpool_limits(limits=1):
        summarize_slope(
            pd.read_csv(OUT / "r5-scores.csv", float_precision="round_trip")
        )

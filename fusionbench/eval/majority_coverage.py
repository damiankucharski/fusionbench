"""Measure the coverage of two candidate vote axes without distillation."""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

from fusionbench.methods.majority_worlds import (
    shared_majority_probability,
    two_world_bounds,
)


def inputs(rng, df, angle, rows):
    # A random orthonormal coordinate system can be fixed to the first four
    # axes without changing any probability in this rotationally defined test.
    a, b = (rng.normal(size=(2, *rows)) >= 0).astype(int)
    v = (
        rng.normal(size=rows)
        if df is None
        else rng.choice([-1, 1], size=rows) * np.sqrt(rng.chisquare(df, size=rows))
    )
    w = rng.normal(size=rows)
    c0 = v >= 0
    c1 = v * np.cos(angle) + w * np.sin(angle) >= 0
    y0 = a + b + c0 >= 2
    y1 = a + b + c1 >= 2
    return a, b, c0, c1, y0, y1


def run(out, seeds, confirm=False):
    start = datetime.now(UTC)
    timer = time.perf_counter()
    out.mkdir(exist_ok=True, parents=True)
    rows, analytic = [], []
    degrees = [45] if confirm else [45, 60]
    shapes = [15] if confirm else [1, 7, 15, 31]
    sample_sizes = [512] if confirm else [32, 128, 512]
    repeats = 256 if confirm else 64
    for deg in degrees:
        angle = np.deg2rad(deg)
        for df in shapes:
            for n in [0, 32, 128, 512, 2048, 8192]:
                result = two_world_bounds(angle, df, n)
                labels95 = math.ceil(math.log(0.05) / math.log1p(-result["p_train"]))
                analytic.append(
                    dict(
                        degrees=deg, chi_df=df, labels=n, n_for_95pct=labels95, **result
                    )
                )
            for seed in seeds:
                rng = np.random.default_rng([seed, df, deg])
                world = rng.integers(0, 2, size=(repeats, 1))
                *_, p0, p1 = inputs(rng, df, angle, (repeats, 512))
                truth = np.where(world, p1, p0)
                a, b, c0, c1, z0, z1 = inputs(rng, None, angle, (repeats, 256))
                target = np.where(world, z1, z0)
                for n in sample_sizes:
                    feasible0 = (p0[:, :n] == truth[:, :n]).all(1, keepdims=True)
                    feasible1 = (p1[:, :n] == truth[:, :n]).all(1, keepdims=True)
                    assert np.all(feasible0 | feasible1)
                    unresolved = feasible0 & feasible1
                    posterior = feasible1 / (feasible0.astype(float) + feasible1)
                    shared = shared_majority_probability(a, b, c0, c1, posterior)
                    direct = (1 - posterior) * z0 + posterior * z1
                    hard_world = posterior > 0.5
                    hard = np.where(hard_world, z1, z0)
                    rows.append(
                        {
                            "seed": seed,
                            "degrees": deg,
                            "chi_df": df,
                            "labels": n,
                            "p_train": np.mean(p0[:, :n] != p1[:, :n]),
                            "p_test": np.mean(z0 != z1),
                            "unresolved": unresolved.mean(),
                            "world_error": np.mean(hard_world != world),
                            "target_error": np.mean((shared >= 0.5) != target),
                            "target_brier": np.mean((shared - target) ** 2),
                            "hard_world_brier": np.mean(hard != target),
                            "network_max_abs": np.abs(shared - direct).max(),
                        }
                    )
    df = pd.DataFrame(rows)
    analytic = pd.DataFrame(analytic)
    keys = ["degrees", "chi_df", "labels"]
    df.to_csv(out / "banks.csv", index=False)
    analytic.to_csv(out / "analytic.csv", index=False)
    theoretical = analytic.set_index(keys)
    means = []
    for group, c in df.groupby(keys):
        for metric in [
            "p_train",
            "p_test",
            "unresolved",
            "world_error",
            "target_error",
            "target_brier",
            "hard_world_brier",
        ]:
            expected = theoretical.loc[
                group, "target_error" if metric == "hard_world_brier" else metric
            ]
            observed = c[metric].mean()
            half = t.ppf(0.975, len(c) - 1) * c[metric].sem()
            means.append(
                dict(
                    zip(keys, group),
                    metric=metric,
                    analytic=expected,
                    observed=observed,
                    mc_low=observed - half,
                    mc_high=observed + half,
                )
            )
    pd.DataFrame(means).to_csv(out / "means.csv", index=False)
    config = {
        "seeds": seeds,
        "banks": len(seeds),
        "repeats_per_bank": repeats,
        "queries_per_repeat": 256,
        "df": shapes,
        "degrees": degrees,
        "simulated_n": sample_sizes,
        "representation_max_abs": df.network_max_abs.max(),
        "compute_start_utc": start.isoformat(),
        "compute_seconds": time.perf_counter() - timer,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args()
    default_seeds = range(6100, 6196) if args.confirm else range(1250, 1274)
    run(args.output, args.seeds or list(default_seeds), args.confirm)

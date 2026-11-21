"""Evaluate continuous identification and prediction risks for an unknown vote direction."""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.special import gamma, stdtr, stdtrit

from fusionbench.methods.majority_interval import fit_majority_interval

MAX_ANGLE = np.pi / 3


def mass(t, k):
    return 0.0 if t <= 0.0 else 0.5 * stdtr(k, -np.sqrt(k) / np.tan(t))


def mean_width(t, k, n):
    # Place quadrature points at the O(1/n) scales in source probability mass.
    f = mass(t, k)
    stop = mass(MAX_ANGLE, k)
    points = {0.0, t, MAX_ANGLE}
    for sign in (-1, 1):
        for level in (1, 5, 20, 40, 100):
            p = f + sign * level / n
            if 0 < p < stop:
                points.add(float(np.arctan2(np.sqrt(k), -stdtrit(k, 2 * p))))

    def integrand(a):
        return np.exp(n * np.log1p(-abs(mass(a, k) - f)))

    return quad(integrand, 0, MAX_ANGLE, points=sorted(points), epsabs=1e-13)[0]


def majority_label(raw, theta):
    third = (raw[:, 2] * np.cos(theta) + raw[:, 3] * np.sin(theta)) >= 0
    return ((raw[:, :2] >= 0).sum(1) + third >= 2).astype(int)


def one_bank(seed, k, truth_id):
    rng = np.random.default_rng([seed, k, truth_id])
    rows = []
    theta_fixed = [0.0, np.pi / 6, None][truth_id]
    for trial in range(32):
        theta = rng.uniform(0, MAX_ANGLE) if theta_fixed is None else theta_fixed
        x = rng.normal(size=(2048, 4))
        x[:, 2] = np.sign(x[:, 2]) * np.sqrt(rng.chisquare(k, size=len(x)))
        y = majority_label(x, theta)
        z = rng.normal(size=(512, 4))
        target = majority_label(z, theta)
        for n in [32, 128, 512, 2048]:
            interval = fit_majority_interval(x[:n], y[:n], MAX_ANGLE)
            if not interval.lower - 1e-12 <= theta <= interval.upper + 1e-12:
                raise ValueError(
                    "true parameter excluded from its own noiseless version interval"
                )
            lower = majority_label(z, interval.lower)
            upper = majority_label(z, interval.upper)
            width = interval.upper - interval.lower
            d = width / (2 * np.pi)
            mid = interval.predict_midpoint(z)
            minimax = interval.predict_minimax_positive(z)
            posterior = interval.predict_uniform_positive(z)
            rows.append(
                {
                    "seed": seed,
                    "k": k,
                    "true_angle": ["0", "30", "uniform"][truth_id],
                    "trial": trial,
                    "labels": n,
                    "theta": theta,
                    "lower": interval.lower,
                    "upper": interval.upper,
                    "ambiguity": d,
                    "observed_ambiguity": np.mean(lower != upper),
                    "midpoint_error": np.mean(mid != target),
                    "minimax_brier": np.mean((minimax - target) ** 2),
                    "posterior_brier": np.mean((posterior - target) ** 2),
                    "midpoint_endpoint_error": 0.5
                    * (np.mean(mid != lower) + np.mean(mid != upper)),
                    "minimax_endpoint_brier": 0.5
                    * (
                        np.mean((minimax - lower) ** 2)
                        + np.mean((minimax - upper) ** 2)
                    ),
                    "posterior_endpoint_brier": 0.5
                    * (
                        np.mean((posterior - lower) ** 2)
                        + np.mean((posterior - upper) ** 2)
                    ),
                }
            )
    return rows


def run(out, seeds):
    start = datetime.now(UTC)
    timer = time.perf_counter()
    out.mkdir(parents=True, exist_ok=True)
    rows, exact = [], []
    for k in [1, 7, 15]:
        for code in range(3):
            for seed in seeds:
                rows += one_bank(seed, k, code)
        for angle in [0, 30]:
            theta = np.deg2rad(angle)
            a = gamma((k + 1) / 2) / (2 * np.sqrt(np.pi) * gamma(k / 2))
            for n in [32, 128, 512, 2048, 8192, 65536, 1000000]:
                width = mean_width(theta, k, n)
                asym = (
                    gamma(1 + 1 / k) * (k / (a * n)) ** (1 / k)
                    if angle == 0
                    else 2 / (n * a * np.sin(theta) ** (k - 1))
                )
                exact.append(
                    {
                        "k": k,
                        "angle_degrees": angle,
                        "labels": n,
                        "width": width,
                        "target_ambiguity": width / (2 * np.pi),
                        "asymptotic_width": asym,
                    }
                )
    df = pd.DataFrame(rows)
    keys = ["k", "true_angle", "labels"]
    quantities = [
        "ambiguity",
        "observed_ambiguity",
        "midpoint_error",
        "minimax_brier",
        "posterior_brier",
        "midpoint_endpoint_error",
        "minimax_endpoint_brier",
        "posterior_endpoint_brier",
    ]
    df.to_csv(
        out / "trials.csv.gz", index=False, compression={"method": "gzip", "mtime": 0}
    )
    per_seed = df.groupby(keys + ["seed"])[quantities].mean()
    per_seed.to_csv(out / "banks.csv")
    per_seed.groupby(keys)[quantities].mean().to_csv(out / "means.csv")
    pd.DataFrame(exact).to_csv(out / "exact.csv", index=False)
    config = {
        "seeds": seeds,
        "trials_per_seed": 32,
        "queries_per_trial": 512,
        "max_angle_degrees": 60,
        "source_sizes": [32, 128, 512, 2048],
        "chi_df": [1, 7, 15],
        "compute_start_utc": start.isoformat(),
        "compute_seconds": time.perf_counter() - timer,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(1350, 1366)))
    args = parser.parse_args()
    run(args.output, args.seeds)

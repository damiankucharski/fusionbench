"""Separate radial invariance from direction-distribution bias in majority recovery."""

from __future__ import annotations

import argparse
import itertools
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, gamma

from fusionbench.eval.majority_moments import orthogonal
from fusionbench.eval.majority_routing import alignment, contrast
from fusionbench.methods.majority_moments import (
    fit_angular_majority_moments,
    fit_majority_moments,
    majority_axes_from_moments,
)


def population():
    d = 9
    truth = np.eye(d)[:3]
    c = np.sqrt(2 / np.pi)
    m = c / 2 * truth.sum(0)
    T = np.zeros((d, d, d))
    for u in truth:
        T -= c / 2 * np.einsum("i,j,k->ijk", u, u, u)
    for u, v, w in itertools.permutations(truth):
        T -= c**3 / 2 * np.einsum("i,j,k->ijk", u, v, w)
    echi = np.sqrt(2) * gamma((d + 1) / 2) / gamma(d / 2)
    first = np.sqrt(d) / echi * m
    first_I = first[:, None, None] * np.eye(d)
    sym = first_I + first_I.transpose(1, 0, 2) + first_I.transpose(1, 2, 0)
    corrected = d**1.5 / ((d + 1) * echi) * T
    naive = corrected - sym / (d + 1)
    out = {
        "d": d,
        "angular_correction": d / (d + 1),
        "tilted_first_component_ratio": 3.2,
        "tilted_theoretical_sum_cosine": 5.2 / np.sqrt(3 * (3.2**2 + 2)),
    }
    for method, tensor in [("corrected", corrected), ("naive", naive)]:
        sv = np.linalg.svd(tensor.reshape(d, -1), compute_uv=False)
        out[method] = {
            "singular_values": sv.tolist(),
            "rank": int(np.sum(sv > 1e-10)),
            "cosine": alignment(majority_axes_from_moments(first, tensor).axes, truth)[
                0
            ],
        }
    return out


def cell(seed):
    rng = np.random.default_rng(seed)
    truth = orthogonal(rng)
    true_sum = truth.sum(0) / np.sqrt(3)
    raw = rng.normal(size=(8192, 9))
    y = (raw @ truth.T > 0).sum(1) >= 2
    contaminated = raw.copy()
    contaminated[rng.random(len(raw)) < 0.02] *= 40
    proj = raw @ truth[0]
    tilted = (
        raw
        + (np.sign(proj) * np.sqrt(rng.chisquare(7, size=len(raw))) - proj)[:, None]
        * truth[0]
    )
    z = rng.normal(size=(4096, 9))
    test_y = (z @ truth.T > 0).sum(1) >= 2
    multiplier = np.full((len(z), 3), 0.5)
    multiplier[np.arange(len(z)), rng.integers(3, size=len(z))] = 4
    shift = z + ((z @ truth.T) * (multiplier - 1)) @ truth
    assert np.array_equal((tilted @ truth.T > 0).sum(1) >= 2, y)
    assert np.array_equal((shift @ truth.T > 0).sum(1) >= 2, test_y)
    metrics, geometry, checks = [], [], []
    for n in [512, 2048, 8192]:
        reference = {}
        for condition, arr in [
            ("clean", raw),
            ("radial", contaminated),
            ("tilted", tilted),
        ]:
            x = arr[:n]
            normalized = (
                np.sqrt(x.shape[1]) * x / np.linalg.norm(x, axis=1, keepdims=True)
            )
            fits = {
                "gaussian": fit_majority_moments(x, y[:n]),
                "angular": fit_angular_majority_moments(x, y[:n]),
                "naive_angular": fit_majority_moments(normalized, y[:n]),
            }
            for method, fit in fits.items():
                axes = fit.axes
                geometry.append(
                    {
                        "seed": seed,
                        "labels": n,
                        "condition": condition,
                        "method": method,
                        "matched_cosine": alignment(axes, truth)[0],
                        "sum_cosine": fit.sum_direction @ true_sum,
                    }
                )
                for query, inp in [("iid", z), ("gain", shift)]:
                    p = expit(5 * np.median(inp @ axes.T, axis=1))
                    metrics.append(
                        {
                            "seed": seed,
                            "labels": n,
                            "condition": condition,
                            "method": method,
                            "query": query,
                            "brier": np.mean((p - test_y) ** 2),
                            "accuracy": np.mean((p >= 0.5) == test_y),
                        }
                    )
                    if condition == "clean":
                        reference[method, query] = p
                    if condition == "radial":
                        checks.append(
                            {
                                "seed": seed,
                                "labels": n,
                                "method": method,
                                "query": query,
                                "max_abs": np.max(np.abs(p - reference[method, query])),
                            }
                        )
    return metrics, geometry, checks


def run(out, seeds):
    start = datetime.now(UTC)
    timer = time.perf_counter()
    out.mkdir(exist_ok=True, parents=True)
    means, geometry, checks = [], [], []
    for seed in seeds:
        m, g, c = cell(seed)
        means.extend(m)
        geometry.extend(g)
        checks.extend(c)
    metrics = pd.DataFrame(means)
    geom = pd.DataFrame(geometry)
    diffs = pd.DataFrame(checks)
    metrics.to_csv(out / "metrics.csv", index=False)
    geom.to_csv(out / "geometry.csv", index=False)
    diffs.to_csv(out / "invariance.csv", index=False)
    metrics.groupby(["labels", "condition", "query", "method"])[
        ["accuracy", "brier"]
    ].mean().to_csv(out / "means.csv")
    geom.groupby(["labels", "condition", "method"])[
        ["matched_cosine", "sum_cosine"]
    ].mean().to_csv(out / "geometry_means.csv")
    paired = []
    for metric, frame in [
        ("accuracy", metrics[metrics["query"] == "gain"]),
        ("matched_cosine", geom),
    ]:
        for (n, condition), group in frame.groupby(["labels", "condition"]):
            wide = group.pivot(index="seed", columns="method", values=metric)
            paired.append(
                contrast(
                    wide.angular - wide.gaussian,
                    labels=n,
                    condition=condition,
                    metric=metric,
                )
            )
    pd.DataFrame(paired).to_csv(out / "paired.csv", index=False)
    pop = population()
    (out / "population.json").write_text(json.dumps(pop, indent=2) + "\n")
    config = {
        "seeds": seeds,
        "train_sizes": [512, 2048, 8192],
        "dimensions": 9,
        "radial_probability": 0.02,
        "radial_multiplier": 40,
        "tilted_chi_df": 7,
        "direct_logit_scale": 5,
        "compute_start_utc": start.isoformat(),
        "compute_seconds": time.perf_counter() - timer,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(1100, 1116)))
    args = parser.parse_args()
    run(args.output, args.seeds)

"""Test conditional moment recovery with an informative or nuisance variance spike."""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import expit

from fusionbench.eval.lattice_compile import train
from fusionbench.eval.majority_moments import orthogonal
from fusionbench.eval.majority_routing import MajorityNetwork, alignment, contrast
from fusionbench.methods.majority import independent_majority_logit
from fusionbench.methods.majority_anchor import fit_spiked_majority


def cell(seed):
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.normal(size=(9, 4)), mode="reduced")
    dirs = (q * np.sign(np.diag(r))).T
    truth = dirs[:3]
    raw = rng.normal(size=(2048, 9))
    z = rng.normal(size=(4096, 9))
    yt = (z @ truth.T > 0).sum(1) >= 2
    scales = np.full((len(z), 3), 0.5)
    scales[np.arange(len(z)), rng.integers(3, size=len(z))] = 4
    shifted = z + ((z @ truth.T) * (scales - 1)) @ truth
    assert np.array_equal((shifted @ truth.T > 0).sum(1) >= 2, yt)
    amplitude = np.sqrt(rng.chisquare(7, size=len(raw)))
    init = orthogonal(rng)
    rows, geometry, routes = [], [], []
    for n in [512, 2048]:
        schedule = rng.integers(0, n, size=(600, 64))
        for cond, spike in [
            ("clean", None),
            ("vote", dirs[0]),
            ("nuisance", dirs[3]),
            ("mixed", (dirs[0] + dirs[3]) / np.sqrt(2)),
        ]:
            x = raw[:n]
            if spike is not None:
                proj = x @ spike
                x = x + ((np.sign(proj) * amplitude[:n] - proj)[:, None]) * spike
            y = (x @ truth.T > 0).sum(1) >= 2
            auto = fit_spiked_majority(x, y)
            routes.append(
                {
                    "seed": seed,
                    "labels": n,
                    "condition": cond,
                    "route": auto.route,
                    "agreements": auto.agreement,
                    "eigenratio": auto.eigenratio,
                    "contradictions": auto.contradictions,
                }
            )
            methods = {"auto": (auto.axes, 0, 5)}
            for route in ["angular", "anchor", "drop"]:
                fit = (
                    auto
                    if route == auto.route
                    else fit_spiked_majority(x, y, force=route)
                )
                methods[route] = (fit.axes, 0, 5)
            net = MajorityNetwork(independent_majority_logit)
            with torch.no_grad():
                net.leaves.weight.copy_(torch.from_numpy(init))
                net.leaves.bias.zero_()
            train(net, torch.from_numpy(x), torch.from_numpy(y.astype(float)), schedule)
            methods["smooth"] = (
                net.leaves.weight.detach().numpy(),
                net.leaves.bias.detach().numpy(),
                1,
            )
            for method, (axes, bias, scale) in methods.items():
                geometry.append(
                    {
                        "seed": seed,
                        "labels": n,
                        "condition": cond,
                        "method": method,
                        "matched_cosine": alignment(axes, truth)[0],
                    }
                )
                for query, inputs in [("iid", z), ("gain", shifted)]:
                    p = expit(scale * np.median(inputs @ axes.T + bias, axis=1))
                    rows.append(
                        {
                            "seed": seed,
                            "labels": n,
                            "condition": cond,
                            "method": method,
                            "query": query,
                            "brier": np.mean((p - yt) ** 2),
                            "accuracy": np.mean((p >= 0.5) == yt),
                        }
                    )
    return rows, geometry, routes


def run(out, seeds):
    torch.set_num_threads(1)
    start = datetime.now(UTC)
    timer = time.perf_counter()
    out.mkdir(exist_ok=True, parents=True)
    raw, geom, choices = [], [], []
    for s in seeds:
        m, g, c = cell(s)
        raw.extend(m)
        geom.extend(g)
        choices.extend(c)
    metrics = pd.DataFrame(raw)
    geometries = pd.DataFrame(geom)
    routes = pd.DataFrame(choices)
    metrics.to_csv(out / "metrics.csv", index=False)
    geometries.to_csv(out / "geometry.csv", index=False)
    routes.to_csv(out / "routes.csv", index=False)
    metrics.groupby(["labels", "condition", "query", "method"])[
        ["accuracy", "brier"]
    ].mean().to_csv(out / "means.csv")
    geometries.groupby(["labels", "condition", "method"]).matched_cosine.mean().to_csv(
        out / "geometry_means.csv"
    )
    routes.groupby(["labels", "condition", "route"]).size().rename("count").to_csv(
        out / "route_counts.csv"
    )
    paired = []
    for metric, frame in [
        ("accuracy", metrics[metrics["query"] == "gain"]),
        ("matched_cosine", geometries),
    ]:
        for (n, cond), group in frame.groupby(["labels", "condition"]):
            wide = group.pivot(index="seed", columns="method", values=metric)
            for other in ["angular", "anchor", "drop", "smooth"]:
                paired.append(
                    contrast(
                        wide.auto - wide[other],
                        labels=n,
                        condition=cond,
                        metric=metric,
                        reference=other,
                    )
                )
    pd.DataFrame(paired).to_csv(out / "paired.csv", index=False)
    config = {
        "seeds": seeds,
        "dimensions": 9,
        "train_sizes": [512, 2048],
        "test_rows": 4096,
        "variance_spike_chi_df": 7,
        "eigenratio_threshold": 2,
        "agreement_threshold": 0.625,
        "direct_logit_scale": 5,
        "smooth_steps": 600,
        "smooth_lr": 0.015,
        "compute_start_utc": start.isoformat(),
        "compute_seconds": time.perf_counter() - timer,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(1200, 1216)))
    args = parser.parse_args()
    run(args.output, args.seeds)

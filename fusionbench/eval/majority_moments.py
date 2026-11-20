"""Test an explicit Gaussian moment estimator for hidden majority directions."""

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
from fusionbench.eval.majority_routing import MajorityNetwork, alignment, contrast
from fusionbench.methods.majority import independent_majority_logit
from fusionbench.methods.majority_moments import fit_majority_moments


def orthogonal(rng):
    q, r = np.linalg.qr(rng.normal(size=(9, 3)), mode="reduced")
    return (q * np.sign(np.diag(r))).T


def summarize_axes(axes, truth):
    spl, _ = np.linalg.qr(axes.T, mode="reduced")
    return {
        "matched_cosine": alignment(axes, truth)[0],
        "span_overlap": np.sum((truth @ spl) ** 2) / 3,
    }


def cell(seed):
    rng = np.random.default_rng([seed, 0])
    truth = orthogonal(rng)
    x = rng.normal(size=(8192, 9))
    z = rng.normal(size=(4096, 9))
    y = (x @ truth.T > 0).sum(1) >= 2
    yt = (z @ truth.T > 0).sum(1) >= 2
    scales = np.full((len(z), 3), 0.5)
    scales[np.arange(len(z)), rng.integers(3, size=len(z))] = 4
    shifted = z + ((z @ truth.T) * (scales - 1)) @ truth
    assert np.array_equal((shifted @ truth.T > 0).sum(1) >= 2, yt)
    queries = {"iid": z, "gain": shifted}
    rows, geometry = [], []

    def add(mode, name, axes, bias=0.0, scale=5.0):
        geometries = [summarize_axes(a, truth) for a in axes if len(a) == 3]
        if geometries:
            geometry.append(
                dict(
                    seed=seed,
                    labels=n,
                    training=mode,
                    method=name,
                    **{k: np.mean([g[k] for g in geometries]) for k in geometries[0]},
                )
            )
        for query, raw in queries.items():
            each = [expit(scale * np.median(raw @ a.T + bias, axis=1)) for a in axes]
            rows.append(
                {
                    "seed": seed,
                    "labels": n,
                    "training": mode,
                    "method": name,
                    "query": query,
                    "brier": np.mean([np.mean((p - yt) ** 2) for p in each]),
                    "accuracy": np.mean([np.mean((p >= 0.5) == yt) for p in each]),
                }
            )

    for n in (128, 512, 2048, 8192):
        raw, target = x[:n], y[:n]
        fit = fit_majority_moments(raw, target)
        control = fit_majority_moments(raw, rng.permutation(target))
        first = np.mean(
            raw * ((2 * target.astype(float) - 1)[:, None]), axis=0, keepdims=True
        )
        add("direct", "moments", [fit.axes])
        add(
            "direct",
            "phase_marginal",
            [fit.with_phase_offset(a) for a in np.arange(12) * 2 * np.pi / 36],
        )
        add("direct", "label_permutation", [control.axes])
        add("direct", "first", [first / np.linalg.norm(first)])
        add("direct", "oracle", [truth])
        if n not in (512, 2048):
            continue
        X, Y = torch.from_numpy(raw), torch.from_numpy(target.astype(float))
        schedule = rng.integers(0, n, size=(600, 64))
        for method, init in [
            ("moments", fit.axes),
            ("random_phase", fit.with_phase_offset(rng.uniform(0, 2 * np.pi / 3))),
            ("random_axes", orthogonal(rng)),
        ]:
            net = MajorityNetwork(independent_majority_logit)
            with torch.no_grad():
                net.leaves.weight.copy_(torch.from_numpy(init))
                net.leaves.bias.zero_()
            train(net, X, Y, schedule)
            add(
                "refined",
                method,
                [net.leaves.weight.detach().numpy()],
                net.leaves.bias.detach().numpy(),
                1.0,
            )
    return rows, geometry


def run(out, seeds):
    torch.set_num_threads(1)
    start = datetime.now(UTC)
    timer = time.perf_counter()
    out.mkdir(parents=True, exist_ok=True)
    rows, geometry = [], []
    for seed in seeds:
        r, g = cell(seed)
        rows.extend(r)
        geometry.extend(g)
    df, geo = pd.DataFrame(rows), pd.DataFrame(geometry)
    df.to_csv(out / "metrics.csv", index=False)
    geo.to_csv(out / "geometry.csv", index=False)
    df.groupby(["labels", "training", "query", "method"])[
        ["accuracy", "brier"]
    ].mean().to_csv(out / "means.csv")
    geo.groupby(["labels", "training", "method"])[
        ["matched_cosine", "span_overlap"]
    ].mean().to_csv(out / "geometry_means.csv")
    paired = []
    for (n, mode, q), group in df.groupby(["labels", "training", "query"]):
        for metric in ["accuracy", "brier"]:
            wide = group.pivot(index="seed", columns="method", values=metric)
            refs = (
                ["phase_marginal"]
                if mode == "direct"
                else ["random_phase", "random_axes"]
            )
            for ref in refs:
                paired.append(
                    contrast(
                        wide.moments - wide[ref],
                        labels=n,
                        training=mode,
                        query=q,
                        metric=metric,
                        reference=ref,
                    )
                )
    for (n, mode), group in geo.groupby(["labels", "training"]):
        wide = group.pivot(index="seed", columns="method", values="matched_cosine")
        refs = (
            ["phase_marginal"] if mode == "direct" else ["random_phase", "random_axes"]
        )
        for ref in refs:
            paired.append(
                contrast(
                    wide.moments - wide[ref],
                    labels=n,
                    training=mode,
                    query="axes",
                    metric="matched_cosine",
                    reference=ref,
                )
            )
    pd.DataFrame(paired).to_csv(out / "paired.csv", index=False)
    config = {
        "seeds": seeds,
        "labels": [128, 512, 2048, 8192],
        "refined_at": [512, 2048],
        "test_rows": 4096,
        "phase_grid": 12,
        "direct_scale": 5.0,
        "adam_steps": 600,
        "learning_rate": 0.015,
        "batch_size": 64,
        "compute_start_utc": start.isoformat(),
        "compute_seconds": time.perf_counter() - timer,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(950, 962)))
    args = parser.parse_args()
    run(args.output, args.seeds)

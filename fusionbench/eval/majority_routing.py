"""Measure latent directions under two supervised training rules for majority."""

from __future__ import annotations

import argparse
import itertools
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import t
from torch import nn

from fusionbench.eval.lattice_compile import train
from fusionbench.methods.majority import independent_majority_logit, median_logit


class MajorityNetwork(nn.Module):
    def __init__(self, decoder, tied_start=False):
        super().__init__()
        self.leaves = nn.Linear(9, 3).double()
        self.decoder = decoder
        if tied_start:
            with torch.no_grad():
                self.leaves.weight[1:].copy_(self.leaves.weight[:1].clone())
                self.leaves.bias[1:].copy_(self.leaves.bias[:1].clone())

    def forward(self, x):
        return self.decoder(self.leaves(x))


def alignment(learned, truth):
    weight = learned / np.linalg.norm(learned, axis=1, keepdims=True)
    cos = weight @ truth.T
    score = max(np.mean(cos[np.arange(3), p]) for p in itertools.permutations(range(3)))
    return score, (weight @ weight.T)[np.triu_indices(3, 1)].mean()


def cell(seed):
    rng = np.random.default_rng([seed, 0])
    q, r = np.linalg.qr(rng.normal(size=(9, 3)), mode="reduced")
    axes = (q * np.sign(np.diag(r))).T
    x = rng.normal(size=(2048, 9))
    z = rng.normal(size=(4096, 9))
    y = (x @ axes.T > 0).sum(1) >= 2
    test_y = (z @ axes.T > 0).sum(1) >= 2
    scales = np.full((len(z), 3), 0.5)
    scales[np.arange(len(z)), rng.integers(3, size=len(z))] = 4
    shifted = z + ((z @ axes.T) * (scales - 1)) @ axes
    assert np.array_equal((shifted @ axes.T > 0).sum(1) >= 2, test_y)
    queries = {"iid": torch.from_numpy(z), "gain": torch.from_numpy(shifted)}
    rows, geometry = [], []
    for n in [512, 2048]:
        X, Y = torch.from_numpy(x[:n]), torch.from_numpy(y[:n].astype(float))
        schedule = rng.integers(0, n, size=(600, 64))
        for name in ["hard", "smooth", "smooth_tie", "relu", "linear"]:
            torch.manual_seed(19 + seed)
            if name == "relu":
                net = nn.Sequential(
                    nn.Linear(9, 3), nn.ReLU(), nn.Linear(3, 1)
                ).double()
            elif name == "linear":
                net = nn.Linear(9, 1).double()
            else:
                decoder = median_logit if name == "hard" else independent_majority_logit
                net = MajorityNetwork(decoder, tied_start=name == "smooth_tie")
            train(net, X, Y, schedule)
            if isinstance(net, MajorityNetwork):
                cos, mutual = alignment(net.leaves.weight.detach().numpy(), axes)
                geometry.append(
                    {
                        "seed": seed,
                        "labels": n,
                        "training": name,
                        "mean_matched_cosine": cos,
                        "mean_pairwise_cosine": mutual,
                    }
                )
            with torch.no_grad():
                for shift, query in queries.items():
                    if isinstance(net, MajorityNetwork):
                        latent = net.leaves(query)
                        outputs = {
                            "hard": median_logit(latent),
                            "smooth": independent_majority_logit(latent),
                        }
                    else:
                        outputs = {"native": net(query)}
                    for dec, logit in outputs.items():
                        p = logit.sigmoid().reshape(-1).numpy()
                        rows.append(
                            {
                                "seed": seed,
                                "labels": n,
                                "training": name,
                                "decoder": dec,
                                "query": shift,
                                "brier": np.mean((p - test_y) ** 2),
                                "accuracy": np.mean((p >= 0.5) == test_y),
                            }
                        )
    return rows, geometry


def contrast(diff, **settings):
    half = t.ppf(0.975, len(diff) - 1) * diff.sem()
    return dict(
        **settings, delta=diff.mean(), low=diff.mean() - half, high=diff.mean() + half
    )


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
    df = pd.DataFrame(rows)
    geo = pd.DataFrame(geometry)
    df.to_csv(out / "metrics.csv", index=False)
    geo.to_csv(out / "geometry.csv", index=False)
    geo.groupby(["labels", "training"])[
        ["mean_matched_cosine", "mean_pairwise_cosine"]
    ].mean().to_csv(out / "geometry_means.csv")
    df.groupby(["labels", "query", "training", "decoder"])[
        ["brier", "accuracy"]
    ].mean().to_csv(out / "means.csv")
    contrasts = []
    for (n, q), group in df[df.decoder == "hard"].groupby(["labels", "query"]):
        for metric in ["brier", "accuracy"]:
            wide = group.pivot(index="seed", columns="training", values=metric)
            contrasts.append(
                contrast(wide.smooth - wide.hard, labels=n, query=q, metric=metric)
            )
    for n, group in geo.groupby("labels"):
        wide = group.pivot(
            index="seed", columns="training", values="mean_matched_cosine"
        )
        contrasts.append(
            contrast(
                wide.smooth - wide.hard,
                labels=n,
                query="axes",
                metric="mean_matched_cosine",
            )
        )
    pd.DataFrame(contrasts).to_csv(out / "paired.csv", index=False)
    config = {
        "seeds": seeds,
        "labels": [512, 2048],
        "test_rows": 4096,
        "batch_size": 64,
        "adam_steps": 600,
        "learning_rate": 0.015,
        "gain": [4, 0.5, 0.5],
        "compute_start_utc": start.isoformat(),
        "compute_seconds": time.perf_counter() - timer,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(900, 912)))
    args = parser.parse_args()
    run(args.output, args.seeds)

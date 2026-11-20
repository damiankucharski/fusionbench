"""Separate finite lattice search failure from the raw student's view prior."""

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
from sklearn.linear_model import LogisticRegression
from torch import nn
from torch.nn import functional as F

from fusionbench.eval.lattice_compile import sample, train
from fusionbench.methods.okapi import OKAPIMethod


class MaskedMedian(nn.Module):
    def __init__(self, mask):
        super().__init__()
        self.leaves = nn.Linear(9, 3).double()
        self.register_buffer("mask", torch.as_tensor(mask, dtype=torch.float64))

    def forward(self, x):
        a, b, c = F.linear(x, self.leaves.weight * self.mask, self.leaves.bias).unbind(
            -1
        )
        return torch.maximum(torch.minimum(a, b), torch.minimum(torch.maximum(a, b), c))


def make_rows(seed, target, predictions, subset, **condition):
    out = []
    for method, p in predictions.items():
        out.append(
            dict(
                seed=seed,
                subset=subset,
                **condition,
                method=method,
                accuracy=np.mean((p >= 0.5) == target),
                brier=np.mean((p - target) ** 2),
            )
        )
    return out


def search(seed, directions, x, y, z, target):
    from okapi.lattice_search import fit_small_lattice

    source, local_labels, _ = sample([seed, 1], directions, 512)

    def score_bank():
        out = []
        for j in range(3):
            view = slice(3 * j, 3 * j + 3)
            model = LogisticRegression(C=1, tol=1e-8, max_iter=4000).fit(
                source[:, view], local_labels[:, j]
            )
            out.append(
                (
                    model.predict_proba(x[:, view])[:, 1],
                    model.predict_proba(z[:, view])[:, 1],
                )
            )
        return [np.column_stack(split) for split in zip(*out)]

    fit_p, test_p = score_bank()
    exact = fit_small_lattice(fit_p, y)
    cube = np.array(list(itertools.product((0, 1), repeat=3)))
    corner_p = 0.01 + 0.98 * cube
    corner_y = cube.sum(1) >= 2
    gp = OKAPIMethod(
        population_size=4,
        n_restarts=1,
        n_seeds=12,
        generations=30,
        tournament_size=3,
        population_multiplier=2,
        fitness="brier",
        seed=100 + seed,
        max_nodes=31,
        operator_family="minmax",
    )

    def with_alias(p):
        cols = p[:, [0, 1, 2, 0]]
        return np.stack((1 - cols, cols), axis=-1)

    try:
        gp.fit(with_alias(fit_p), y.astype(int), task="binary", n_classes=2)
        gp_test = gp.predict(with_alias(test_p))[:, 1]
        gp_corners = gp.predict(with_alias(corner_p))[:, 1]
        stats = {
            "seed": seed,
            "candidates": exact.n_candidates,
            "terms": repr(exact.terms),
            "gp_tree": gp.tree_repr(),
            "gp_correct_corners": np.sum((gp_corners >= 0.5) == corner_y),
            "exact_correct_corners": np.sum(
                (exact.predict(corner_p) >= 0.5) == corner_y
            ),
        }
    finally:
        gp._tmp.cleanup()
    pred = {
        "gp": gp_test,
        "exact": exact.predict(test_p),
        "source_median": np.median(test_p, axis=1),
    }
    return make_rows(seed, target, pred, "fusion", layout="aligned", labels=512), stats


def cell(seed):
    rng = np.random.default_rng([seed, 0])
    dirs = rng.normal(size=(3, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    x, _, y = sample([seed, 2], dirs, 512)
    z, _, target = sample([seed, 3], dirs, 4096)
    rows, programs = search(seed, dirs, x, y, z, target)
    q, r = np.linalg.qr(rng.normal(size=(9, 9)))
    q *= np.sign(np.diag(r))
    fixed = np.repeat(np.eye(3), 3, axis=1)
    crossed = np.concatenate([np.eye(3)[rng.permutation(3)] for _ in range(3)], axis=1)
    masks = {
        "original_blocks": fixed,
        "crossed_blocks": crossed,
        "dense_leaves": np.ones((3, 9)),
    }
    for n in (128, 512):
        schedule = rng.integers(0, n, size=(600, 64))
        Y = torch.as_tensor(y[:n], dtype=torch.float64)
        for layout, raw, query in [("aligned", x, z), ("rotated", x @ q, z @ q)]:
            X, Z = torch.from_numpy(raw[:n]), torch.from_numpy(query)
            pred = {}
            for name, mask in list(masks.items()) + [("linear", None)]:
                torch.manual_seed(19 + seed)
                net = (
                    MaskedMedian(mask) if mask is not None else nn.Linear(9, 1).double()
                )
                train(net, X, Y, schedule)
                with torch.no_grad():
                    pred[name] = net(Z).reshape(-1).sigmoid().numpy()
            rows += make_rows(seed, target, pred, "student", layout=layout, labels=n)
    return rows, programs


def run(output, seeds):
    from loguru import logger

    logger.disable("okapi")
    torch.set_num_threads(1)
    start = datetime.now(UTC)
    timer = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    metrics, programs = [], []
    for s in seeds:
        m, p = cell(s)
        metrics.extend(m)
        programs.append(p)
    pd.DataFrame(programs).to_csv(output / "programs.csv", index=False)
    frame = pd.DataFrame(metrics)
    cond = ["subset", "labels", "layout"]
    frame.to_csv(output / "metrics.csv", index=False)
    frame.groupby(cond + ["method"])[["brier", "accuracy"]].mean().to_csv(
        output / "means.csv"
    )
    contrasts = []
    for group, cell_df in frame.groupby(cond):
        candidates = (
            ("exact", ["gp"])
            if group[0] == "fusion"
            else ("original_blocks", ["crossed_blocks", "dense_leaves", "linear"])
        )
        for metric in ["brier", "accuracy"]:
            wide = cell_df.pivot(index="seed", columns="method", values=metric)
            for reference in candidates[1]:
                diff = wide[candidates[0]] - wide[reference]
                half = t.ppf(0.975, len(diff) - 1) * diff.sem()
                contrasts.append(
                    dict(
                        zip(cond, group),
                        method=candidates[0],
                        reference=reference,
                        metric=metric,
                        delta=diff.mean(),
                        low=diff.mean() - half,
                        high=diff.mean() + half,
                    )
                )
    pd.DataFrame(contrasts).to_csv(output / "paired.csv", index=False)
    config = {
        "seeds": seeds,
        "expert_rows": 512,
        "fusion_rows": 512,
        "student_rows": [128, 512],
        "test_rows": 4096,
        "gp_generations": 30,
        "adam_steps": 600,
        "learning_rate": 0.015,
        "batch_size": 64,
        "compute_start_utc": start.isoformat(),
        "compute_seconds": time.perf_counter() - timer,
    }
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(850, 862)))
    args = parser.parse_args()
    run(args.output, args.seeds)

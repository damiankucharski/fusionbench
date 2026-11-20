"""Evaluate exact compilation and fresh supervised training of evolved min/max circuits."""

from __future__ import annotations

import argparse
import itertools
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from torch import nn
from torch.nn import functional as F

from fusionbench.methods.okapi import OKAPIMethod


def sample(seed, directions, n):
    x = np.random.default_rng(seed).normal(size=(n, 3, 3))
    bits = (x * directions).sum(axis=-1) >= 0
    return x.reshape(n, 9), bits, bits.sum(axis=1) >= 2


def known_median():
    from okapi.node import MaxNode, MinNode, ValueNode
    from okapi.tree import Tree

    def leaf(i):
        return ValueNode(None, np.zeros((1, 1)), f"m{i}.npy")

    # max(min(a,b), min(max(a,b),c)).
    root = leaf(0)
    root.add_child(MinNode([leaf(1)]))
    other = leaf(0)
    other.add_child(MaxNode([leaf(1)]))
    other.add_child(MinNode([leaf(2)]))
    root.add_child(MaxNode([other]))
    return Tree.create_tree_from_root(root)


def train(net, x, y, schedule):
    optimizer = torch.optim.Adam(net.parameters(), lr=0.015)
    for indices in schedule:
        logits = net(x[indices]).reshape(-1)
        loss = F.binary_cross_entropy_with_logits(logits, y[indices])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def cell(seed):
    from okapi.lattice import compile_binary_lattice
    from okapi.node import MaxNode, MinNode, ValueNode

    rng = np.random.default_rng([seed, 0])
    directions = rng.normal(size=(3, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    a, bits, _ = sample([seed, 1], directions, 512)
    x, ybits, y = sample([seed, 2], directions, 512)
    z, _, target = sample([seed, 3], directions, 4096)
    models = []
    affines = {}
    for j in range(3):
        section = slice(3 * j, 3 * j + 3)
        m = LogisticRegression(C=1, tol=1e-8, max_iter=4000).fit(
            a[:, section], bits[:, j]
        )
        weight = np.zeros(9)
        weight[section] = m.coef_[0]
        affines[f"m{j}.npy"] = (weight, m.intercept_[0])
        models.append(m)
    affines["m3.npy"] = affines["m0.npy"]

    def probs(raw):
        ps = [m.predict_proba(raw[:, 3 * j : 3 * j + 3]) for j, m in enumerate(models)]
        return np.stack(ps + [ps[0]], axis=1)

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
    try:
        gp.fit(probs(x), y.astype(int), task="binary", n_classes=2)
        test_probs = probs(z)
        native = gp.predict(test_probs)[:, 1]
        cube = np.array(list(itertools.product((0, 1), repeat=3)))
        truth = cube.sum(1) >= 2
        high_low = 0.01 + 0.98 * cube[:, [0, 1, 2, 0]]
        corners = np.stack((1 - high_low, high_low), axis=-1)
        decisions = gp.predict(corners)[:, 1] > 0.5
        compiled = compile_binary_lattice(gp.best_tree, affines)
        plain = compile_binary_lattice(gp.best_tree, affines, signature_limit=0)
        oracle = compile_binary_lattice(known_median(), affines)
        shell = gp.best_tree.copy()
        inner = gp.best_tree.copy().root
        inner.add_child(MinNode([ValueNode(None, np.zeros((1, 1)), "m2.npy")]))
        shell.root.add_child(MaxNode([inner]))
        # Build newly connected tree metadata from its root.
        shell = type(shell).create_tree_from_root(shell.root)
        reduced = compile_binary_lattice(shell, affines)
        X, Y, Z = (
            torch.from_numpy(x),
            torch.from_numpy(y.astype(float)),
            torch.from_numpy(z),
        )

        def predict(net):
            with torch.no_grad():
                return net(Z).sigmoid().reshape(-1).numpy()

        output = {
            "native_tree": native,
            "compiled": predict(compiled),
            "source_median": predict(oracle),
        }
        shell_p = predict(reduced)
        stats = dict(
            seed=seed,
            tree=gp.tree_repr(),
            nodes=int(gp.best_nodes),
            match_corners=int((decisions == truth).sum()),
            gates_original=compiled.original_gates,
            gates_syntax=len(plain.gates),
            gates_compiled=len(compiled.gates),
            active_affines=compiled.leaves.out_features,
            shell_gates_original=reduced.original_gates,
            shell_gates_compiled=len(reduced.gates),
            native_max_abs=np.abs(native - output["compiled"]).max(),
            shell_max_abs=np.abs(native - shell_p).max(),
        )
        schedule = rng.integers(0, len(X), size=(600, 64))
        for name, net in [
            ("learned_topology", compiled),
            ("median_topology", oracle),
            (
                "dense",
                nn.Sequential(nn.Linear(9, 8), nn.ReLU(), nn.Linear(8, 1)).double(),
            ),
            ("linear", nn.Linear(9, 1).double()),
        ]:
            torch.manual_seed(19 + seed)
            for layer in net.modules():
                if isinstance(layer, nn.Linear):
                    layer.reset_parameters()
            train(net, X, Y, schedule)
            output[name] = predict(net)
        rows = [
            dict(
                seed=seed,
                method=name,
                brier=np.mean((p - target) ** 2),
                accuracy=np.mean((p >= 0.5) == target),
            )
            for name, p in output.items()
        ]
        return rows, stats
    finally:
        gp._tmp.cleanup()


def run(output, seeds):
    from loguru import logger
    from scipy.stats import t

    logger.disable("okapi")
    start = datetime.now(timezone.utc)
    timer = time.perf_counter()
    seeds = list(seeds)
    torch.set_num_threads(1)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows, stats = [], []
    for seed in seeds:
        r, s = cell(seed)
        rows.extend(r)
        stats.append(s)
    df = pd.DataFrame(rows)
    df.to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(stats).to_csv(output / "programs.csv", index=False)
    df.groupby("method")[["accuracy", "brier"]].mean().to_csv(output / "means.csv")
    contrasts = []
    for measure in ["accuracy", "brier"]:
        wide = df.pivot(index="seed", columns="method", values=measure)
        for ref in ["compiled", "median_topology", "dense", "linear"]:
            diff = wide["learned_topology"] - wide[ref]
            half = t.ppf(0.975, len(diff) - 1) * diff.sem()
            contrasts.append(
                dict(
                    metric=measure,
                    reference=ref,
                    delta=diff.mean(),
                    low=diff.mean() - half,
                    high=diff.mean() + half,
                )
            )
    pd.DataFrame(contrasts).to_csv(output / "paired.csv", index=False)
    (output / "config.json").write_text(
        json.dumps(
            dict(
                seeds=seeds,
                generations=30,
                fit_rows=512,
                source_concept_rows=512,
                test_rows=4096,
                adam_steps=600,
                batch_size=64,
                lr=0.015,
                compute_start_utc=start.isoformat(),
                compute_seconds=time.perf_counter() - timer,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(800, 808)))
    args = parser.parse_args()
    run(args.output, args.seeds)

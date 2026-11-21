"""Compute bank-paired registry contrasts and exact probabilities of template coverage."""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from itertools import permutations
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t


def all_orderings(width):
    corners = np.arange(2**width, dtype=int)
    signals = [
        sum(1 << i for i in np.flatnonzero((corners >> bit) & 1).tolist())
        for bit in range(width)
    ]
    branch, chain = Counter(), Counter()
    for ia, ib, ic, id_ in permutations(range(width), 4):
        a, b, c, d = (signals[i] for i in (ia, ib, ic, id_))
        branch[(a & b) | (c & d)] += 1
        branch[(a | b) & (c | d)] += 1
        chain[(((a & b) | c) & d)] += 1
        chain[(((a | b) & c) | d)] += 1
    assert not (branch.keys() & chain.keys())
    rows = []
    for shape, pool, orbit_size in (("branch", branch, 6), ("chain", chain, 24)):
        n = len(pool)
        if n != comb(width, 4) * orbit_size or len(set(pool.values())) != 1:
            raise AssertionError(
                "Coverage probability differs by the identity of the true rule"
            )
        coverage = 1 - (1 - 1 / n) ** 30
        rows.append(
            {
                "width": width,
                "method": shape,
                "unique_functions": n,
                "plans": sum(pool.values()),
                "exact_match_per_proposal": 1 / n,
                "exact_match_in_30": coverage,
                "expected_distinct_in_30": n * coverage,
            }
        )
    return rows


def interval(values):
    values = np.asarray(values)
    mean = float(np.mean(values))
    half = t.ppf(0.975, len(values) - 1) * np.std(values, ddof=1) / np.sqrt(len(values))
    return mean, mean - half, mean + half


def registry(folder):
    raw = pd.read_csv(folder / "standard.csv.gz", dtype={"checkpoint": str})
    keys = ["scenario", "split", "data_seed", "method", "checkpoint", "selection"]
    metrics = [
        "brier",
        "accuracy",
        "macro_f1",
        "nodes",
        "archive_trees",
        "archive_unique_fit_predictions",
        "seed_unique_fit_predictions",
        "search_seconds",
    ]
    draws = raw.groupby(keys, as_index=False)[metrics].mean()
    means = draws.groupby([k for k in keys if k != "data_seed"], as_index=False)[
        metrics
    ].mean()
    means.to_csv(folder / "standard-means.csv", index=False)
    fixed = (
        draws.groupby([k for k in keys if k != "scenario"], as_index=False)[metrics]
        .mean()
        .assign(scenario="fixed_suite")
    )
    diffs = []
    for key, g in pd.concat([draws, fixed]).groupby(
        ["scenario", "split", "checkpoint", "selection"]
    ):
        if not {"branch", "chain", "stock"}.issubset(g.method):
            continue
        for metric in metrics:
            wide = g.pivot(index="data_seed", columns="method", values=metric)
            for left, right in [
                ("branch", "chain"),
                ("branch", "stock"),
                ("chain", "stock"),
            ]:
                diff = wide[left] - wide[right]
                assert len(diff) == 6 and diff.notna().all()
                point, lo, hi = interval(diff)
                diffs.append(
                    {
                        "scenario": key[0],
                        "split": key[1],
                        "checkpoint": key[2],
                        "selection": key[3],
                        "metric": metric,
                        "left": left,
                        "right": right,
                        "mean_left": wide[left].mean(),
                        "mean_right": wide[right].mean(),
                        "difference": point,
                        "ci_low": lo,
                        "ci_high": hi,
                        "min_bank": diff.min(),
                        "max_bank": diff.max(),
                    }
                )
    pd.DataFrame(diffs).to_csv(folder / "standard-paired.csv.gz", index=False)
    fixed.groupby(["split", "method", "checkpoint", "selection"], as_index=False)[
        metrics
    ].mean().to_csv(folder / "standard-suite.csv", index=False)


def plot(folder):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    focal = pd.read_csv(folder / "focal-means.csv", dtype={"checkpoint": str})
    reg = pd.read_csv(folder / "standard-paired.csv.gz", dtype={"checkpoint": str})
    fig, axs = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    colors = {"branch": "#b44c19", "chain": "#1e6f83", "stock": "#4b5158"}
    markers = {"branch": "o", "chain": "s", "stock": "^"}
    for row, law in enumerate(("branch", "chain")):
        for col, width in enumerate((4, 8)):
            ax = axs[row, col]
            sel = focal[
                (focal.law == law)
                & (focal.width == width)
                & (focal.selector == "validation")
            ]
            for method, color in colors.items():
                v = (
                    sel[sel.method == method]
                    .set_index("checkpoint")
                    .loc[["initial", "600", "1200"], "iid_excess"]
                )
                ax.plot(
                    range(3),
                    v.values,
                    color=color,
                    marker=markers[method],
                    label=method.capitalize(),
                )
            ax.set_title(
                f"{law.capitalize()} truth, {width} inputs", loc="left", fontsize=12
            )
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", alpha=0.23)
            ax.set_xticks(range(3), ["Initialization", "600", "1,200"])
            if col == 0:
                ax.set_ylabel("Exact IID excess Brier")
            if row == 1:
                ax.set_xlabel("Paid GP fitness evaluations")
    fig.suptitle("Tree shape and search under two known fusion rules", fontsize=15)
    fig.legend(
        *axs[0, 0].get_legend_handles_labels(),
        loc="lower center",
        ncol=3,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(folder / "geometry-search.png", dpi=165)
    plt.close(fig)
    subset = reg[
        (reg.metric == "brier")
        & (reg.checkpoint == "1200")
        & (reg.selection == "validation")
        & (reg.left == "branch")
        & (reg.right == "chain")
        & (reg.split.isin(["iid", "shift"]))
    ]
    order = [*sorted(set(subset.scenario) - {"fixed_suite"}), "fixed_suite"]
    fig, axs = plt.subplots(1, 2, figsize=(10, 7.2), sharey=True)
    for ax, split in zip(axs, ["iid", "shift"]):
        d = subset[subset.split == split].set_index("scenario")
        ax.axvline(0, color="#464646", linewidth=1)
        for i, name in enumerate(order):
            if name not in d.index:
                continue
            x = d.loc[name]
            ax.errorbar(
                x.difference,
                i,
                xerr=[[x.difference - x.ci_low], [x.ci_high - x.difference]],
                fmt="D" if name == "fixed_suite" else "o",
                markersize=4,
                color="#1e6f83",
                capsize=2,
            )
        ax.set_title("IID" if split == "iid" else "Predefined shift", loc="left")
        ax.grid(axis="x", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("Full Brier, branch minus chain")
    axs[0].set_yticks(range(len(order)), [n.replace("_", " ") for n in order])
    axs[0].invert_yaxis()
    fig.suptitle("Paired differences across six data banks at 1,200 evaluations")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(folder / "geometry-regression.png", dpi=165)
    plt.close(fig)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    coverage = [r for width in (4, 8) for r in all_orderings(width)]
    pd.DataFrame(coverage).to_csv(args.folder / "template-coverage.csv", index=False)
    registry(args.folder)
    plot(args.folder)

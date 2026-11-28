"""Render the pair-coverage figure from the archived study tables."""

from math import comb
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, PercentFormatter

OUT = Path(__file__).resolve().parent / "results"
INK, MUTED, LIGHT = "#192f3d", "#465c6b", "#dde4e8"
TEAL, DARK, GOLD = "#117476", "#25466b", "#a36029"


def heading(fig, x, y, letter, title, subtitle):
    fig.text(x, y, letter, color=TEAL, weight="bold", fontsize=13)
    fig.text(x + 0.022, y, title, color=INK, weight="bold", fontsize=13)
    fig.text(
        x, y - 0.019, subtitle, color=MUTED, fontsize=9.3, va="top", linespacing=1.35
    )


def style(ax):
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#91a2ab")
    ax.tick_params(axis="both", length=0, pad=5, labelcolor=MUTED)
    ax.xaxis.label.set_color(INK)
    ax.grid(axis="x", color=LIGHT, linewidth=0.7)
    ax.set_axisbelow(True)


def pair_coverage(fig):
    r1 = pd.read_csv(OUT / "r1-scores.csv").query(
        "law == 'min' and method in ('iid','unique')"
    )
    r2 = (
        pd.read_csv(OUT / "r2-coverage.csv")
        .query("law == 'max'")
        .set_index(["method", "n_providers"])
    )
    rows = []
    for method, chance in (("iid", 1 - (14 / 15) ** 8), ("unique", 8 / 15)):
        selected = r1[r1.method == method]
        assert len(selected) == 32 and selected.seed.is_unique
        rows.append(
            (int(selected.true_pair_offered.eq(True).sum()), len(selected), chance)
        )
    for m in (6, 12):
        for method, slots in (("balanced8", 4), ("max8", 8)):
            bank = r2.loc[(method, m)]
            assert bank.banks == 24
            rows.append(
                (int(bank.planted_max_offered), int(bank.banks), slots / comb(m, 2))
            )
    heading(
        fig,
        0.035,
        0.83,
        "A",
        "Offering the planted pair",
        "Each bank receives eight terminal probes. Fractions count banks\n"
        "whose menu offered the two correct inputs with the correct operator.",
    )
    ax = fig.add_axes((0.194, 0.243, 0.238, 0.419))
    style(ax)
    positions = [6.0, 5.1, 3.58, 2.68, 1.16, 0.26]
    labels = [
        "MIN · IID",
        "MIN · unique",
        "4 MIN + 4 MAX",
        "8 MAX",
        "4 MIN + 4 MAX",
        "8 MAX",
    ]
    for y, (hits, n, chance) in zip(positions, rows, strict=True):
        frequency = hits / n
        ax.plot([chance, frequency], [y, y], color="#a5b4bc", lw=1.4, zorder=1)
        # A small vertical offset keeps equal quantities independently visible.
        ax.scatter(
            chance,
            y + 0.09,
            facecolor="white",
            edgecolor=GOLD,
            linewidth=1.3,
            marker="D",
            s=38,
            zorder=3,
        )
        ax.scatter(
            frequency,
            y - 0.09,
            color=TEAL,
            edgecolor="white",
            linewidth=0.5,
            s=48,
            zorder=4,
        )
        ax.text(
            0.785, y, f"{hits}/{n}", va="center", ha="right", color=INK, fontsize=9.2
        )
    ax.set_ylim(-0.15, 6.54)
    ax.set_yticks(positions, labels)
    ax.set_xlim(0, 0.80)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_xlabel("Banks offered the correct pair")
    for y in (4.43, 2.0):
        ax.axhline(y, color=LIGHT, linestyle="--", lw=0.75)
    for y, label in (
        (6.53, "MIN target  ·  6 providers"),
        (4.08, "MAX target  ·  6 providers"),
        (1.68, "MAX target  ·  12 providers"),
    ):
        ax.text(
            -0.665,
            y,
            label,
            transform=ax.get_yaxis_transform(),
            color=DARK,
            size=9.6,
            weight="bold",
            va="center",
        )
    ax.text(0.785, 6.53, "Observed", fontsize=8.3, color=MUTED, ha="right", va="center")
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            color=TEAL,
            label="Observed offer rate",
            markersize=6,
        ),
        Line2D(
            [],
            [],
            marker="D",
            linestyle="",
            markerfacecolor="white",
            markeredgecolor=GOLD,
            label="Design probability",
            markersize=5,
        ),
    ]
    fig.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(0.083, 0.705),
        ncols=2,
        frameon=False,
        handletextpad=0.2,
        columnspacing=1.5,
        fontsize=9,
        labelcolor=INK,
    )
    fig.text(
        0.035,
        0.138,
        "IID draws with replacement. Unique draws distinct pairs for each operator.\n"
        "With a MAX target, eight MIN probes have zero chance of offering the target.",
        ha="left",
        va="top",
        fontsize=9.0,
        color=MUTED,
        linespacing=1.35,
    )


def interval(ax, y, row, live=False):
    color = GOLD if live else DARK
    lo, mid, hi = 1000 * np.array([row.ci_low, row.estimate, row.ci_high])
    assert lo <= mid <= hi
    ax.hlines(y, lo, hi, color=color, lw=1.6)
    ax.vlines([lo, hi], y - 0.09, y + 0.09, color=color, lw=1)
    ax.scatter(
        mid,
        y,
        marker="s" if live else "o",
        s=38,
        color=color,
        edgecolor="white",
        linewidth=0.55,
        zorder=3,
    )


def forest(ax, rows, labels, positions, live_index, xlabel):
    style(ax)
    for j, (row, y) in enumerate(zip(rows, positions, strict=True)):
        interval(ax, y, row, j == live_index)
    ax.axvline(0, linestyle="--", lw=1, color=MUTED, zorder=1)
    ax.set_ylim(min(positions) - 0.55, max(positions) + 0.52)
    ax.set_yticks(positions, labels)
    ax.tick_params(axis="y", labelsize=9.0)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.margins(x=0.085)
    ax.set_xlabel(xlabel)


def plot_risks(fig):
    r1 = (
        pd.read_csv(OUT / "r1-summary.csv")
        .query("law == 'min' and metric == 'audit_excess'")
        .set_index(["method", "reference"])
    )
    r2 = (
        pd.read_csv(OUT / "r2-contrasts.csv", dtype={"n_providers": str})
        .query(
            "law == 'max' and metric == 'audit_excess' and method == 'balanced8' and reference == 'min8'"
        )
        .set_index("n_providers")
    )
    r3 = pd.read_csv(OUT / "r3-contrasts.csv").set_index(["method", "reference"])
    live = (
        pd.read_csv(OUT / "live-contrasts.csv")
        .rename(columns={"difference": "estimate", "low": "ci_low", "high": "ci_high"})
        .set_index("cohort")
    )
    heading(
        fig,
        0.50,
        0.83,
        "B",
        "Error on the controlled targets",
        "Each comparison pairs policies on the same banks. The MIN experiment\n"
        "uses 32 banks. The MAX experiment uses 24, paired across pool sizes.",
    )
    a = fig.add_axes((0.735, 0.516, 0.232, 0.185))
    forest(
        a,
        [
            r1.loc["unique", "iid"],
            r2.loc["6"],
            r2.loc["12"],
            r2.loc["both"],
            live.loc["min"],
        ],
        [
            "MIN · unique − IID",
            "MAX · balanced − MIN, 6 providers",
            "MAX · balanced − MIN, 12 providers",
            "MAX · balanced − MIN, both sizes",
            "MIN · live insertion − ordinary live",
        ],
        [4.0, 3.2, 2.4, 1.6, 0.5],
        4,
        "Change in conditional excess Brier (×10⁻³)",
    )
    heading(
        fig,
        0.50,
        0.388,
        "C",
        "Test loss on the standard benchmark",
        "Four paired banks per family. Each of the 25 standard-v4 families\n"
        "has equal weight.",
    )
    b = fig.add_axes((0.735, 0.155, 0.232, 0.113))
    forest(
        b,
        [r3.loc["unique", "iid"], r3.loc["unique", "gp"], live.loc["panel"]],
        [
            "Terminal unique − terminal IID",
            "Terminal unique − frozen GP",
            "Live insertion − ordinary live",
        ],
        [2, 1, 0],
        2,
        "Change in observed test Brier (×10⁻³)",
    )
    fig.text(
        0.50,
        0.060,
        "Negative favors the first policy. Bars show paired 95% bootstrap intervals.\n"
        "Squares compare two live searches with equal offspring budgets.",
        color=MUTED,
        fontsize=8.9,
        linespacing=1.3,
    )


def main():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.labelsize": 9.35,
            "axes.labelpad": 6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(13.35, 7.8), facecolor="white")
    fig.text(
        0.035,
        0.938,
        "What eight pair proposals change in an ensemble search",
        fontsize=19.4,
        color=INK,
        weight="bold",
    )
    fig.text(
        0.035,
        0.895,
        "Terminal probes compete with a frozen GP finalist. In the live comparison, eight probes replace eight ordinary offspring during evolution.",
        fontsize=10.55,
        color=MUTED,
    )
    pair_coverage(fig)
    plot_risks(fig)
    for fmt in ("png", "pdf"):
        fig.savefig(OUT / f"pair-coverage.{fmt}", dpi=220, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()

"""Render the same-population and transfer studies from their archived summary tables."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

OUT = Path(__file__).resolve().parent / "results"
INK, MUTED, GRID = "#162c39", "#435765", "#d9e1e4"
TEAL, LIGHT_TEAL, ORANGE, BLUE = "#087971", "#98d3c7", "#ad5427", "#3a5d91"
SIZES = [32, 128, 512]


def section(fig, x, y, letter, title, subtitle):
    fig.text(x, y, letter, color=TEAL, weight="bold", fontsize=13, va="top")
    fig.text(x + 0.025, y, title, color=INK, weight="bold", fontsize=13, va="top")
    fig.text(
        x, y - 0.038, subtitle, color=MUTED, fontsize=9.8, va="top", linespacing=1.35
    )


def axes_style(ax, grid_axis):
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#98a7af")
    ax.set_axisbelow(True)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.65)
    ax.tick_params(axis="both", labelcolor=MUTED, length=0, pad=5)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def same_population(fig, parts):
    section(
        fig,
        0.050,
        0.883,
        "A",
        "Learning from a fixed population",
        "One provider has a weak advantage ($d=0.25$).\n"
        "Curves show mean excess Brier over 32 training banks.",
    )
    ax = fig.add_axes((0.090, 0.589, 0.326, 0.180))
    axes_style(ax, "y")
    styles = [
        ("mean", "Arithmetic mean", ORANGE, "D", "--"),
        ("regularized", "Regularized OKAPI", TEAL, "o", "-"),
        ("prob_lr", "Probability stacking", BLUE, "s", "-."),
    ]
    handles = []
    for method, label, color, marker, ls in styles:
        rows = parts.loc[(0.25, SIZES, method), :].sort_index()
        y = rows["excess_brier"].to_numpy()
        (handle,) = ax.plot(
            range(3),
            y,
            color=color,
            ls=ls,
            marker=marker,
            ms=5.4,
            lw=1.8,
            markeredgecolor="white",
            markeredgewidth=0.5,
            label=label,
        )
        handles.append(handle)
    ax.set_xlim(-0.12, 2.12)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.set_xticks(range(3), SIZES)
    ax.set_xlabel("Training observations", labelpad=3)
    ax.set_ylabel("Excess Brier", labelpad=4)
    fig.legend(
        handles=[handles[1], handles[0], handles[2]],
        frameon=False,
        ncols=2,
        handlelength=2.1,
        handletextpad=0.45,
        columnspacing=1.0,
        fontsize=8.9,
        loc="upper left",
        bbox_to_anchor=(0.051, 0.520),
        borderaxespad=0.0,
    )


def decomposition(fig, parts):
    section(
        fig,
        0.510,
        0.883,
        "B",
        "What contributes to OKAPI’s error?",
        "Bars split excess Brier into squared bias and variance across 32 banks.\n"
        "Rows give training sample size. The search seed is fixed.",
    )
    ax = fig.add_axes((0.630, 0.590, 0.316, 0.184))
    axes_style(ax, "x")
    locations = np.array([6, 5, 4, 2.6, 1.6, 0.6])
    labels = ["32", "128", "512"] * 2
    for d, yy in zip((0.25, 0.75), [locations[:3], locations[3:]], strict=True):
        for n, y in zip(SIZES, yy, strict=True):
            reg = parts.loc[(d, n, "regularized")]
            avg = parts.loc[(d, n, "mean")]
            np.testing.assert_allclose(
                reg.bias_squared + reg.variance, reg.excess_brier, atol=2e-13
            )
            ax.barh(y, reg.bias_squared, color=TEAL, height=0.65)
            ax.barh(
                y, reg.variance, left=reg.bias_squared, color=LIGHT_TEAL, height=0.65
            )
            ax.plot(
                avg.excess_brier,
                y,
                ls="",
                marker="D",
                markerfacecolor="white",
                markeredgecolor=ORANGE,
                mew=1.1,
                ms=4.8,
            )
    for y, label in [(5, "Weak\n$d=0.25$"), (1.6, "Strong\n$d=0.75$")]:
        ax.text(
            -0.22,
            y,
            label,
            transform=ax.get_yaxis_transform(),
            clip_on=False,
            va="center",
            ha="center",
            color=INK,
            size=9.4,
            linespacing=1.05,
        )
    ax.set_yticks(locations, labels, fontsize=8.7)
    ax.set_ylim(0, 6.62)
    ax.set_xlim(left=0)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.set_xlabel("Excess Brier", labelpad=3)
    ax.axhline(3.31, color=GRID, lw=0.65)
    handles = [
        Patch(color=TEAL, label="Squared bias"),
        Patch(color=LIGHT_TEAL, label="Variance"),
        Line2D(
            [],
            [],
            marker="D",
            ls="",
            ms=5.2,
            markerfacecolor="white",
            markeredgecolor=ORANGE,
            label="Arithmetic mean’s total",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.510, 0.520),
        ncols=3,
        frameon=False,
        borderaxespad=0,
        fontsize=8.8,
        handlelength=1.2,
        handletextpad=0.4,
        columnspacing=1.0,
    )


def transfer(fig):
    data = (
        pd.read_csv(OUT / "main-contrasts.csv")
        .query("method == 'regularized' and metric == 'target_brier'")
        .set_index("comparison")
    )
    conditions = [
        ("A_rare_source", "Specialist region grows from 5% to 50%"),
        ("B_source_shortcut", "Training shortcut disappears"),
        ("C_train_noise_only", "Provider noise during training only"),
        ("D_target_noise_only", "Provider noise at deployment only"),
        ("E_joint_mismatch", "Region grows, shortcut disappears, deployment noise"),
    ]
    section(
        fig,
        0.050,
        0.425,
        "C",
        "How much does a mismatched training population cost?",
        "Each row compares regularized OKAPI fitted under the stated mismatch with the same method fitted on a matched source.\n"
        "Both fits use 512 observations and are evaluated on the same target. Intervals resample 24 paired training banks.",
    )
    ax = fig.add_axes((0.495, 0.122, 0.300, 0.190))
    axes_style(ax, "x")
    ax.spines["bottom"].set_visible(False)
    ax.axvline(0, color="#7e8f99", lw=1)
    ax.set_yticks([])
    ax.set_ylim(-0.45, 4.45)
    ax.set_xlim(-0.009, 0.33)
    ax.set_xticks([0, 0.1, 0.2, 0.3], ["0", "+0.10", "+0.20", "+0.30"])
    ax.set_xlabel("Increase in target Brier", labelpad=5)
    for y, (key, label) in zip(range(4, -1, -1), conditions, strict=True):
        row = data.loc[key]
        assert row.banks == 24 and row.ci_low <= row.estimate <= row.ci_high
        ax.plot([row.ci_low, row.ci_high], [y, y], c=TEAL, lw=2)
        ax.scatter(row.estimate, y, s=33, color=INK, zorder=3)
        y_fig = fig.transFigure.inverted().transform(ax.transData.transform((0, y)))[1]
        fig.text(0.052, y_fig, label, color=INK, fontsize=10.0, va="center")
        fig.text(
            0.816,
            y_fig,
            f"{row.estimate:+.3f}  [{row.ci_low:.3f}, {row.ci_high:.3f}]",
            color=INK,
            fontsize=9.6,
            va="center",
        )
    fig.text(0.816, 0.321, "Mean [95% interval]", color=MUTED, size=9, weight="bold")


def main():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "pdf.fonttype": 42,
            "axes.unicode_minus": True,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    parts = (
        pd.read_csv(OUT / "stratified-decomposition.csv")
        .set_index(["specialization", "size", "method"])
        .sort_index()
    )
    assert parts.index.is_unique and len(parts) == 54
    fig = plt.figure(figsize=(11.8, 8))
    fig.text(
        0.05,
        0.962,
        "Training-sample size and transfer in probability fusion",
        fontsize=21,
        color=INK,
        weight="bold",
        va="top",
    )
    same_population(fig, parts)
    decomposition(fig, parts)
    fig.add_artist(
        Line2D(
            (0.05, 0.951),
            (0.455, 0.455),
            color=GRID,
            linewidth=0.9,
            transform=fig.transFigure,
        )
    )
    transfer(fig)
    fig.text(
        0.05,
        0.039,
        "Population risks integrate the full two-class Brier score over 512 possible inputs in A–B and 16 inputs in C.",
        fontsize=9.1,
        color=MUTED,
        va="center",
    )
    fig.savefig(OUT / "transfer-factorial.png", dpi=220)
    fig.savefig(
        OUT / "transfer-factorial.pdf",
        metadata={
            "Title": "Training-sample size and transfer in probability fusion",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(fig)


if __name__ == "__main__":
    main()

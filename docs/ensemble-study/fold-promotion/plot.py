"""Draw the two rounds' paired audit-risk differences from their published tables."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

HERE = Path(__file__).resolve().parent
BLUE, ORANGE, INK = "#176286", "#a04b23", "#243344"
POLICIES = ("pooled_fold", "fold_pooled", "fold_fold")


def interval(ax, row, height, color, symbol):
    point, lo, hi = 1_000 * row[["estimate", "ci_low", "ci_high"]].to_numpy(float)
    if not np.isfinite([point, lo, hi]).all() or lo > hi:
        raise ValueError("Invalid interval")
    ax.hlines(height, lo, hi, color=color, linewidth=1.9)
    ax.vlines([lo, hi], height - 0.055, height + 0.055, color=color, linewidth=1.2)
    ax.plot(
        point,
        height,
        marker=symbol,
        markersize=7.5,
        markeredgecolor=color,
        markeredgewidth=1.65,
        markerfacecolor="white" if lo <= 0 <= hi else color,
    )


def draw(r1, r2):
    a = r1.query(
        "law == 'signal' and method == 'seeded' and metric == 'audit_excess'"
        " and contrast == '24x8_minus_192x1'"
    ).set_index("scene_error_share", verify_integrity=True)
    b = r2.query(
        "scene_error_share == 1 and n_scenes == 24 and rows_per_scene == 8"
        " and metric == 'audit_excess' and reference == 'pooled_pooled'"
        " and method in @POLICIES"
    ).set_index(["method", "law"], verify_integrity=True)
    if set(a.index) != {0, 0.5, 1} or set(b.index) != {
        (p, law) for p in POLICIES for law in ("signal", "equal")
    }:
        raise ValueError("Expected all three scene shares and all six policy contrasts")

    plt.rcParams.update(
        {
            "font.size": 11,
            "pdf.fonttype": 42,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
        }
    )
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.4, 5.3))
    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.29, top=0.68, wspace=0.47)
    fig.text(
        0.035,
        0.94,
        "Scene clustering and fold-based tree selection",
        fontsize=19,
        weight="bold",
        ha="left",
    )
    fig.text(
        0.035,
        0.875,
        "Layouts give scenes × rows per scene. The fold rule scores the gain over the anchor "
        "reached in at least three of four fit folds.",
        fontsize=10,
    )
    for ax, title, subtitle in (
        (
            left,
            "A    Concentrating 192 fit rows",
            "One provider reports the true class probability.",
        ),
        (
            right,
            "B    Changing the promotion rule",
            "24 scenes × 8 rows, with provider noise shared within scenes.",
        ),
    ):
        ax.set_title(title, loc="left", fontsize=13, weight="bold", pad=33)
        ax.text(
            0, 1.13, subtitle, transform=ax.transAxes, fontsize=9.4, color="#465969"
        )
        ax.axvline(0, color="#667a89", linewidth=1.15, linestyle="--", zorder=0)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#a7b2bc")
        ax.tick_params(axis="y", length=0, pad=8)
        ax.grid(axis="x", color="#e4e9ed", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_ylim(-0.53, 2.53)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.margins(x=0.12)

    for y, share in zip((2, 1, 0), (0, 0.5, 1), strict=True):
        interval(left, a.loc[share], y, BLUE, "o")
    left.set_yticks((2, 1, 0), ("0%", "50%", "100%"))
    left.set_ylabel("Scene-shared residual variance")
    left.set_xlabel("24 × 8 minus 192 × 1   (×10⁻³ Brier)")

    for y, policy in zip((2, 1, 0), POLICIES, strict=True):
        interval(right, b.loc[policy, "equal"], y + 0.13, ORANGE, "s")
        interval(right, b.loc[policy, "signal"], y - 0.13, BLUE, "o")
    right.set_yticks((2, 1, 0), ("Final choice only", "Search only", "Both"))
    right.set_xlabel("Changed rule minus pooled rule   (×10⁻³ Brier)")
    handles = [
        Line2D([], [], color=color, marker=marker, linestyle="", label=label)
        for color, marker, label in (
            (ORANGE, "s", "Equal competence"),
            (BLUE, "o", "Exact posterior provider"),
        )
    ]
    fig.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.56, 0.12),
        ncols=2,
        frameon=False,
        fontsize=10,
        handletextpad=0.15,
        columnspacing=1.1,
    )
    fig.text(
        0.035,
        0.082,
        "Points show differences in conditional excess Brier on fresh scenes. "
        "Negative values favor 24 × 8 in A and the changed rule in B.",
        fontsize=9.3,
    )
    fig.text(
        0.035,
        0.043,
        "Bars show 95% paired bootstrap intervals across 32 banks in each round. "
        "An open marker means the interval includes zero.",
        fontsize=9.3,
    )
    return fig


if __name__ == "__main__":
    results = HERE / "results"
    r1 = pd.read_csv(HERE.parent / "scene-dependence/results/paired-contrasts.csv")
    figure = draw(r1, pd.read_csv(results / "contrasts.csv"))
    for fmt in ("png", "pdf"):
        figure.savefig(
            results / f"transfer-decisions.{fmt}", dpi=220, facecolor="white"
        )
    plt.close(figure)

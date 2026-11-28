"""Draw the three-round figure directly from archived scores, contrasts and witnesses."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

OUT = Path(__file__).resolve().parent / "results"
INK = "#192b38"
MUTED = "#43596a"
GRID = "#dde4e8"
ZERO = "#858e97"
LOWER = "#126e70"
HIGHER = "#ab4b29"
SMALL = "#284969"
LARGE = "#96703a"


def clean_axis(ax, grid):
    ax.set_axisbelow(True)
    ax.grid(axis=grid, linewidth=0.6, color=GRID)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(ZERO)
    ax.tick_params(axis="both", colors=MUTED, length=0, pad=4)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def heading(fig, x, y, label, title, subtitle):
    fig.text(x, y, label, weight="bold", color=LOWER, size=13)
    fig.text(x + 0.025, y, title, weight="bold", color=INK, size=13)
    fig.text(x, y - 0.018, subtitle, size=9.3, va="top", color=MUTED, linespacing=1.27)


def interval_text(row, scale):
    def number(v):
        return f"{v * scale:+.2f}".replace("-", "−")

    return f"Mean {number(row.estimate)}   ·   95% paired bootstrap interval [{number(row.ci_low)}, {number(row.ci_high)}]"


def final_rule_panel(fig, out):
    scores = pd.read_csv(out / "r1-scores.csv", float_precision="round_trip")
    risks = scores.query("law == 'signal' and n_scenes == 24").pivot(
        index="seed", columns="method", values="audit_excess"
    )
    deltas = (risks.fold_joint - risks.fold_fold).sort_values().to_numpy()
    if len(deltas) != 48 or not np.isfinite(deltas).all():
        raise ValueError("Expected the 48 paired primary checkpoint banks")
    tab = pd.read_csv(out / "r1-contrasts.csv").query(
        "law == 'signal' and n_scenes == 24 and metric == 'audit_excess' "
        "and method == 'fold_joint' and reference == 'fold_fold'"
    )
    assert len(tab) == 1 and np.isclose(tab.iloc[0].estimate, deltas.mean())
    heading(
        fig,
        0.035,
        0.838,
        "A",
        "Changing the final choice",
        "One exact provider, 24 fit scenes × 8 rows. Each point compares\n"
        "joint and fold-only choices from the same fold-trained population.",
    )
    ax = fig.add_axes((0.099, 0.549, 0.366, 0.181))
    clean_axis(ax, "y")
    x = np.arange(1, 49)
    y = 1000 * deltas
    colors = np.where(deltas < -1e-12, LOWER, np.where(deltas > 1e-12, HIGHER, ZERO))
    ax.axhline(0, color=ZERO, lw=0.85)
    ax.vlines(x, 0, y, colors=colors, linewidth=1)
    ax.scatter(x, y, c=colors, edgecolor="white", linewidth=0.3, s=14, zorder=3)
    ax.set_xlim(0.3, 48.7)
    ax.set_xticks([1, 12, 24, 36, 48])
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.set_ylabel("Δ excess Brier (×10⁻³)")
    ax.set_xlabel("Banks ordered by paired change")
    fig.text(0.099, 0.749, interval_text(tab.iloc[0], 1000), color=INK, size=9.1)
    fig.text(
        0.099, 0.485, "Negative favors the joint final rule.", color=MUTED, size=8.8
    )


def discovery_panel(fig, out):
    diagnostics = pd.read_csv(out / "r2-diagnostics.csv").query("law == 'min'")
    scores = pd.read_csv(out / "r2-scores.csv").query(
        "law == 'min' and method == 'fold_joint'"
    )
    witness = pd.read_csv(out / "r2-witness.csv").query("law == 'min'")
    for frame in (diagnostics, scores, witness):
        assert len(frame) == 64 and not frame.duplicated(["seed", "n_rows"]).any()
    summary = pd.concat(
        (
            diagnostics.groupby("n_rows")[
                ["exact_ever_visited", "exact_in_final"]
            ].sum(),
            scores.groupby("n_rows").selected_is_exact.sum(),
            witness.groupby("n_rows").accept_joint.sum(),
        ),
        axis=1,
    )
    assert (summary.exact_in_final <= summary.exact_ever_visited).all()
    assert (summary.selected_is_exact <= summary.exact_in_final).all()
    heading(
        fig,
        0.035,
        0.421,
        "B",
        "Finding and choosing the exact MIN",
        "The target takes the lower of two anonymous scores. Each stage\n"
        "counts banks with a prediction matching the target.",
    )
    ax = fig.add_axes((0.203, 0.129, 0.255, 0.204))
    clean_axis(ax, "x")
    labels = [
        "Evaluated by search",
        "Present at the end",
        "Chosen by joint rule",
        "Accepted when supplied",
    ]
    positions = np.array([3.15, 2.24, 1.33, 0.0])
    ax.set_yticks(positions, labels=labels)
    ax.tick_params(axis="y", pad=7)
    ax.axhline(0.67, color=GRID, lw=1.0, linestyle="--")
    for n, offset, color, marker in (
        (192, 0.17, SMALL, "o"),
        (1536, -0.17, LARGE, "s"),
    ):
        counts = summary.loc[n].to_numpy(dtype=int)
        ys = positions + offset
        ax.hlines(ys, 0, counts, color=color, linewidth=1.35, alpha=0.72)
        ax.scatter(counts, ys, color=color, s=20, marker=marker, zorder=3)
        for val, y in zip(counts, ys, strict=True):
            ax.text(
                val + 0.9,
                y,
                str(val),
                size=8.5,
                color=color,
                va="center",
                weight="bold",
            )
    ax.set_xlim(-0.7, 35.8)
    ax.set_ylim(-0.55, 3.7)
    ax.set_xticks([0, 8, 16, 24, 32])
    ax.set_xlabel("Banks (out of 32)")
    handles = [
        Line2D([], [], color=c, marker=m, markersize=5, linewidth=1.2, label=l)
        for c, m, l in ((SMALL, "o", "192 fit rows"), (LARGE, "s", "1,536 fit rows"))
    ]
    fig.legend(
        handles=handles,
        bbox_to_anchor=(0.199, 0.356),
        loc="upper left",
        ncols=2,
        frameon=False,
        borderaxespad=0,
        columnspacing=1.3,
        handlelength=1.25,
        handletextpad=0.45,
        fontsize=8.7,
        labelcolor=INK,
    )
    fig.text(
        0.035,
        0.049,
        "The last row offers the known MIN tree directly to the selector in a separate test.",
        color=MUTED,
        size=8.4,
    )


def training_panel(fig, out):
    scores = pd.read_csv(out / "r3-panel-scores.csv", float_precision="round_trip")
    pairs = scores.pivot(
        index=["scenario_id", "data_seed"], columns="method", values="test_brier"
    )
    differences = (pairs.joint_joint - pairs.fold_joint).unstack("data_seed")
    avg = differences.mean(axis=1).sort_values(kind="stable")
    if differences.shape != (25, 4) or differences.isna().any().any():
        raise ValueError("Expected four paired held-out banks in each of 25 scenarios")
    contrast = pd.read_csv(out / "r3-panel-contrasts.csv").query(
        "method == 'joint_joint' and reference == 'fold_joint'"
    )
    assert len(contrast) == 1 and np.isclose(avg.mean(), contrast.iloc[0].estimate)
    heading(
        fig,
        0.536,
        0.838,
        "C",
        "Changing the search criterion",
        "Joint-trained versus fold-trained search, with joint final choice in both.\n"
        "Large dots show scenario means, small marks show four paired banks.",
    )
    fig.text(0.536, 0.749, interval_text(contrast.iloc[0], 1000), color=INK, size=9.1)
    ax = fig.add_axes((0.700, 0.130, 0.264, 0.591))
    clean_axis(ax, "x")
    y = np.arange(len(avg))
    for i, (name, d) in enumerate(avg.items()):
        color = LOWER if d < -1e-12 else HIGHER if d > 1e-12 else ZERO
        seeds = differences.loc[name].to_numpy() * 1000
        ax.hlines(i, min(seeds), max(seeds), color="#b5c0c8", linewidth=1, zorder=1)
        ax.scatter(
            seeds,
            np.full(4, i),
            color="#8998a3",
            s=14,
            linewidth=0.75,
            marker="|",
            zorder=2,
        )
        ax.scatter(
            d * 1000, i, color=color, edgecolor="white", linewidth=0.5, s=33, zorder=3
        )
    ax.axvline(0, color=ZERO, linestyle="--", linewidth=1.0, zorder=0)
    ax.set_yticks(y, labels=[k.replace("_", " ") for k in avg.index])
    ax.tick_params(axis="y", labelsize=8.35, pad=7)
    ax.set_ylim(len(y) - 0.4, -0.65)
    # A narrow linear center shows small regressions while keeping large bank
    # effects visible in the same panel. Explicit ticks show the change of scale.
    ax.set_xscale("symlog", linthresh=3, linscale=0.85)
    ticks = [-300, -30, -3, 0, 3, 30, 300]
    ax.set_xticks(ticks, labels=[str(t).replace("-", "−") for t in ticks])
    limit = max(525, 1050 * differences.abs().to_numpy().max())
    ax.set_xlim(-limit, limit)
    ax.minorticks_off()
    ax.set_xlabel("Δ test Brier (×10⁻³)")
    fig.text(
        0.536,
        0.049,
        "Negative favors joint-trained search. The axis is linear within ±3 and logarithmic beyond.",
        color=MUTED,
        size=8.4,
    )


def main():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.labelsize": 9.4,
            "axes.labelpad": 5.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(13.0, 8.6), facecolor="white")
    fig.text(
        0.035,
        0.938,
        "Requiring pooled improvement and gains in three of four folds",
        fontsize=19.2,
        weight="bold",
        color=INK,
    )
    fig.text(
        0.035,
        0.900,
        "Three experiments on synthetic provider predictions, with separate banks and held-out evaluation in every round.",
        color=MUTED,
        fontsize=10.6,
    )
    final_rule_panel(fig, OUT)
    discovery_panel(fig, OUT)
    training_panel(fig, OUT)
    for fmt in ("png", "pdf"):
        fig.savefig(OUT / f"joint-evidence.{fmt}", dpi=220, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()

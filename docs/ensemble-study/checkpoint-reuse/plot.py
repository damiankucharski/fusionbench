"""Plot the registered six-method comparison from analyze.py's paired summaries."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator, ScalarFormatter

METHOD_LABELS = {
    "gp_brier": "Native GP · Brier fitness",
    "gp_f1": "Native GP · F1 fitness",
    "regularized": "Regularized native OKAPI",
    "mean": "Arithmetic mean",
    "top_n": "Ranked-prefix Top-N",
    "prob_lr": "Probability stacking",
}
PANELS = (
    ("audit_brier", "Full-class Brier", "Positive = higher loss with reuse", 1.0),
    ("audit_f1", "Positive-class F1", "Positive = higher F1 with reuse", 100.0),
)


def plot(summary, output):
    selected = summary.query("k == 8 and comparison == 'reuse_minus_disjoint'")
    figure, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    figure.subplots_adjust(left=0.245, right=0.965, top=0.71, bottom=0.20, wspace=0.20)
    figure.text(
        0.035,
        0.94,
        "Effect of reusing scenes on fresh-scene performance",
        fontsize=17,
        weight="semibold",
        color="#192634",
    )
    figure.text(
        0.035,
        0.875,
        "Same versus separate scenes for checkpoint selection and fusion fitting  ·  Eight checkpoints per provider",
        fontsize=10.5,
        color="#425262",
    )
    y = np.arange(len(METHOD_LABELS))
    for ax, (metric, title, direction, multiplier) in zip(axes, PANELS, strict=True):
        panel = selected[selected.metric.eq(metric)].set_index("method")
        if not panel.index.is_unique or set(panel.index) != set(METHOD_LABELS):
            raise ValueError(f"Expected one {metric} contrast per registered method")
        panel = panel.loc[list(METHOD_LABELS)]
        if not panel.n_banks.eq(32).all():
            raise ValueError("Figure expects the completed 32-bank comparison")
        points, lo, hi = (
            panel[col].to_numpy(dtype=float) * multiplier
            for col in ("estimate", "ci_low", "ci_high")
        )
        if not np.isfinite([points, lo, hi]).all() or (lo > hi).any():
            raise ValueError("Invalid paired estimates or intervals")

        ax.set_title(title, loc="left", fontsize=13, fontweight="semibold", pad=24)
        ax.text(
            0, 1.04, direction, transform=ax.transAxes, fontsize=9.5, color="#425262"
        )
        ax.set_axisbelow(True)
        ax.grid(axis="x", linewidth=0.65, color="#e0e5e9")
        ax.axvline(0, color="#657481", linewidth=1.1)
        # Separate artists also display degenerate intervals, such as the mean's exact zero.
        ax.hlines(y, lo, hi, color="#215c68", linewidth=1.7)
        ax.plot(lo, y, "|", markersize=8, color="#215c68")
        ax.plot(hi, y, "|", markersize=8, color="#215c68")
        ax.scatter(
            points, y, s=40, color="#124a58", edgecolor="white", linewidth=0.7, zorder=3
        )

        span = max(np.max(np.abs([points, lo, hi])), 1e-12) * 1.11
        ax.set_xlim(-span, span)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, symmetric=True))
        formatter = ScalarFormatter(useOffset=False)
        formatter.set_scientific(False)
        ax.xaxis.set_major_formatter(formatter)
        ax.set_xlabel(
            "Brier difference"
            if multiplier == 1
            else "F1 difference (percentage points)",
            labelpad=9,
        )
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#afbac1")
        ax.tick_params(axis="x", colors="#425262", labelsize=9)
        ax.tick_params(axis="y", length=0, pad=11, colors="#22303d")
    axes[0].set_yticks(y, list(METHOD_LABELS.values()), fontsize=10.5)
    axes[0].set_ylim(len(y) - 0.48, -0.55)
    figure.text(
        0.035,
        0.045,
        "Points show reuse minus independent fitting. Bars show paired 95% bootstrap intervals across 32 independent banks.",
        fontsize=9.5,
        color="#425262",
    )
    figure.savefig(output, dpi=180, facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", type=Path, default=Path(__file__).resolve().parent / "results"
    )
    args = parser.parse_args()
    plot(pd.read_csv(args.results / "summary.csv"), args.results / "transfer.png")

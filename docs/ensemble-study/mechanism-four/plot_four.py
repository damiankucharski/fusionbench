"""Render one research figure from the scored, frozen round-four comparison."""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from stats_four import CELLS, CODES, OUT

GREEN = "#176758"
BLUE = "#285990"
ORANGE = "#ad552d"
GRAY = "#536070"
LIGHT = "#dfe4e8"
INCUMBENT = "pull_mean"


def style(ax):
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(LIGHT)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=LIGHT, lw=0.6)
    ax.set_axisbelow(True)


def draw(policy, out=OUT):
    levels = pd.read_csv(out / "r1-levels.csv")
    contrasts = pd.read_csv(out / "r4-contrasts.csv")
    families = pd.read_csv(out / "r4-families.csv")
    fname = {
        "cal_pull": "calibrated incumbent",
        "sparse_gp": "sparse-seeded GP",
        "cal_sparse": "calibrated sparse-seeded GP",
    }[policy]
    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(12.8, 10.4))
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=(1.15, 1),
        left=0.12,
        right=0.96,
        top=0.87,
        bottom=0.095,
        hspace=0.53,
        wspace=0.54,
    )
    fig.text(
        0.035,
        0.975,
        "Rare-event fusion across matched synthetic worlds",
        fontsize=19,
        weight="bold",
        va="top",
    )
    fig.text(
        0.035,
        0.938,
        f"Rounds 1–4 on simulated data  •  Confirmation candidate: {fname}",
        fontsize=11,
        color=GRAY,
        va="top",
    )

    # Absolute risks on identical latent inputs, labels and residual draws across switches.
    ax = fig.add_subplot(grid[0, 0])
    labels = [
        " + ".join(a for a, on in zip("HCS", code, strict=True) if on) or "All off"
        for code in CODES
    ]
    left = (
        levels[levels.group.isin(CELLS)]
        .pivot(index="group", columns="method", values="expected_brier")
        .loc[CELLS]
    )
    positions = np.arange(8)
    for m, name, color, marker in (
        ("mean", "Arithmetic mean", GREEN, "o"),
        ("lr", "Logistic stacking", BLUE, "^"),
        (INCUMBENT, "Full-data incumbent", ORANGE, "s"),
    ):
        ax.scatter(
            left[m],
            positions,
            label=name,
            color=color,
            marker=marker,
            edgecolors="white",
            linewidths=0.4,
            s=42,
            zorder=3,
        )
    ax.set(
        yticks=positions,
        yticklabels=labels,
        xlabel="Conditional expected Brier  (lower is better)",
    )
    ax.invert_yaxis()
    ax.set_title(
        "A   Which mechanism changes the outcome?",
        loc="left",
        weight="bold",
        fontsize=12,
        pad=14,
    )
    ax.legend(
        loc="upper left", bbox_to_anchor=(0, -0.19), frameon=False, fontsize=8.5, ncol=2
    )
    style(ax)

    ax = fig.add_subplot(grid[1, 0])
    ax.set_title(
        "B   Confirmation versus the full-data incumbent",
        loc="left",
        weight="bold",
        fontsize=12,
        pad=14,
    )
    labels = (
        ("factorial", "expected_brier", "Factorial average*"),
        ("scene16", "brier", "16 training scenes"),
        ("scene128", "brier", "128 training scenes"),
        ("weak", "brier", "Weak specialization"),
        ("strong", "brier", "Strong specialization"),
        ("standard", "brier", "All 30 families"),
    )
    pairs = contrasts[(contrasts.method == policy) & (contrasts.reference == INCUMBENT)]
    for idx, (group, metric, _) in enumerate(labels):
        d = pairs[(pairs.group == group) & (pairs.metric == metric)].iloc[0]
        # Percentile intervals may exclude the point estimate; draw endpoints directly.
        ax.hlines(idx, d.lo, d.hi, lw=2, color=BLUE)
        ax.scatter(d.delta, idx, s=35, color=BLUE, zorder=3)
    ax.axvline(0, color=GRAY, lw=1)
    ax.set(
        yticks=np.arange(len(labels)),
        yticklabels=[v[2] for v in labels],
        xlabel="Brier difference  (negative favors candidate)",
        ylim=(len(labels) - 0.5, -0.5),
    )
    style(ax)

    ax = fig.add_subplot(grid[:, 1])
    ax.set_title(
        "C   Change in each standard family",
        loc="left",
        weight="bold",
        fontsize=12,
        pad=14,
    )
    rows = families[families.method == policy].sort_values("delta_pull_mean")
    assert len(rows) == 30
    xs = rows.delta_pull_mean.to_numpy()
    ys = np.arange(len(rows))
    colors = [GREEN if d < -1e-12 else ORANGE if d > 1e-12 else GRAY for d in xs]
    ax.axvline(0, color=GRAY, lw=1)
    ax.hlines(ys, 0, xs, color=colors, linewidth=1.4)
    ax.scatter(xs, ys, color=colors, s=26, zorder=3)
    ax.set(
        yticks=ys,
        yticklabels=rows.family,
        ylim=(len(ys) - 0.3, -0.7),
        xlabel="Brier difference vs incumbent",
    )
    ax.tick_params(axis="y", labelsize=9)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    style(ax)

    fig.text(
        0.035,
        0.038,
        "H = heterogeneous provider noise, C = cross-provider common error, S = local specialist. "
        "*Factorial uses conditional expected Brier.",
        fontsize=9,
        color=GRAY,
    )
    fig.text(
        0.035,
        0.019,
        "Intervals in B resample paired data seeds. Family means in C describe four draws each. "
        "Programs use the fixed research search budget.",
        fontsize=9,
        color=GRAY,
    )
    fig.savefig(out / "mechanism-four.pdf", bbox_inches="tight")
    fig.savefig(out / "mechanism-four.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "policy", nargs="?", choices=("cal_pull", "cal_sparse", "sparse_gp")
    )
    supplied = parser.parse_args().policy
    from choose_four import locked

    selected = locked()["chosen"]
    if supplied is not None and supplied != selected:
        parser.error(f"Expected the registered confirmation policy {selected}")
    draw(selected)

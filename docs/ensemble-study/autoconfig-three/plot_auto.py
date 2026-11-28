"""Plot fresh-bank changes relative to the fixed action trained in round one."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator, PercentFormatter

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
DARK, GREEN, RED, GREY = "#162d3c", "#137b75", "#ab402d", "#677582"
GROUPS = {
    "standard": "Standard-v6 (30)",
    "trained24": "Training families (24)",
    "heldout6": "Held-out families (6)",
    "factorial": "Eight variants",
    "scene16": "16 training scenes",
    "scene128": "128 training scenes",
}
ACTIONS = {
    "mean": ("Mean", "#648f95"),
    "top_n": ("Top-N", "#a7b1ad"),
    "cal_mean": ("Calibrated mean", "#dbaf6c"),
    "flat": ("Flat logit", "#6578a4"),
    "cal_pull": ("Calibrated GP", "#22796c"),
}


def draw():
    confirm = json.loads((OUT / "confirmation.json").read_text())
    chosen = confirm["chosen"]
    families = pd.read_csv(OUT / "r3-families.csv")
    diff = pd.read_csv(OUT / "r3-contrasts.csv")
    counts = pd.read_csv(OUT / "r3-actions.csv")
    choices = pd.read_csv(OUT / "r3-scores.csv", usecols=["method", "action"])
    (comparator,) = choices.loc[choices.method == "constant", "action"].unique()
    label = {"tree_rule": "Fingerprint rule", "cv_rule": "Cross-validated rule"}[chosen]
    other = "cv_rule" if chosen == "tree_rule" else "tree_rule"
    other_label = "cross-validated" if other == "cv_rule" else "fingerprint"
    fig = plt.figure(figsize=(13.4, 10.4))
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.04, 1],
        height_ratios=[1.03, 0.97],
        wspace=0.44,
        hspace=0.44,
        left=0.204,
        right=0.97,
        top=0.85,
        bottom=0.13,
    )
    ax = fig.add_subplot(gs[:, 0])
    family = (
        families.pivot(index="family", columns="method", values="delta_constant")
        .sort_values([chosen, other])
        .reset_index()
    )
    from summarize_auto import HELDOUT

    vals = family[chosen].to_numpy()
    face = [GREEN if v < -1e-12 else RED if v > 1e-12 else GREY for v in vals]
    ix = np.arange(len(family))
    ax.barh(ix, vals, color=face, height=0.61)
    ax.scatter(
        np.zeros(np.sum(np.abs(vals) <= 1e-12)),
        ix[np.abs(vals) <= 1e-12],
        s=9,
        color=GREY,
        zorder=4,
    )
    ax.scatter(
        family[other],
        ix,
        s=20,
        facecolors="none",
        edgecolors=DARK,
        linewidths=0.8,
        zorder=5,
    )
    ax.set_yticks(
        ix, [n + (" *" if n in HELDOUT else "") for n in family.family], fontsize=8.5
    )
    ax.invert_yaxis()
    ax.set_xlabel("Change in held-out Brier")
    ax.axvline(0, lw=0.9, color=DARK)
    ax.set_title("Changes by scenario", loc="left", size=14, pad=16)
    ax = fig.add_subplot(gs[0, 1])
    c = diff[(diff.reference == "constant") & (diff.metric == "brier")].set_index(
        ["group", "method"]
    )
    for i, g in enumerate(GROUPS):
        for rule, y in ((chosen, i - 0.12), (other, i + 0.12)):
            lo, mean, hi = c.loc[(g, rule), ["lo", "delta", "hi"]]
            col = (
                (GREEN if mean < -1e-12 else RED if mean > 1e-12 else GREY)
                if rule == chosen
                else DARK
            )
            ax.hlines(y, lo, hi, color=col, lw=1.5)
            ax.scatter(
                mean,
                y,
                edgecolors=col,
                facecolors=col if rule == chosen else "white",
                s=24,
                zorder=3,
            )
    ax.axvline(0, lw=0.9, color=DARK)
    ax.set_yticks(np.arange(len(GROUPS)), list(GROUPS.values()), fontsize=9)
    ax.set_ylim(len(GROUPS) - 0.4, -0.6)
    ax.set_title("Paired changes and 95% intervals", loc="left", size=14, pad=16)
    ax.set_xlabel("Change in held-out Brier")
    ax = fig.add_subplot(gs[1, 1])
    frequencies = counts[counts.method == "cv_rule"].pivot(
        index="group", columns="action", values="fraction"
    )
    yy = np.arange(len(GROUPS))
    start = np.zeros(len(yy))
    for action, (name, col) in ACTIONS.items():
        vals = frequencies.loc[list(GROUPS), action].to_numpy()
        ax.barh(yy, vals, left=start, height=0.56, color=col, label=name)
        start += vals
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.set_yticks(yy, list(GROUPS.values()), fontsize=9)
    ax.invert_yaxis()
    ax.set_title("Cross-validated configuration choices", loc="left", size=14, pad=16)
    ax.legend(
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.45, -0.27),
        frameon=False,
        fontsize=8,
        columnspacing=0.9,
        handlelength=1.3,
    )
    for axis in fig.axes:
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.grid(axis="x", alpha=0.18)
        axis.set_axisbelow(True)
        axis.tick_params(axis="y", length=0)
    fig.text(
        0.045,
        0.953,
        "Does the source pool identify a better configuration?",
        size=21,
        weight="bold",
        color=DARK,
    )
    fig.text(
        0.045,
        0.911,
        f"Filled marks show the selected {label.lower()} versus fixed {ACTIONS[comparator][0].lower()}. Open circles show the {other_label} alternative. Lower Brier is better.",
        size=12,
        color=DARK,
    )
    fig.text(
        0.045,
        0.043,
        "* Family excluded from fitting the fingerprint rule, choosing the constant and selecting the round-three rule.\n"
        "Intervals resample test-bank seeds within fixed families. The eight variants share each source and evaluation draw.",
        size=9,
        color=GREY,
    )
    for fmt in ("png", "pdf"):
        fig.savefig(OUT / f"autoconfig-three.{fmt}", dpi=145, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    draw()

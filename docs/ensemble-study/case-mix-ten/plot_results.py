"""Plot the registered R10 confirmation and the earlier stage-isolation experiments."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
BLUE = "#06635f"
RED = "#ae4f2b"
PURPLE = "#574783"

FAMILIES = {
    "mixed": "Mixed",
    "reversed": "Reversed relation",
    "extreme": "Extreme prevalence",
    "small128": "Mixed, 128 rows",
    "noisy_context": "Noisy context",
    "case_mix": "Case mix only",
    "varied": "Activity only",
    "homogeneous": "Homogeneous",
}


def table(round_no, name):
    return pd.read_csv(RESULTS / f"r{round_no:02d}-{name}.csv")


def draw_points(ax, frame, ys, color, marker):
    delta = frame.delta.to_numpy() * 1000
    bounds = frame[["lo", "hi"]].to_numpy().T * 1000
    ax.hlines(ys, *bounds, linewidth=1.5, color=color, zorder=2)
    ax.plot(delta, ys, marker=marker, ls="none", ms=5.5, color=color, zorder=3)


def labels(ax, names, xlabel):
    ax.set_yticks(np.arange(len(names)), names)
    ax.set_ylim(len(names) - 0.5, -0.5)
    ax.axvline(0, lw=1, color="#65676b", zorder=1)
    ax.grid(axis="x", color="#e3e3e3", lw=0.75)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0, pad=6)
    ax.spines[["left", "right", "top"]].set_visible(False)
    ax.spines["bottom"].set_color("#c6c6c6")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.set_xlabel(xlabel, labelpad=7)


def focal(ax, contrasts, support, pop):
    ys = np.arange(len(FAMILIES))
    fit = contrasts[(contrasts.world == "max_regret") & (contrasts.method == "full")]
    fit = fit.set_index("family").loc[list(FAMILIES)]
    counts = (
        support[
            (support.endpoint == "max_regret")
            & (support.method == "table_full")
            & (support.reference == "table_native")
        ]
        .set_index("family")
        .loc[list(FAMILIES)]
    )
    theory = pop[pop.policy == "full"].set_index("profile").loc[list(FAMILIES)]
    if not (fit.banks.eq(20).all() and counts.source_banks.eq(20).all()):
        raise ValueError("Confirmation expects 20 source draws in each focal profile")
    independent = (
        support[
            (support.endpoint == "max_regret")
            & (support.method == "full")
            & (support.reference == "native")
        ]
        .set_index("family")
        .loc[list(FAMILIES)]
    )
    np.testing.assert_allclose(
        fit[["delta", "lo", "hi"]],
        independent[["delta", "lo", "hi"]],
        rtol=0,
        atol=1e-12,
    )
    draw_points(ax, fit, ys - 0.20, BLUE, "o")
    draw_points(ax, counts, ys, RED, "s")
    ax.scatter(
        theory.delta_max_regret_vs_unweighted * 1000,
        ys + 0.20,
        s=30,
        ec=PURPLE,
        fc="white",
        marker="D",
        lw=1.5,
        zorder=3,
    )
    labels(ax, list(FAMILIES.values()), "Change in worst endpoint regret (×1,000)")
    ax.set_title("A   Full weighting across eight source profiles", loc="left", pad=13)
    handles = [
        Line2D(
            [],
            [],
            marker=marker,
            ls="none",
            color=color,
            markerfacecolor=face,
            markersize=6,
            label=label,
        )
        for marker, color, face, label in (
            ("o", BLUE, BLUE, "Native OKAPI"),
            ("s", RED, RED, "16-pattern table"),
            ("D", PURPLE, "white", "Population Bayes"),
        )
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        frameon=False,
        ncol=3,
        fontsize=8.6,
        columnspacing=1.6,
        handlelength=0.8,
    )


def standard(ax, contrasts):
    subset = contrasts[contrasts.group == "standard"].set_index("method")
    names = {
        "full": "Full weighting",
        "cap2": "Cap at 2",
        "guarded": "ESS guard",
        "mean": "Mean",
        "top_n": "Top-N",
        "lr": "Logistic stacking",
    }
    if not subset.loc[list(names)].banks.eq(180).all():
        raise ValueError(
            "Confirmation expects six banks for each of 30 standard scenarios"
        )
    draw_points(ax, subset.loc[list(names)[:3]], np.arange(3), BLUE, "o")
    draw_points(ax, subset.loc[list(names)[3:]], np.arange(3, 6), "#555c65", "o")
    ticks = [
        f"{name}  ({1000 * subset.loc[method, 'delta']:+.1f})"
        for method, name in names.items()
    ]
    labels(ax, ticks, "Brier difference vs native OKAPI (×1,000)")
    ax.set_title("B   All 30 standard scenarios", loc="left", pad=11)


def stages(ax):
    rounds = (
        (3, "discovery_only", "R3    Discovery only"),
        (3, "full", "R3    All stages"),
        (4, "selection_only", "R4    Selection only"),
        (4, "refit_only", "R4    Refit only"),
        (4, "full", "R4    All stages"),
    )
    for i, (r, m, _) in enumerate(rounds):
        c = table(r, "contrasts")
        row = c[
            (c.group == "mixed/max_regret")
            & (c.reference == "native")
            & (c.method == m)
        ]
        if len(row) != 1 or row.banks.iloc[0] != 6:
            raise ValueError("Stage isolation expects six source draws per round")
        draw_points(ax, row, [i], BLUE if m == "full" else "#555c65", "o")
    labels(
        ax,
        [label for _, _, label in rounds],
        "Change in worst endpoint regret (×1,000)",
    )
    ax.set_title("C   Where weights enter native OKAPI", loc="left", pad=11)


def main():
    plt.rcParams.update(
        {
            "font.size": 9.8,
            "axes.titlesize": 12.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.5,
            "pdf.fonttype": 42,
        }
    )
    c = table(10, "contrasts")
    c = c[c.reference == "native"]
    fig = plt.figure(figsize=(13.8, 8.1))
    layout = fig.add_gridspec(
        2,
        2,
        left=0.13,
        right=0.973,
        top=0.86,
        bottom=0.21,
        width_ratios=[1.27, 1],
        wspace=0.55,
        hspace=0.65,
    )
    focal(
        fig.add_subplot(layout[:, 0]),
        c,
        table(10, "support-contrasts"),
        pd.read_csv(RESULTS / "population.csv"),
    )
    standard(fig.add_subplot(layout[0, 1]), c)
    stages(fig.add_subplot(layout[1, 1]))
    fig.text(
        0.029,
        0.966,
        "How much does correcting the training case mix change transfer?",
        fontsize=19,
        fontweight="bold",
        va="top",
    )
    fig.text(
        0.029,
        0.922,
        "Negative differences favor the weighted method or named comparator. Lines show paired 95% bootstrap intervals.",
        fontsize=10.6,
        color="#42444a",
        va="top",
    )
    fig.text(
        0.029,
        0.055,
        "A compares each method with its own unweighted fit. Regret is the larger excess over the Bayes Brier of the inactive and active shortcut worlds.\n"
        "A and B use fresh confirmation draws (20 per focal profile and six per standard scenario). C uses six mixed-source draws in each round.",
        fontsize=9.1,
        color="#42444a",
        va="bottom",
        linespacing=1.5,
    )
    for ext in ("png", "pdf"):
        fig.savefig(RESULTS / f"r10-case-mix.{ext}", dpi=190, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()

"""Draw the full-data comparisons from the frozen method and confirmation tables."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

OUT = Path(__file__).resolve().parent / "results"
GROUPS = ["weak", "strong", "scene16", "scene128", "standard", "digits_449"]
HEADERS = [
    "Weak signal",
    "Strong signal",
    "16 scenes",
    "128 scenes",
    "All 27 families",
    "Digits (449)",
]
NAMES = {
    "reduced": "Train and choose on 70%",
    "split": "Train on 70%, choose on 30%",
    "seeded_mean": "Seed a tree for the full mean",
    "pull_mean": "Mean seed and distance penalty",
    "paired_se": "Mean seed and uncertainty penalty",
    "full": "Original full-data fit",
}
INK, MUTED = "#172a37", "#415565"
BLUE, ORANGE, GREY = "#176780", "#a34520", "#697786"


def signed(x):
    if abs(x) < 0.00005:
        return "0.0000"
    return f"{x:+.4f}".replace("-", "−")


def read_tables(chosen, out):
    pairs = [
        (1, "reduced"),
        (1, "split"),
        (2, "seeded_mean"),
        (3, "pull_mean"),
        (4, "paired_se"),
        (5, chosen),
    ]
    changes = []
    for r, method in pairs:
        levels = pd.read_csv(out / f"round{r}-levels.csv").pivot(
            index="group", columns="method", values="brier"
        )
        delta = (levels[method] - levels["full"]).loc[GROUPS]
        if method != "full":
            c = pd.read_csv(out / f"round{r}-contrasts.csv")
            confirmed = c[
                (c.metric == "brier") & (c.method == method) & (c.reference == "full")
            ]
            np.testing.assert_allclose(
                delta, confirmed.set_index("group").loc[GROUPS, "delta"], atol=1e-14
            )
        changes.append(delta)
    frame = pd.read_csv(out / "round5-families.csv")
    family = frame.loc[frame.method == chosen, ["family", "delta_full"]]
    if family.family.nunique() != 27 or len(family) != 27:
        raise ValueError("Every standard family is required for the figure")
    np.testing.assert_allclose(
        family.delta_full.mean(), changes[-1]["standard"], atol=1e-14
    )
    return pairs, np.array(changes), family.sort_values(["delta_full", "family"])


def draw(chosen, out=OUT):
    pairs, changes, family = read_tables(chosen, out)
    fig = plt.figure(figsize=(14.2, 9.1), facecolor="white")
    grid = fig.add_gridspec(
        2,
        3,
        left=0.035,
        right=0.98,
        bottom=0.10,
        top=0.84,
        hspace=0.54,
        wspace=0.16,
        height_ratios=[1.03, 1.46],
    )
    fig.text(
        0.035,
        0.965,
        "Selecting an OKAPI tree with all fusion-training examples",
        fontsize=22,
        color=INK,
        weight="semibold",
        va="top",
    )
    fig.text(
        0.035,
        0.907,
        "Held-out Brier difference from ordinary full-data OKAPI. "
        "Negative values favor the method in that row. Each round uses fresh draws.",
        fontsize=11.4,
        color=MUTED,
        va="top",
    )

    ax = fig.add_subplot(grid[0, :])
    ax.axis("off")
    ax.text(
        0,
        1.09,
        "Change by evaluation setting",
        transform=ax.transAxes,
        fontsize=16,
        weight="semibold",
        color=INK,
    )
    bounds = [0, 0.295, 0.41, 0.525, 0.64, 0.75, 0.887, 1.0]
    for j, head in enumerate(["Round and intervention", *HEADERS]):
        ax.text(
            bounds[j] if j == 0 else (bounds[j] + bounds[j + 1]) / 2,
            1,
            head,
            transform=ax.transAxes,
            ha="left" if j == 0 else "center",
            va="center",
            fontsize=10.4,
            weight="semibold",
            color=INK,
        )
    for i, (r, method) in enumerate(pairs):
        y = 0.865 - i * 0.151
        label = f"{r}   {NAMES.get(method, method)}"
        if r == 5:
            label = {
                "pull_mean": "5   Re-test mean seed and distance penalty",
                "seeded_mean": "5   Re-test mean initialization",
                "paired_se": "5   Re-test mean seed and uncertainty penalty",
            }.get(method, f"5   Confirm {method}")
        ax.text(
            0,
            y,
            label,
            transform=ax.transAxes,
            va="center",
            fontsize=10.5,
            color=INK,
            weight="semibold" if r == 5 else "normal",
        )
        if i in (0, 5):
            ax.plot(
                [0, 1], [y + 0.071] * 2, transform=ax.transAxes, color="#c2d0d5", lw=0.9
            )
        for j, value in enumerate(changes[i], start=1):
            color = BLUE if value < -1e-12 else ORANGE if value > 1e-12 else GREY
            ax.text(
                (bounds[j] + bounds[j + 1]) / 2,
                y,
                signed(value),
                color=color,
                transform=ax.transAxes,
                va="center",
                ha="center",
                fontsize=11,
                weight="semibold" if r == 5 else "normal",
                family="DejaVu Sans Mono",
            )

    delta = family.delta_full
    improved, worse = int((delta < -1e-12).sum()), int((delta > 1e-12).sum())
    fig.text(
        0.035,
        0.487,
        "Confirmation on each standard scenario",
        fontsize=16,
        weight="semibold",
        color=INK,
    )
    fig.text(
        0.035,
        0.455,
        f"Across 27 family means: {improved} improved, {27 - improved - worse} tied and "
        f"{worse} worsened. Each mean covers four paired draws.",
        color=MUTED,
        fontsize=10.7,
    )
    lim = max(0.002, abs(delta).max() * 1.17)
    for j in range(3):
        # The labels, dots and values have dedicated space within each block.
        block = grid[1, j].subgridspec(1, 3, width_ratios=[1.32, 1, 0.62], wspace=0.01)
        labels, dots, values = [fig.add_subplot(block[0, k]) for k in range(3)]
        part = family.iloc[9 * j : 9 * j + 9]
        for axis in [labels, dots, values]:
            axis.set_ylim(8.6, -0.6)
            axis.set_yticks([])
            axis.spines[:].set_visible(False)
        for axis in [labels, values]:
            axis.set_xlim(0, 1)
            axis.set_xticks([])
        for i, row in enumerate(part.itertuples()):
            value = row.delta_full
            color = BLUE if value < -1e-12 else ORANGE if value > 1e-12 else GREY
            labels.text(
                0, i, row.family.replace("_", " "), color=INK, fontsize=9.7, va="center"
            )
            dots.plot([0, value], [i, i], color=color, lw=1.25)
            dots.plot(value, i, "o", color=color, markersize=4)
            values.text(
                0.98,
                i,
                signed(value),
                color=color,
                fontsize=9.6,
                ha="right",
                va="center",
                family="DejaVu Sans Mono",
            )
        dots.axvline(0, color="#adbac2", lw=0.8, zorder=0)
        dots.set_xlim(-lim, lim)
        dots.xaxis.set_major_locator(MaxNLocator(nbins=3))
        dots.tick_params(axis="x", length=0, colors=MUTED, labelsize=8)
    fig.text(
        0.035,
        0.035,
        "Digits entries summarize repeated partitions of one dataset. "
        "See the paired contrast tables for uncertainty intervals and sample counts.",
        fontsize=10,
        color=MUTED,
    )
    fig.savefig(out / "full-data-five.pdf", bbox_inches="tight")
    fig.savefig(out / "full-data-five.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "chosen", nargs="?", help="Optionally check the name in the frozen record"
    )
    supplied = parser.parse_args().chosen
    from decide import read_locked

    chosen = read_locked()["chosen"]
    if supplied is not None and supplied != chosen:
        parser.error(f"Expected the frozen method {chosen}")
    draw(chosen)

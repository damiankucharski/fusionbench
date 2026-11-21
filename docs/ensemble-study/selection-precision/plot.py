"""Plot exact population excess Brier on the finite four-provider challenge."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot(folder):
    data = pd.read_csv(folder / "lattice-means.csv")
    x = data[
        (data.n_fit == 256)
        & (data.budget.isna() | ((data.budget == 400) & data.units.eq("nominal")))
        & data.selector.isin(("fit", "fixed"))
    ].copy()
    x["name"] = x["method"].where(x.method.ne("gp"), "GP, " + x.precision.fillna(""))
    names = ("exact_erm", "GP, round3", "GP, full", "mean", "top_n", "stack_logit")
    labels = (
        "Exact empirical MIN/MAX optimum",
        "GP, three decimals",
        "GP, unrounded",
        "Mean of four providers",
        "Ranked-prefix Top-N",
        "Logistic stacking",
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharex=True, sharey=True)
    for axis, (law, title) in zip(
        axes,
        (
            ("matched", "Two disjoint MIN pathways"),
            ("additive", "Additive logit control"),
        ),
        strict=True,
    ):
        sub = x[x.law.eq(law)].set_index("name").loc[list(names)]
        ys = np.arange(len(names))
        for i, v in zip(ys, sub.iid_excess_bayes, strict=True):
            color = "#17637a" if i < 3 else "#786f55"
            axis.plot([0, v], [i, i], lw=3, color=color, alpha=0.5)
            axis.scatter(v, i, color=color, s=40, zorder=3)
            axis.annotate(
                f"{v:.4f}",
                (v, i),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                fontsize=9,
            )
        axis.set_title(title)
        axis.set_xlabel("Exact population excess Brier")
        axis.set_xlim(-0.003, 0.09)
        axis.tick_params(axis="y", length=0)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.grid(axis="y", color="#e4e7e8", lw=0.7)
        axis.set_axisbelow(True)
    axes[0].set_yticks(np.arange(len(names)), labels)
    axes[0].invert_yaxis()
    fig.suptitle(
        "Finding the best fusion rule on 256 known input states",
        fontweight="bold",
        x=0.02,
        ha="left",
    )
    fig.text(
        0.02,
        0.88,
        "256 training labels · 12 data banks · 400 true fitness evaluations per native start",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.85))
    fig.savefig(folder / "finite-population.png", dpi=160, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    plot(parser.parse_args().directory)

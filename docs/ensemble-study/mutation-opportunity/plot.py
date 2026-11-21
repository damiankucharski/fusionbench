"""Plot paired retype-versus-identity differences for every default IID scenario."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot(folder):
    dif = pd.read_csv(folder / "paired.csv.gz")
    sub = dif.query(
        "selection == 'fit' and split == 'iid' and metric == 'brier' and a == 'retype' and b == 'identity'"
    )
    order = sorted(sub.scenario.unique())
    y = np.arange(len(order))
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 8.2), sharey=True, sharex=True)
    for ax, (cp, title) in zip(
        axes,
        (
            ("calls", "1,200 training fitness calls"),
            ("generation", "10 completed generations"),
        ),
        strict=True,
    ):
        g = sub[sub.checkpoint.eq(cp)].set_index("scenario").loc[order]
        ax.axvline(0, color="#54606a", linewidth=1)
        ax.errorbar(
            g["mean"],
            y,
            xerr=[g["mean"] - g.low_95, g.high_95 - g["mean"]],
            fmt="o",
            ms=4.7,
            lw=1.6,
            capsize=2.2,
            color="#135e77",
            ecolor="#5b8f9d",
            zorder=3,
        )
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Change in full Brier score")
        ax.grid(axis="y", color="#e8edf0", linewidth=0.8)
        ax.tick_params(axis="y", length=0)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.set_axisbelow(True)
    axes[0].set_yticks(y, [name.replace("_", " ") for name in order])
    axes[0].invert_yaxis()
    fig.suptitle(
        "Replacing an operator family versus leaving the same mutation opportunity neutral",
        x=0.02,
        y=0.985,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.02,
        0.945,
        "Held-out IID test cases · lower favors retype · paired 95% intervals over six data banks per scenario",
        ha="left",
        fontsize=10,
        color="#43545f",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=2)
    fig.savefig(folder / "retype-effects.png", dpi=150, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    plot(parser.parse_args().directory)

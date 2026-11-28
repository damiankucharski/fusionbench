"""Final environment study: diagnosis, deployment trade-off and standard regression."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
NAMES = {
    "raw": "Raw filter",
    "adjusted": "Adjusted filter",
    "strict": "Strict adjusted filter",
    "native": "Native reference",
    "mean": "Mean",
    "top_n": "Top-N",
    "lr": "Logistic stacking",
    "sham": "Shuffled environments",
}
COLORS = {
    "raw": "#B26E20",
    "adjusted": "#008179",
    "strict": "#277393",
    "native": "#232E3F",
    "mean": "#A74748",
    "lr": "#777888",
    "top_n": "#827043",
    "sham": "#ADACB2",
}


def main(r):
    levels = pd.read_csv(OUT / f"r{r:02}-levels.csv")
    drops = pd.read_csv(OUT / f"r{r:02}-drops.csv")
    chosen = json.loads((HERE / "selection.json").read_text())["chosen"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
    panels = [
        ("case_mix", "specialist_drop_rate", "Case mix only\nspecialist discarded"),
        ("mixed", "shortcut_drop_rate", "Both vary\nshortcut discarded"),
        ("reversed", "shortcut_drop_rate", "Reversed pairing\nshortcut discarded"),
        ("small128", "shortcut_drop_rate", "128 observations\nshortcut discarded"),
    ]
    for j, method in enumerate(
        ("raw", chosen, "sham") if chosen != "raw" else ("adjusted", chosen, "sham")
    ):
        heights = [
            drops[
                (drops.family == f) & (drops.method == method) & (drops.world == "s0")
            ].iloc[0][metric]
            for f, metric, _ in panels
        ]
        left.bar(
            np.arange(len(panels)) + (j - 1) * 0.23,
            heights,
            width=0.22,
            color=COLORS[method],
            label=NAMES[method],
        )
    left.set_xticks(np.arange(len(panels)), [lab for _, _, lab in panels], fontsize=8)
    left.set_ylim(0, 1.09)
    left.set_ylabel("Fraction of 24 source draws")
    left.set_title("Which provider did the filter discard?", loc="left")
    left.legend(fontsize=8, frameon=False)
    support = ["s0", "s25", "s50", "s75", "s100"]
    for name in ("mean", "top_n", "lr", "native", chosen):
        rows = levels[(levels.method == name) & (levels.family == "mixed")].set_index(
            "world"
        )
        right.plot(
            [0, 25, 50, 75, 100],
            rows.loc[support, "brier"],
            marker="o",
            ms=4,
            color=COLORS[name],
            label=NAMES[name],
            ls="--" if name == chosen else "-",
        )
    oracle = pd.read_csv(OUT / f"r{r:02}-oracle.csv")
    low = oracle.query("family=='mixed' and world=='s0'").iloc[0].full_bayes_brier
    right.axhline(low, color="#444444", ls=":", lw=1.2, label="Minimax bound")
    right.set_xticks([0, 25, 50, 75, 100])
    right.set_xlabel("Target rows with an active shortcut (%)")
    right.set_ylabel("Expected full-class Brier")
    right.set_title("Same source, five possible targets", loc="left")
    right.legend(fontsize=8, frameon=False, ncol=2)
    for ax in [left, right]:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.16)
        ax.set_axisbelow(True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"environment-ten.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main(10)

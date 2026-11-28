"""Draw the registered endpoint and show the consequences of an isolated shortcut shift."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
LABELS = {
    "cv": "Minimum cross-validated loss",
    "one_se": "Simplicity within one SE",
    "paired": "Gain of one paired SE",
    "sign": "Gain in all three folds",
    "stress": "Gain at low and high disagreement",
    "augmented": "Native search with two extra pools",
}


def main():
    chosen = json.loads((HERE / "selection.json").read_text())["chosen"]
    contrasts = pd.read_csv(OUT / "r10-contrasts.csv")
    levels = pd.read_csv(OUT / "r10-levels.csv")
    fig, (left, right) = plt.subplots(
        1,
        2,
        figsize=(11.4, 4.7),
        layout="constrained",
        gridspec_kw={"width_ratios": [1.28, 1]},
    )
    colors = {
        "native": "#343c4d",
        "mean": "#b04a31",
        "top_n": "#986b13",
        "lr": "#686e76",
        chosen: "#087c71",
    }
    panel = contrasts[
        (contrasts.group == "standard") & (contrasts.reference == "native")
    ].set_index("method")
    for i, k in enumerate(LABELS):
        point = panel.loc[k]
        col = colors[chosen] if k == chosen else "#546174"
        left.hlines(i, point.lo, point.hi, color=col, lw=1.8)
        left.scatter(
            point.delta,
            i,
            color=col,
            marker="D" if k == chosen else "o",
            s=48,
            zorder=3,
        )
    left.set_yticks(
        range(len(LABELS)),
        [LABELS[k] + (" *" if k == chosen else "") for k in LABELS],
        fontsize=9,
    )
    left.invert_yaxis()
    left.axvline(0, ls="--", lw=1, color="#697080")
    left.set_title("Fresh draws on 30 scenarios", loc="left")
    left.set_xlabel("Test Brier minus native reference")
    names = {
        "native": "Native reference",
        "mean": "Mean",
        "top_n": "Top-N",
        "lr": "Logistic stacking",
        chosen: "Locked rule",
    }
    for k in ("lr", "top_n", "mean", "native", chosen):
        v = (
            levels[levels.method == k]
            .set_index("group")
            .loc[["shortcut_100", "shortcut_050", "shortcut_000"], "brier"]
        )
        right.plot(
            [100, 50, 0],
            v,
            marker="D" if k == chosen else "o",
            ms=4.5,
            label=names[k],
            color=colors[k],
            ls="--" if k == chosen else "-",
            lw=1.5,
        )
    right.set_xticks([100, 50, 0], ["100%", "50%", "0%"])
    right.invert_xaxis()
    right.set_xlabel("Fraction of target rows with active shortcut")
    right.set_ylabel("Expected full-class Brier")
    right.set_title("Same source, changing target", loc="left")
    right.legend(fontsize=8, frameon=False)
    fig.suptitle("Choosing when to simplify OKAPI", fontsize=16, weight="bold")
    for ax in (left, right):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", alpha=0.15)
    fig.text(
        0.01,
        -0.045,
        "* Rule selected on 24 development families. Intervals resample data seeds within the 30 fixed scenarios. Lower Brier is better.",
        fontsize=9,
    )
    for ext in ["png", "pdf"]:
        fig.savefig(OUT / f"native-choices.{ext}", bbox_inches="tight", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()

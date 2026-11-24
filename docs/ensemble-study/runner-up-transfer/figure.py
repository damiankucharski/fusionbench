"""Plot the frozen fit profiles and historical matched class-permutation contrasts."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent


def plot():
    centers = pd.read_csv(HERE / "predictions/centroids.csv")
    contrasts = pd.read_csv(HERE / "permutation/contrasts.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.5), layout="constrained")
    blue, orange = "#175984", "#aa4b1c"
    ax = axes[0]
    measure = "wrong_class_repeatability_scaled"
    for y, label in enumerate(["Digits", "Iris", "Seeds"]):
        s = centers[
            (centers.id == label.lower()) & (centers.source == "external_fit")
        ].iloc[0]
        sim = centers[(centers.source == "synthetic") & (centers.stratum == s.stratum)]
        ax.plot(
            [sim[measure].min(), sim[measure].max()],
            [y, y],
            color=blue,
            lw=6,
            solid_capstyle="round",
            label="Range of 9 synthetic means" if y == 0 else None,
        )
        ax.plot(
            s[measure],
            y,
            "o",
            color=orange,
            label="External fit data" if y == 0 else None,
        )
    ax.set(
        yticks=[0, 1, 2],
        yticklabels=["Digits", "Iris", "Seeds"],
        ylim=(2.6, -0.6),
        xlabel="Wrong-class repeatability",
        title="Fit-side coverage",
    )
    ax.set_xlim(-0.012, 0.25)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.48), frameon=False)
    ax = axes[1]
    data = contrasts[contrasts.comparison == "native_minus_prob_lr"]
    for cohort, label, color in [
        ("digits", "Historical digits", orange),
        ("panel_E", "Directed synthetic", blue),
    ]:
        d = (
            data[data.cohort == cohort]
            .set_index("quantity")
            .loc[["raw_gap", "permuted_gap"]]
        )
        ax.plot([0, 1], d["mean"], marker="o", lw=2, label=label, color=color)
    ax.axhline(0, color="0.45", lw=0.8)
    ax.set(
        xticks=[0, 1],
        xticklabels=["Original", "Permuted per row"],
        xlim=(-0.15, 1.15),
        ylabel="Brier difference (OKAPI minus stacking)",
        title="Effect of class identity",
    )
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.48), frameon=False)
    for panel in axes:
        panel.spines[["top", "right"]].set_visible(False)
    fig.savefig(HERE / "coverage-and-permutation.png", dpi=175, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


if __name__ == "__main__":
    plot()

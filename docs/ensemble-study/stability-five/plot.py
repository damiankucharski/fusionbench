"""Plot the five prespecified comparisons and the locked candidate's confirmation."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MaxNLocator

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
SCREEN = [
    (1, "gp_top8"),
    (2, "simplex"),
    (3, "bootstrap_medoid"),
    (4, "cf_gain"),
    (4, "cf_margin"),
]
NAMES = {
    "gp_top8": "Eight-tree shortlist",
    "simplex": "Penalized simplex",
    "bootstrap_medoid": "Bootstrap medoid",
    "cf_gain": "Cross-fit gain",
    "cf_margin": "Cross-fit one-SE",
}
INK, MUTED, GUIDE, TEAL, RUST = "#202d35", "#495963", "#d5dedf", "#00736e", "#b54c22"
FAMILY_NAMES = {
    "hardrescue": "Hard rescue",
    "hardveto": "Hard veto",
    "similar_classes_38": "Similar classes (3 ↔ 8)",
    "native_close_switch": "Native CLOSE switch",
}


def signed(x, digits=3):
    if abs(x) < 0.5 * 10**-digits:
        return "0"
    return f"{x:+.{digits}f}".replace("-", "−")


def color(value):
    return MUTED if abs(value) < 1e-12 else TEAL if value < 0 else RUST


def read_results():
    with (HERE / "selection.json").open() as f:
        lock = json.load(f)
    chosen = lock["candidate"]
    rounds = SCREEN + [(5, chosen)]
    contrasts, levels = {}, {}
    for r in range(1, 6):
        contrasts[r] = pd.read_csv(OUT / f"round{r}-contrasts.csv").set_index(
            ["group", "method", "reference"]
        )
        levels[r] = pd.read_csv(OUT / f"round{r}-levels.csv").set_index(
            ["group", "method"]
        )
    family = pd.read_csv(OUT / "round5-families.csv")
    family = family[family.method == chosen].sort_values(
        ["delta_reg", "family"], ascending=[False, True]
    )
    if len(family) != 27 or family.family.nunique() != 27:
        raise ValueError("Confirmation must have all 27 distinct scenario families")
    for r, method in rounds:
        for group, ref in (
            ("weak", "regularized"),
            ("standard", "regularized"),
            ("digits_449", "regularized"),
            ("weak", "mean"),
        ):
            row = contrasts[r].loc[(group, method, ref)]
            a = levels[r].loc[(group, method), "brier"]
            b = levels[r].loc[(group, ref), "brier"]
            np.testing.assert_allclose(row.delta, a - b, rtol=0, atol=2e-14)
    np.testing.assert_allclose(
        family.delta_reg.mean(),
        contrasts[5].loc[("standard", chosen, "regularized"), "delta"],
        atol=2e-14,
    )
    return chosen, rounds, contrasts, levels, family


def axes_style(ax):
    ax.spines[["top", "bottom", "right", "left"]].set_visible(False)
    ax.tick_params(axis="both", length=0, pad=4, colors=MUTED, labelsize=8.4)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=GUIDE, linewidth=0.5, alpha=0.65)
    ax.axvline(0, color="#697981", linewidth=0.9)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: signed(x)))


def draw(out=OUT):
    chosen, rounds, contrasts, levels, family = read_results()
    fig = plt.figure(figsize=(15.2, 9.4), facecolor="white")
    fig.text(
        0.033,
        0.96,
        "How each change affected held-out Brier score",
        fontsize=23,
        fontweight="semibold",
        color=INK,
        va="top",
    )
    fig.text(
        0.033,
        0.909,
        "Negative differences favor the candidate. Each round uses fresh banks. "
        "Round 5 tests the locked choice.",
        fontsize=11,
        color=MUTED,
    )
    groups = [
        ("weak", "regularized", "Weak specialization", "vs regularized OKAPI"),
        ("standard", "regularized", "27-family average", "vs regularized OKAPI"),
        ("digits_449", "regularized", "Digits · 449 fit rows", "vs regularized OKAPI"),
        ("weak", "mean", "Weak specialization", "vs arithmetic mean"),
    ]
    labels = [f"R{r}   {NAMES[m]}" for r, m in rounds]
    y = np.array([0, 1, 2, 3, 4, 5.65])
    for j, (group, reference, heading, subheading) in enumerate(groups):
        left, width = 0.186 + j * 0.205, 0.169
        ax = fig.add_axes([left, 0.555, width, 0.255])
        ax.set_facecolor("none")
        axes_style(ax)
        ax.set_ylim(6.18, -0.55)
        ax.axhline(4.85, color=GUIDE, linewidth=0.9)
        ax.set_yticks(y, labels=labels if j == 0 else [""] * len(labels))
        if not j:
            for i, label in enumerate(ax.get_yticklabels()):
                label.set_fontsize(9.6)
                label.set_color(INK)
                if i == len(labels) - 1:
                    label.set_fontweight("semibold")
        values = []
        for pos, (r, method) in zip(y, rounds):
            row = contrasts[r].loc[(group, method, reference)]
            value = row.delta
            values.extend([value, 0])
            if group != "digits_449":
                assert np.isfinite([row.lo, row.hi]).all()
                ax.hlines(pos, row.lo, row.hi, color=color(value), linewidth=1.65)
                values.extend([row.lo, row.hi])
            else:
                assert np.isnan(row.lo) and np.isnan(row.hi)
            ax.scatter(
                value,
                pos,
                s=48 if r == 5 else 28,
                marker="D" if r == 5 else "o",
                color=color(value),
                edgecolor="white",
                linewidth=0.55,
                zorder=3,
            )
        vmin, vmax = min(values), max(values)
        pad = 0.11 * max(vmax - vmin, 0.001)
        ax.set_xlim(vmin - pad, vmax + pad)
        fig.text(left, 0.859, heading, color=INK, fontsize=12, weight="semibold")
        fig.text(left, 0.835, subheading, color=MUTED, fontsize=9.5)
        confirmed = contrasts[5].loc[(group, chosen, reference), "delta"]
        fig.text(
            left,
            0.513,
            f"R5   {signed(confirmed, 4)}",
            color=color(confirmed),
            fontsize=9.5,
            weight="semibold",
        )

    digits100 = contrasts[5].loc[("digits_100", chosen, "regularized"), "delta"]
    fig.text(
        0.033,
        0.492,
        "Bars show paired 95% bootstrap intervals for the controlled banks and the fixed suite. "
        f"Digits splits overlap. With 100 digits for fitting, R5 minus regularized was {signed(digits100, 4)}.",
        color=MUTED,
        fontsize=9.1,
    )
    fig.text(
        0.033,
        0.436,
        "Confirmation on all 27 scenario families",
        fontsize=16.5,
        color=INK,
        weight="semibold",
    )
    per_family_n = int(levels[5].loc[("standard", chosen), "seeds_per_family"])
    fig.text(
        0.033,
        0.405,
        f"{NAMES[chosen]} minus regularized OKAPI. Each point averages {per_family_n} fresh "
        "banks. Families are ordered from largest loss to largest gain, reading down each column.",
        color=MUTED,
        fontsize=9.7,
    )

    # A shared horizontal scale keeps the nine-family columns directly comparable.
    low, high = min(0.0, family.delta_reg.min()), max(0.0, family.delta_reg.max())
    pad = max(0.0004, 0.07 * (high - low))
    limits = (low - pad, high + pad)
    for j, start in enumerate((0, 9, 18)):
        part = family.iloc[start : start + 9]
        ax = fig.add_axes([0.162 + j * 0.326, 0.077, 0.125, 0.29])
        axes_style(ax)
        ax.set_xlim(*limits)
        ax.set_ylim(8.55, -0.65)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=2))
        ax.set_yticks(
            range(9),
            labels=[
                FAMILY_NAMES.get(f, f.replace("_", " ").capitalize())
                for f in part.family
            ],
        )
        ax.tick_params(axis="y", labelsize=9.1)
        for pos, value in enumerate(part.delta_reg):
            ax.hlines(pos, 0, value, color=color(value), alpha=0.6, linewidth=1.2)
            ax.scatter(
                value,
                pos,
                marker="o",
                color=color(value),
                s=28,
                zorder=3,
                edgecolor="white",
                linewidth=0.5,
            )
            ax.text(
                1.035,
                pos,
                signed(value, 4),
                transform=ax.get_yaxis_transform(),
                fontsize=8,
                color=color(value),
                va="center",
                ha="left",
                clip_on=False,
            )
    improved = int((family.delta_reg < -1e-12).sum())
    worsened = int((family.delta_reg > 1e-12).sum())
    tied = len(family) - improved - worsened
    standard = contrasts[5].loc[("standard", chosen, "regularized")]
    fig.text(
        0.033,
        0.018,
        f"Across family averages, {improved} improved, {tied} tied and {worsened} worsened. "
        f"Across individual banks, {int(standard.worse_banks)} of {int(standard.total)} had higher Brier. "
        "Family effects are descriptive.",
        fontsize=8.8,
        color=MUTED,
    )
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "stability-five.png", dpi=190, facecolor="white")
    fig.savefig(
        out / "stability-five.pdf",
        facecolor="white",
        metadata={"Title": "Five rounds of native OKAPI stability experiments"},
    )
    plt.close(fig)


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    draw()

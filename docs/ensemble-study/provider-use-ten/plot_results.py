"""Plot the frozen R10 comparison and summarize the paired R3/R10 selector controls."""

import gzip
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
NAMES = {
    "native": "Native reference",
    "hard": "Remove and refit",
    "half": "Uniform half",
    "protect": "Protect",
    "adaptive": "Adaptive",
    "mean": "Mean",
    "top_n": "Top-N",
    "lr": "Logistic stacking",
}
ORIGINAL = "#23657A"
SHUFFLED = "#B05B25"
INK = "#293A44"
MUTED = "#576A74"
PAPER = "#FFFFFF"


def selector_data(r):
    records = []
    with gzip.open(OUT / f"r{r:02}-source.jsonl.gz", "rt") as stream:
        for row in map(json.loads, stream):
            if row["cohort"] == "gate_control":
                evidence = row["evidence"][f"{row['selector']}/real"]
                records.append(
                    {
                        "family": row["family"],
                        "seed": row["seed"],
                        "direct_gain": evidence["direct_gain"],
                        "routing_gain": evidence["routing_gain"],
                    }
                )
    wide = pd.DataFrame(records).pivot(index="seed", columns="family")
    n = {3: 6, 10: 24}[r]
    worlds = ["gated_original", "gated_shuffled"]
    if (
        len(wide) != n
        or set(wide.columns.get_level_values("family")) != set(worlds)
        or not np.isfinite(wide.to_numpy()).all()
    ):
        raise ValueError("Incomplete pairs for the selector control")
    # Resample the source bank, keeping both worlds and both metrics paired.
    idx = np.random.default_rng(93000 + r).integers(0, n, size=(10_000, n))
    summary = []
    for metric in ("direct_gain", "routing_gain"):
        a, b = wide[metric][worlds].to_numpy().T
        for group, vector in zip(
            (*worlds, "original_minus_shuffled"), (a, b, a - b), strict=True
        ):
            lo, hi = np.quantile(vector[idx].mean(axis=1), [0.025, 0.975])
            summary.append(
                {
                    "round": r,
                    "metric": metric,
                    "sample": group,
                    "paired_source_banks": n,
                    "mean": vector.mean(),
                    "lo": lo,
                    "hi": hi,
                    "banks_above_002": (vector > 0.02).sum()
                    if group in worlds
                    else None,
                }
            )
    return wide, pd.DataFrame(summary)


def compare_methods(ax, group, title, levels, contrasts, chosen, show_names):
    levels = levels[levels.group == group].set_index("method")
    comp = contrasts[
        (contrasts.group == group) & (contrasts.reference == "native")
    ].set_index("method")
    ax.axvline(0, color=MUTED, lw=1, zorder=1)
    ax.axhline(4.5, color="#CFD6D8", lw=0.7)
    for pos, method in enumerate(NAMES):
        if method == "native":
            delta = lo = hi = 0.0
        else:
            delta, lo, hi = comp.loc[method, ["delta", "lo", "hi"]]
        selected = method == chosen
        color = ORIGINAL if selected else INK if pos < 5 else MUTED
        ax.plot([lo, hi], [pos, pos], lw=1.6, color=color, zorder=2)
        ax.scatter(
            delta,
            pos,
            s=56 if selected else 32,
            marker="D" if selected else "o",
            edgecolors=color,
            facecolors=color if pos < 5 else PAPER,
            zorder=3,
        )
    ax.set_ylim(len(NAMES) - 0.4, -0.7)
    labels = [v + (" *" if k == chosen else "") for k, v in NAMES.items()]
    ax.set_yticks(range(len(NAMES)), labels if show_names else [])
    if show_names:
        tick = ax.get_yticklabels()[list(NAMES).index(chosen)]
        tick.set_color(ORIGINAL)
        tick.set_fontweight("bold")
    ax.set_title(title, loc="left", fontsize=11, weight="bold", color=INK, pad=6)
    n = levels.at["native", "banks"]
    scope = f"{n // 30} draws/scenario" if group == "standard" else f"{n} source draws"
    ax.text(
        0,
        -0.205,
        f"Native {levels.at['native', 'risk']:.3f}  ·  {scope}",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=8.3,
    )
    delta, lo, hi = (
        (0.0, 0.0, 0.0)
        if chosen == "native"
        else comp.loc[chosen, ["delta", "lo", "hi"]]
    )
    delta_text, lo_text, hi_text = [f"{x:+.2g}" if x else "0" for x in (delta, lo, hi)]
    ax.text(
        0,
        -0.3,
        f"Chosen Δ {delta_text}  [{lo_text}, {hi_text}]",
        transform=ax.transAxes,
        color=ORIGINAL,
        fontsize=8.2,
    )
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.grid(axis="x", color="#DDE3E6", linewidth=0.7)
    ax.set_axisbelow(True)


def compare_selector(ax, pairs, summary):
    summary = summary.set_index(["metric", "sample"])
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.axhline(0.02, color=MUTED, lw=0.85, ls=(0, (2, 3)))
    ax.text(
        0.0,
        0.027,
        "Gain threshold 0.02",
        transform=ax.get_yaxis_transform(),
        color=MUTED,
        fontsize=8.1,
        va="bottom",
    )
    for i, metric in enumerate(("direct_gain", "routing_gain")):
        left, right = i - 0.14, i + 0.14
        data = pairs[metric][["gated_original", "gated_shuffled"]].to_numpy()
        ax.plot([left, right], data.T, color="#9EABB1", alpha=0.43, lw=0.65, zorder=1)
        for x, sample, color, mark in (
            (left, "gated_original", ORIGINAL, "o"),
            (right, "gated_shuffled", SHUFFLED, "s"),
        ):
            mean, lo, hi = summary.loc[(metric, sample), ["mean", "lo", "hi"]]
            ax.vlines(x, lo, hi, lw=2, color=color, zorder=3)
            ax.hlines([lo, hi], x - 0.035, x + 0.035, color=color, lw=1.3, zorder=3)
            ax.scatter(
                x,
                mean,
                s=48,
                color=color,
                marker=mark,
                edgecolor=PAPER,
                linewidth=0.65,
                zorder=4,
            )
    ax.set_xticks([0, 1], ["Direct prediction\n(D)", "Choosing an expert\n(G)"])
    ax.set_xlim(-0.52, 1.5)
    ax.set_ylabel("Cross-fitted Brier gain (higher is better)")
    ax.grid(axis="y", color="#E4E8EA", lw=0.65)
    ax.set_axisbelow(True)
    handles = [
        Line2D([], [], marker=m, ls="none", color=c, label=t, markersize=6)
        for m, c, t in (("o", ORIGINAL, "Original"), ("s", SHUFFLED, "Shuffled"))
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        frameon=False,
        fontsize=9,
        handlelength=0.8,
        borderaxespad=0.25,
    )


def figure(levels, contrasts, pairs, selector, chosen):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "pdf.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(11.1, 4.7), facecolor=PAPER)
    grid = fig.add_gridspec(
        1,
        3,
        width_ratios=(1, 1, 1.6),
        left=0.135,
        right=0.985,
        bottom=0.29,
        top=0.78,
        wspace=0.42,
    )
    first, second, gate = [fig.add_subplot(grid[j]) for j in range(3)]
    compare_methods(
        first, "standard", "30-scenario mean Brier", levels, contrasts, chosen, True
    )
    compare_methods(
        second,
        "mixed/max_regret",
        "Mixed worst-case regret",
        levels,
        contrasts,
        chosen,
        False,
    )
    compare_selector(gate, pairs, selector)
    for ax in (first, second, gate):
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#BBC6CA")
        ax.tick_params(axis="y", length=0)
    pos = [ax.get_position() for ax in (first, second, gate)]
    fig.text(
        pos[0].x0,
        0.94,
        "A   Effect of changing provider use",
        fontsize=13,
        weight="bold",
    )
    fig.text(
        pos[2].x0 - 0.025,
        0.94,
        "B   What the selector contributes",
        fontsize=13,
        weight="bold",
    )
    fig.text(
        pos[0].x0,
        0.876,
        "Difference from native OKAPI (lower is better).",
        fontsize=9.3,
        color=MUTED,
    )
    fig.text(
        pos[2].x0 - 0.025,
        0.876,
        "Selector reports permuted independently in each split.",
        fontsize=9.1,
        color=MUTED,
    )
    fig.text(
        pos[0].x0,
        0.087,
        "*  Rule chosen in round 9. Intervals are paired 95% bootstrap intervals.",
        fontsize=8.7,
    )
    fig.text(
        pos[0].x0,
        0.042,
        "Worst regret is the larger excess over Bayes Brier at 0% or 100% active shortcut, calculated per source.",
        fontsize=8.7,
    )
    fig.text(
        pos[2].x0 - 0.025,
        0.173,
        f"Grey lines connect {len(pairs)} pairs of source banks.",
        fontsize=8.4,
        color=MUTED,
    )
    fig.text(
        pos[2].x0 - 0.025,
        0.128,
        "D compares with the source prior, G with one fixed expert.",
        fontsize=8.2,
        color=MUTED,
    )
    return fig


def main():
    for r in (3, 10):
        pairs, summary = selector_data(r)
        summary.to_csv(OUT / f"r{r:02}-selector.csv", index=False)
    decision = json.loads((HERE / "selection.json").read_text())
    fig = figure(
        pd.read_csv(OUT / "r10-levels.csv"),
        pd.read_csv(OUT / "r10-contrasts.csv"),
        pairs,
        summary,
        decision["chosen"],
    )
    fig.savefig(OUT / "provider-use-ten.png", dpi=220)
    fig.savefig(OUT / "provider-use-ten.pdf", metadata={"CreationDate": None})
    plt.close(fig)


if __name__ == "__main__":
    main()

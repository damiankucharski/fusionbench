"""Plot population retention, actual deployment and fresh-scene conditional loss."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, PercentFormatter

# The same symbols identify policies in the population, at deployment and on audit.
POLICIES = {
    "sham": ("Repeated-provider seed (sham)", "#986444", "s"),
    "seeded": ("Mean seed", "#1a7774", "o"),
    "carry": ("Mean seed retained", "#414c76", "D"),
}
LAWS = {"equal": "Equal competence", "signal": "One exact provider"}
GROUP_OFFSETS = np.linspace(-0.20, 0.20, len(POLICIES))


def score_rows(summary, metric):
    table = summary.loc[
        summary.allocation.eq("reuse")
        & summary.method.isin(POLICIES)
        & summary.metric.eq(metric)
    ].set_index(["law", "method"])
    if not table.index.is_unique or set(table.index) != {
        (law, method) for law in LAWS for method in POLICIES
    }:
        raise ValueError(f"Incomplete or duplicate {metric} summaries")
    if not table.n_banks.eq(32).all():
        raise ValueError("Expected all 32 registered banks")
    return table


def plot(results):
    summary = pd.read_csv(results / "summary.csv")
    generations = pd.read_csv(results / "generations.csv")
    rates = score_rows(summary, "selected_is_mean")
    excess = score_rows(summary, "audit_excess")
    curves = generations.query(
        "allocation == 'reuse' and law == 'equal' and metric == 'mean_equiv_present'"
    )
    figure, (population, loss) = plt.subplots(
        1, 2, figsize=(11.4, 5.6), gridspec_kw={"width_ratios": [1.45, 1]}
    )
    figure.subplots_adjust(left=0.083, right=0.975, top=0.675, bottom=0.20, wspace=0.30)
    figure.text(
        0.03,
        0.935,
        "Seeding and retaining the arithmetic mean in native GP",
        fontsize=17,
        fontweight="semibold",
        color="#192634",
    )
    figure.text(
        0.03,
        0.870,
        "Checkpoint selection and fusion fitting reuse the same scenes. Every deployed tree is evaluated on fresh scenes.",
        fontsize=10.25,
        color="#425262",
    )

    handles = []
    for method, offset in zip(POLICIES, GROUP_OFFSETS, strict=True):
        label, color, marker = POLICIES[method]
        series = curves.loc[curves.method.eq(method)].sort_values("generation")
        if (
            series.generation.tolist() != list(range(17))
            or not series.n_banks.eq(32).all()
        ):
            raise ValueError(f"Incomplete generations for {method}")
        values = series.estimate.to_numpy()
        if not np.isfinite(values).all() or min(values) < 0 or max(values) > 1:
            raise ValueError(f"Invalid mean-presence frequencies for {method}")
        # Final selection is a separate decision; leave a break after generation 16.
        population.plot(
            series.generation,
            values,
            color=color,
            linewidth=1.8,
            marker=marker,
            markersize=4,
            markevery=(1 if method == "carry" else 0, 2),
        )
        population.plot(
            19.6 + 2.2 * offset,
            rates.loc[("equal", method), "estimate"],
            marker=marker,
            markersize=6.8,
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.7,
        )
        for index, law in enumerate(LAWS):
            row = excess.loc[(law, method)]
            estimate, low, high = row[["estimate", "ci_low", "ci_high"]]
            if (
                not np.isfinite([estimate, low, high]).all()
                or min(estimate, low) < 0
                or high < low
            ):
                raise ValueError(f"Invalid conditional excess for {law}, {method}")
            x = index + offset
            loss.vlines(x, low, high, color=color, linewidth=1.6)
            loss.plot([x, x], [low, high], "_", markersize=7.5, color=color)
            loss.plot(
                x,
                estimate,
                marker=marker,
                markersize=6.8,
                color=color,
                markeredgecolor="white",
                markeredgewidth=0.7,
            )
        handles.append(
            Line2D(
                [],
                [],
                color=color,
                marker=marker,
                linewidth=1.8,
                markersize=5.4,
                label=label,
            )
        )

    figure.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.027, 0.836),
        ncol=3,
        frameon=False,
        columnspacing=2.5,
        fontsize=10,
    )
    population.set_title(
        "Mean present or deployed",
        loc="left",
        fontsize=13,
        fontweight="semibold",
        pad=21,
    )
    population.text(
        0,
        1.035,
        "Equal-competence law · share of search runs",
        transform=population.transAxes,
        fontsize=9.5,
        color="#425262",
    )
    population.set_xlim(-0.6, 21.1)
    population.set_ylim(-0.055, 1.055)
    population.set_yticks(np.linspace(0, 1, 5))
    population.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    population.set_xticks(
        [0, 4, 8, 12, 16, 19.6], ["0", "4", "8", "12", "16", "Deployed"]
    )
    population.axvline(17.5, color="#bcc6cc", linestyle=":", linewidth=1.2)
    population.set_xlabel("Generation", labelpad=8)
    loss.set_title(
        "Conditional excess Brier",
        loc="left",
        fontsize=13,
        fontweight="semibold",
        pad=21,
    )
    loss.text(
        0,
        1.035,
        "Fresh-scene loss · lower is better",
        transform=loss.transAxes,
        fontsize=9.5,
        color="#425262",
    )
    loss.set_xlim(-0.48, 1.48)
    loss.set_xticks(range(len(LAWS)), list(LAWS.values()))
    maximum = max(1e-12, excess[["estimate", "ci_high"]].to_numpy().max())
    loss.set_ylim(-0.035 * maximum, 1.12 * maximum)
    loss.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
    for ax in (population, loss):
        ax.set_axisbelow(True)
        ax.grid(axis="y", linewidth=0.65, color="#e0e5e9")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#afbac1")
        ax.tick_params(axis="y", length=0, labelsize=9, colors="#425262")
        ax.tick_params(axis="x", length=3, labelsize=9.5, colors="#334250")
    figure.text(
        0.03,
        0.072,
        "Mean presence requires matching probabilities on every fit and audit row. Whiskers show 95% bootstrap intervals.",
        fontsize=9.3,
        color="#425262",
    )
    figure.text(
        0.03,
        0.034,
        "Both checkpoint/fit swaps are averaged within each of the 32 independent banks.",
        fontsize=9.3,
        color="#425262",
    )
    figure.savefig(results / "seeded-transfer.png", dpi=180, facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", type=Path, default=Path(__file__).resolve().parent / "results"
    )
    plot(parser.parse_args().results)

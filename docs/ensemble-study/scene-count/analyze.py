"""Summarize paired scene-count banks and render the figure from archived CSVs."""

import argparse
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

SEEDS = range(286100, 286132)
LAWS = ("equal", "signal")
DESIGNS = ((24, 8), (192, 1), (192, 8), (1536, 1))
METHODS = ("seeded", "mean", "top_n", "prob_lr", "regularized")
LAYOUT = ["n_scenes", "rows_per_scene"]
SCORE_METRICS = [
    "fit_brier",
    "audit_brier",
    "audit_excess",
    "fit_f1",
    "audit_f1",
    "n_providers",
]
TRAJECTORY_METRICS = [
    "fit_selected",
    "audit_selected",
    "audit_selected_excess",
    "fit_f1",
    "audit_f1",
]
COMPARISONS = {
    "clustered_minus_independent_192": ((24, 8), (192, 1)),
    "clustered_minus_independent_1536": ((192, 8), (1536, 1)),
    "large_minus_small_clustered": ((192, 8), (24, 8)),
    "large_minus_small_independent": ((1536, 1), (192, 1)),
    "extra_rows_same_192_scenes": ((192, 8), (192, 1)),
}


def validate(frame, *, trajectory=False):
    keys = ["seed", "law", *LAYOUT, "method"]
    expected = [
        (s, law, n, r, m)
        for s in SEEDS
        for law in LAWS
        for n, r in DESIGNS
        for m in (("seeded",) if trajectory else METHODS)
    ]
    if trajectory:
        keys.append("generation")
        expected = [(*row, g) for row in expected for g in range(17)]
    index = pd.MultiIndex.from_frame(frame[keys])
    if not index.is_unique or set(index) != set(expected):
        raise ValueError(f"Missing, extra or duplicate bank keys in {keys}")
    if not frame.n_rows.eq(frame.n_scenes * frame.rows_per_scene).all():
        raise ValueError("Inconsistent number of observations")
    metrics = TRAJECTORY_METRICS if trajectory else SCORE_METRICS
    if not np.isfinite(frame[metrics]).all().all():
        raise ValueError(f"Nonfinite value in {metrics}")
    return frame.sort_values(keys)


def interval(values, draws, *, signs=False):
    x = values.to_numpy(dtype=float)
    if len(x) != len(SEEDS) or not np.isfinite(x).all():
        raise ValueError("Every contrast must have 32 finite paired bank values")
    lo, hi = np.quantile(x[draws].mean(axis=1), [0.025, 0.975])
    out = {"n_banks": len(x), "estimate": x.mean(), "ci_low": lo, "ci_high": hi}
    if signs:
        out.update(
            n_lower=int((x < -1e-12).sum()),
            n_tied=int((np.abs(x) <= 1e-12).sum()),
            n_higher=int((x > 1e-12).sum()),
        )
    return out


def summarize_scores(scores, draws):
    levels, contrasts, controls = [], [], []
    primary = None
    for (law, method), group in scores.groupby(["law", "method"]):
        cells = {
            pair: rows.set_index("seed").reindex(SEEDS)
            for pair, rows in group.groupby(LAYOUT)
        }
        targets = ["mean", "signal"] if law == "signal" else ["mean"]
        for (n, r), cell in cells.items():
            for target in targets:
                field = "selected_is_" + target
                flags = pd.Series(as_bool(cell[field]).astype(float), index=cell.index)
                levels.append(
                    {
                        "law": law,
                        "method": method,
                        "n_scenes": n,
                        "rows_per_scene": r,
                        "n_rows": n * r,
                        "metric": field,
                        **interval(flags, draws),
                    }
                )
        for metric in SCORE_METRICS:
            for (n, r), cell in cells.items():
                levels.append(
                    dict(
                        law=law,
                        method=method,
                        n_scenes=n,
                        rows_per_scene=r,
                        n_rows=n * r,
                        metric=metric,
                        **interval(cell[metric], draws),
                    )
                )
            for name, (a, b) in COMPARISONS.items():
                diff = cells[a][metric] - cells[b][metric]
                contrasts.append(
                    dict(
                        law=law,
                        method=method,
                        metric=metric,
                        contrast=name,
                        **interval(diff, draws, signs=True),
                    )
                )
                if (law, method, metric, name) == (
                    "signal",
                    "seeded",
                    "audit_excess",
                    "clustered_minus_independent_192",
                ):
                    primary = pd.DataFrame(
                        {
                            "clustered_excess": cells[a][metric],
                            "independent_excess": cells[b][metric],
                            "difference": diff,
                        }
                    )
    for (law, n, r), group in scores.groupby(["law", *LAYOUT]):
        for metric in SCORE_METRICS:
            table = group.pivot(index="seed", columns="method", values=metric).reindex(
                SEEDS
            )
            for ref in METHODS[1:]:
                controls.append(
                    dict(
                        law=law,
                        n_scenes=n,
                        rows_per_scene=r,
                        n_rows=n * r,
                        method="seeded",
                        reference=ref,
                        metric=metric,
                        **interval(table.seeded - table[ref], draws, signs=True),
                    )
                )
    return (
        pd.DataFrame(levels),
        pd.DataFrame(contrasts),
        pd.DataFrame(controls),
        primary.reset_index(),
    )


def as_bool(series):
    """Handle native boolean columns and booleans mixed with missing CSV cells."""
    values = series.map({True: True, False: False, "True": True, "False": False})
    if values.isna().any():
        raise ValueError("Missing selection flag where a target function is defined")
    return values.to_numpy(dtype=bool)


def summarize_search(trajectory, draws):
    stages, changes, selections = [], [], []
    for (law, n, r), group in trajectory.groupby(["law", *LAYOUT]):
        keys = {"law": law, "n_scenes": n, "rows_per_scene": r, "n_rows": n * r}
        start = group[group.generation == 0].set_index("seed").reindex(SEEDS)
        stop = group[group.generation == 16].set_index("seed").reindex(SEEDS)
        for name in TRAJECTORY_METRICS:
            changes.append(
                dict(
                    **keys,
                    metric=name,
                    contrast="final_minus_initial",
                    **interval(stop[name] - start[name], draws, signs=True),
                )
            )
        for gen, rows in group.groupby("generation"):
            cell = rows.set_index("seed").reindex(SEEDS)
            for name in TRAJECTORY_METRICS:
                stages.append(
                    dict(
                        **keys,
                        generation=gen,
                        metric=name,
                        **interval(cell[name], draws),
                    )
                )
            for target in ("mean", "signal") if law == "signal" else ("mean",):
                available = cell[f"n_active_{target}_equiv"].to_numpy() > 0
                chosen = as_bool(cell[f"selected_{target}_equiv"])
                if (chosen & ~available).any():
                    raise ValueError(
                        "Selected winner absent from the recorded survivors"
                    )
                passed = available & ~chosen
                event = dict(
                    **keys,
                    generation=gen,
                    target=target,
                    n_banks=len(cell),
                    available_count=int(available.sum()),
                    chosen_count=int(chosen.sum()),
                    passed_count=int(passed.sum()),
                    unavailable_count=int((~available).sum()),
                    available_fraction=available.mean(),
                    chosen_fraction=chosen.mean(),
                    passed_fraction=passed.mean(),
                    chosen_given_available=chosen.sum() / available.sum()
                    if available.any()
                    else np.nan,
                )
                if target == "signal":
                    # Here q is supplied by a known provider, so excess is its exact per-bank cost gap.
                    risk = cell.audit_selected_excess.to_numpy()
                    if np.any(risk[chosen] > 1e-12):
                        raise ValueError(
                            "Exact-signal winner must have zero conditional excess"
                        )
                    event["missed_cost_given_passed"] = (
                        risk[passed].mean() if passed.any() else np.nan
                    )
                    event["missed_cost_contribution"] = (risk * passed).mean()
                    event["unavailable_cost_contribution"] = (risk * ~available).mean()
                    if not np.isclose(
                        event["missed_cost_contribution"]
                        + event["unavailable_cost_contribution"],
                        risk.mean(),
                        atol=1e-12,
                        rtol=0,
                    ):
                        raise ValueError("Deployment cost decomposition failed")
                selections.append(event)
    return pd.DataFrame(stages), pd.DataFrame(changes), pd.DataFrame(selections)


def make_figure(final_scores, selections, destination):
    """Plot seed-paired loss and functional availability/selection on the signal law."""
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#a6b4b6",
            "xtick.color": "#45545b",
            "ytick.color": "#45545b",
            "legend.frameon": False,
        }
    )
    colors = {8: "#b04b2c", 1: "#176f79"}
    fig, axes = plt.subplots(
        1, 3, figsize=(10.4, 3.66), gridspec_kw={"width_ratios": [1.34, 1, 1]}
    )
    fig.subplots_adjust(left=0.073, right=0.983, top=0.745, bottom=0.233, wspace=0.41)
    ax = axes[0]
    data = final_scores.query("law == 'signal' and method == 'seeded'")
    groups = {
        pair: group.set_index("seed").reindex(SEEDS).audit_excess
        for pair, group in data.groupby(LAYOUT)
    }
    for offset, (a, b) in zip(
        (0, 2.75), [(DESIGNS[0], DESIGNS[1]), (DESIGNS[2], DESIGNS[3])], strict=True
    ):
        positions = np.array([offset, offset + 1])
        left, right = groups[a], groups[b]
        for va, vb in zip(left, right, strict=True):
            ax.plot(positions, [va, vb], color="#a6b2b4", lw=0.6, alpha=0.55, zorder=1)
        ax.plot(
            positions, [left.mean(), right.mean()], color="#263c42", lw=1.7, zorder=3
        )
        for x, pair in zip(positions, (a, b), strict=True):
            values = groups[pair]
            ax.scatter(
                [x] * len(values),
                values,
                s=7,
                color=colors[pair[1]],
                alpha=0.45,
                linewidths=0,
                zorder=2,
            )
            ax.plot(
                x,
                values.mean(),
                marker="D",
                mfc=colors[pair[1]],
                mec="white",
                mew=0.75,
                ms=7,
                zorder=4,
            )
    ax.set(
        xlim=(-0.32, 4.07),
        xticks=[0, 1, 2.75, 3.75],
        xticklabels=["24×8", "192×1", "192×8", "1,536×1"],
        ylabel="Conditional excess Brier",
    )
    ax.set_ylim(bottom=-0.001)
    ax.tick_params(axis="x", length=0)
    ax.text(
        0.19,
        -0.205,
        "192 observations",
        transform=ax.transAxes,
        ha="center",
        fontsize=8,
    )
    ax.text(
        0.83,
        -0.205,
        "1,536 observations",
        transform=ax.transAxes,
        ha="center",
        fontsize=8,
    )
    ax.set_title(
        "A   Final loss on fresh scenes",
        loc="left",
        fontweight="semibold",
        fontsize=9.5,
        pad=10,
    )
    ax.grid(axis="y", color="#E1E7E7", lw=0.6, zorder=0)
    sig = selections.query("law == 'signal' and target == 'signal'")
    for i, (ax, total, letter) in enumerate(
        zip(axes[1:], (192, 1536), ("B", "C"), strict=True)
    ):
        for n, r in [pair for pair in DESIGNS if np.prod(pair) == total]:
            rows = sig[(sig.n_scenes == n) & (sig.rows_per_scene == r)].sort_values(
                "generation"
            )
            for name, style in [("available_fraction", "--"), ("chosen_fraction", "-")]:
                ax.plot(
                    rows.generation,
                    rows[name],
                    style,
                    color=colors[r],
                    lw=1.8,
                    marker="o" if style == "-" else None,
                    ms=2.15,
                )
        ax.set(
            xlim=(0, 16),
            ylim=(-0.035, 1.035),
            xticks=[0, 4, 8, 12, 16],
            yticks=[0, 0.25, 0.5, 0.75, 1],
            xlabel="Generation",
        )
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        ax.set_title(
            f"{letter}   {total:,} fit observations",
            loc="left",
            fontweight="semibold",
            fontsize=9.5,
            pad=10,
        )
        ax.grid(axis="y", color="#E1E7E7", lw=0.6)
        if i == 0:
            ax.set_ylabel("Banks with exact provider")
    handles = [
        Line2D([], [], color=colors[8], lw=2, label="8 rows per scene"),
        Line2D([], [], color=colors[1], lw=2, label="1 row per scene"),
        Line2D(
            [], [], color="#42535b", linestyle="--", label="Available among survivors"
        ),
        Line2D(
            [],
            [],
            color="#42535b",
            lw=1.8,
            marker="o",
            ms=3,
            label="Chosen by fit Brier",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.51, 0.96),
        ncols=4,
        fontsize=8,
    )
    fig.savefig(
        destination / "scene-count.pdf",
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(destination / "scene-count.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", type=Path, default=Path(__file__).resolve().parent / "results"
    )
    p = parser.parse_args().results
    scores = validate(pd.read_csv(p / "scores.csv"))
    trajectory = validate(pd.read_csv(p / "trajectories.csv.gz"), trajectory=True)
    draws = np.random.default_rng(97823).integers(len(SEEDS), size=(10_000, len(SEEDS)))
    levels, comparisons, controls, primary = summarize_scores(scores, draws)
    stages, changes, selection = summarize_search(trajectory, draws)
    for filename, data in [
        ("summary", levels),
        ("design-contrasts", comparisons),
        ("control-contrasts", controls),
        ("primary-by-bank", primary),
        ("generation-performance", stages),
        ("generation-changes", changes),
        ("selection-events", selection),
    ]:
        data.to_csv(p / (filename + ".csv"), index=False)
    make_figure(scores, selection, p)
    print(
        comparisons.query(
            "law=='signal' and method=='seeded' and metric=='audit_excess' "
            "and contrast=='clustered_minus_independent_192'"
        ).to_string(index=False)
    )


if __name__ == "__main__":
    main()

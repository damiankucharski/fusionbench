"""Fixed-bank risk curves, paired contrasts, audit costs, and a figure from archived selections."""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import t

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def interval(series):
    v = np.asarray(series, dtype=float)
    point = v.mean()
    half = t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return {"mean": point, "lo": point - half, "hi": point + half, "banks": len(v)}


def pooled(frame, group, metrics):
    estimates = []
    banked = frame.groupby([*group, "data_seed"], as_index=False)[metrics].mean()
    for key, rows in banked.groupby(group):
        key = key if isinstance(key, tuple) else (key,)
        for metric in metrics:
            clean = rows[metric].dropna()
            if len(clean) > 1:
                estimates.append(
                    {
                        **dict(zip(group, key, strict=True)),
                        "metric": metric,
                        **interval(clean),
                    }
                )
    return pd.DataFrame(estimates)


def results(folder):
    s = pd.read_csv(folder / "selections.csv.gz")
    risk = pd.read_csv(folder / "risks.csv.gz")
    a = pd.read_csv(folder / "audit.csv.gz")
    score_cols = [
        "d",
        "slope",
        "estimated_p",
        "different_program",
        "strict_rank_reversal",
        "transport_brier",
        "validation_brier",
        "known_p_brier",
        "oracle_brier",
        "transport_regret",
        "validation_regret",
    ]
    out = pooled(s, ["target_p", "lambda"], score_cols)
    out.to_csv(folder / "selection-intervals.csv", index=False)
    baseline = risk[risk.method.isin(["mean", "top_n", "stack_logit"])]
    pooled(baseline, ["target_p", "lambda", "method"], ["brier"]).to_csv(
        folder / "baselines.csv", index=False
    )
    audit_cols = [
        "test_brier",
        "test_regret",
        "mse",
        "rmse",
        "true_estimator_variance",
        "mean_unique_labels",
        "pool_sign_error",
        "independent_sign_error",
        "cp_excludes_zero",
        "cp_contains_pool",
        "det_certified",
    ]
    summary = pooled(a, ["target_p", "lambda", "budget", "proposal"], audit_cols)
    summary.to_csv(folder / "audit-intervals.csv", index=False)
    paired = a[a.proposal.isin(["gap", "uniform"])].pivot(
        index=["data_seed", "repeat", "target_p", "lambda", "budget"],
        columns="proposal",
        values=[
            "test_regret",
            "mse",
            "true_estimator_variance",
            "pool_sign_error",
            "mean_unique_labels",
        ],
    )
    diff = paired.xs("gap", level=1, axis=1) - paired.xs("uniform", level=1, axis=1)
    auditdiff = pooled(
        diff.reset_index(), ["target_p", "lambda", "budget"], list(diff.columns)
    )
    auditdiff.to_csv(folder / "audit-paired.csv", index=False)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.4), layout="constrained")
    left = axs[0]
    for p, color in [(0.1, "#42765c"), (0.5, "#173e87")]:
        q = out[(out.metric == "d") & (out.target_p == p)]
        left.plot(
            q["lambda"],
            q["mean"],
            label=f"Changed context: {p:.0%}",
            color=color,
            marker="o",
        )
        left.fill_between(q["lambda"], q.lo, q.hi, color=color, alpha=0.13)
    left.axhline(0, color="#424242", linewidth=0.8)
    left.set(
        title="Mixture selection versus ordinary validation",
        xlabel="Fraction of labels flipped in the changed context",
        ylabel="Transport minus validation Brier",
    )
    left.legend(frameon=False, fontsize=8)
    right = axs[1]
    for policy, marker in [("gap", "o"), ("uniform", "s")]:
        for lam, color in [(0, "#42765c"), (1, "#9d492a")]:
            q = summary[
                (summary.proposal == policy)
                & (summary.target_p == 0.5)
                & (summary.metric == "test_regret")
                & (summary["lambda"] == lam)
            ]
            right.plot(
                q.budget,
                q["mean"],
                marker=marker,
                linestyle="-" if policy == "gap" else "--",
                color=color,
                label=f"{policy.capitalize()}, flip {lam:.0%}",
            )
    right.set(
        title="Target-label audit of the two fixed choices",
        xlabel="Draws with replacement",
        ylabel="Test Brier above the better of the pair",
        xscale="log",
    )
    right.set_xticks([8, 32, 128], ["8", "32", "128"])
    right.legend(frameon=False, fontsize=8)
    for ax in axs:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.2)
    fig.savefig(folder / "ranking-shift.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    cli = ArgumentParser(description=__doc__)
    cli.add_argument("folder", type=Path)
    results(cli.parse_args().folder)

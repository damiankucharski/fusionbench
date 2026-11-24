"""Compare the published predictions with external outcomes and summarize ablations."""

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import kendalltau

CORE = ["native", "mean", "top_n", "prob_lr", "logit"]


def direction(gap):
    if abs(gap) <= 0.001:
        return "tie"
    return "native_lower" if gap < 0 else "prob_lr_lower"


def run(root):
    synthetic = pd.read_csv(root / "synthetic-results/scores.csv")
    external = pd.read_csv(root / "external-results/scores.csv")
    predictions = pd.read_csv(root / "predictions/forecasts.csv")
    external_means = (
        external.pivot(index=["suite", "seed"], columns="method", values="brier")
        .groupby("suite")
        .mean()
    )

    checks = []
    for record in predictions.to_dict("records"):
        actual = external_means.loc[record["suite"], CORE]
        gap = actual["native"] - actual["prob_lr"]
        ranks = actual.rank()
        tau = kendalltau([record["rank_" + m] for m in CORE], ranks[CORE]).statistic
        checks.append(
            {
                "suite": record["suite"],
                "library": record["library"],
                "config": record["config"],
                "distance": record["distance"],
                "predicted_gap": record["native_minus_prob_lr"],
                "observed_gap": gap,
                "absolute_gap_error": abs(record["native_minus_prob_lr"] - gap),
                "predicted_direction": record["direction"],
                "observed_direction": direction(gap),
                "matches_direction": direction(gap) == record["direction"],
                "kendall_tau_b": tau,
            }
        )
    comparisons = pd.DataFrame(checks)
    assert len(comparisons) == 6
    comparisons.to_csv(root / "forecast-evaluation.csv", index=False)

    syn = synthetic.pivot(
        index=["stratum", "replicate", "config"], columns="method", values="brier"
    )
    syn["native_minus_prob_lr"] = syn.native - syn.prob_lr
    pairs = {
        "soft_finite_votes": ("panel_D", "panel_A"),
        "strong_finite_votes": ("panel_B", "panel_C"),
        "directed_confusion": ("panel_E", "panel_B"),
    }
    rows = []
    for label, (changed, base) in pairs.items():
        delta = syn.xs(changed, level="config") - syn.xs(base, level="config")
        for stratum, group in delta.groupby("stratum"):
            for col in [*CORE, "symmetric_lr", "pooled_lr", "native_minus_prob_lr"]:
                rows.append(
                    {
                        "intervention": label,
                        "stratum": stratum,
                        "endpoint": col,
                        "change": group[col].mean(),
                        "negative_replicates": (group[col] < 0).sum(),
                        "replicates": len(group),
                    }
                )
    pd.DataFrame(rows).to_csv(root / "paired-interventions.csv", index=False)
    external_means.to_csv(root / "observed-brier.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    run(parser.parse_args().root)

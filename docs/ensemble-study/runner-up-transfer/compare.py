"""Score the frozen external forecasts and the prespecified promotion rule."""

import argparse
import json
from pathlib import Path

import pandas as pd
from scipy.stats import kendalltau

METHODS = ("native", "mean", "top_n", "prob_lr", "logit")
TARGETS = {"digits": 3300000, "iris": 3310000, "seeds": 3320000}
PROFILES = ("base", "margin_only", "augmented")


def direction(gap):
    if abs(gap) <= 0.001:
        return "tie"
    return "native" if gap < 0 else "prob_lr"


def evaluate(root):
    forecasts = pd.read_csv(root / "predictions/method_losses.csv")
    observed = pd.read_csv(root / "external-results/scores.csv")
    predictions = forecasts.pivot(
        index=["suite", "profile"], columns="method", values="mean_brier"
    )
    scores = observed.pivot(index=["suite", "seed"], columns="method", values="brier")
    expected = {
        (suite, start + i) for suite, start in TARGETS.items() for i in range(8)
    }
    if set(scores.index) != expected or scores[list(METHODS)].isna().any().any():
        raise ValueError("Expected all paired test losses for the 24 registered splits")
    rows = []
    means = scores.groupby("suite").mean()
    for suite in TARGETS:
        actual = means.loc[suite, list(METHODS)]
        gap = actual.native - actual.prob_lr
        per_seed = scores.loc[suite]
        for profile in PROFILES:
            pred = predictions.loc[(suite, profile), list(METHODS)]
            predicted_gap = pred.native - pred.prob_lr
            rows.append(
                {
                    "suite": suite,
                    "profile": profile,
                    "predicted_gap": predicted_gap,
                    "observed_gap": gap,
                    "absolute_error": abs(predicted_gap - gap),
                    "predicted_direction": direction(predicted_gap),
                    "observed_direction": direction(gap),
                    "direction_correct": direction(gap) == direction(predicted_gap),
                    "kendall_tau_b": kendalltau(pred, actual).statistic,
                    "native_lower_splits": int(
                        (per_seed.native < per_seed.prob_lr).sum()
                    ),
                }
            )
    checks = pd.DataFrame(rows)
    if checks[["predicted_gap", "observed_gap", "kendall_tau_b"]].isna().any().any():
        raise ValueError("Incomplete forecasts or undefined rank correlation")
    checks.to_csv(root / "evaluation.csv", index=False)
    means.to_csv(root / "observed-brier.csv")
    errors = checks.pivot(index="suite", columns="profile", values="absolute_error")
    changes = errors.augmented - errors.base
    passed = changes.mean() < 0 and (changes < 0).sum() >= 2
    summary = {
        "mean_absolute_error": errors.mean().to_dict(),
        "improved_targets": int((changes < 0).sum()),
        "total_targets": len(TARGETS),
        "improvement_criterion_passed": bool(passed),
    }
    (root / "decision.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    evaluate(parser.parse_args().root)

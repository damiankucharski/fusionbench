"""Summarize the preregistered block swaps and bootstrap whole banks jointly."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SEEDS = tuple(range(284100, 284132))
METHODS = ("mean", "top_n", "prob_lr", "gp_brier", "gp_f1", "regularized")
BOOTSTRAPS = 10_000
BOOTSTRAP_SEED = 97821
TOLERANCE = 1e-12
KEYS = ["method", "k", "metric", "comparison"]


def validate_cells(frame, dimensions):
    """Require one complete Cartesian product, with each expected key exactly once."""
    keys = list(dimensions)
    expected = pd.MultiIndex.from_product(dimensions.values(), names=keys)
    actual = pd.MultiIndex.from_frame(frame[keys])
    if (
        not actual.is_unique
        or len(expected.difference(actual))
        or len(actual.difference(expected))
    ):
        raise ValueError(f"Missing, unexpected or duplicate cells for {keys}")


def record(frame, comparison, keys):
    return (
        frame.melt(ignore_index=False, var_name="metric", value_name="value")
        .reset_index()[[*keys, "metric", "value"]]
        .assign(comparison=comparison)
    )


def check_zero(series, name):
    if not np.allclose(series, 0.0, rtol=0.0, atol=TOLERANCE):
        raise ValueError(
            f"{name} failed (maximum absolute difference {series.abs().max():.3g})"
        )


def fusion_rows(scores):
    scores = scores.copy()
    for field in ("checkpoint_block", "fit_block"):
        scores[field] = scores[field].str.upper()
    validate_cells(
        scores,
        {
            "seed": SEEDS,
            "k": [1, 8],
            "method": METHODS,
            "checkpoint_block": ["A", "B"],
            "fit_block": ["A", "B"],
        },
    )
    scores = scores.rename(
        columns={"fit_selected": "fit_brier", "audit_selected": "audit_brier"}
    )
    base = ["fit_brier", "audit_brier", "fit_f1", "audit_f1"]
    if not np.isfinite(scores[base]).all().all():
        raise ValueError("Nonfinite deployed scores")
    scores["optimism_brier"] = scores.audit_brier - scores.fit_brier
    scores["optimism_f1"] = scores.fit_f1 - scores.audit_f1
    metrics = [*base, "optimism_brier", "optimism_f1"]

    # Optional oracle values describe the same fitted trajectory in a second metric.
    for suffix, metric, direction in (("", "brier", 1), ("_f1", "f1", -1)):
        final, seen = f"audit_best_final{suffix}", f"audit_best_seen{suffix}"
        if final not in scores and seen not in scores:
            continue
        levels = [f"audit_final_best_{metric}", f"audit_seen_best_{metric}"]
        if final not in scores or seen not in scores:
            raise ValueError(f"Incomplete archived hindsight metrics for {metric}")
        scores[levels] = scores[[final, seen]]
        finite = scores[levels].notna()
        if not finite.iloc[:, 0].equals(finite.iloc[:, 1]):
            raise ValueError(f"Mismatched hindsight coverage for {metric}")
        counts = finite.iloc[:, 0].groupby(scores.method).agg(["sum", "size"])
        if ((counts["sum"] != 0) & (counts["sum"] != counts["size"])).any():
            raise ValueError(
                f"Incomplete hindsight coverage within a method for {metric}"
            )
        gaps = [f"selected_to_final_gap_{metric}", f"final_to_seen_gap_{metric}"]
        scores[gaps[0]] = direction * (scores[f"audit_{metric}"] - scores[levels[0]])
        scores[gaps[1]] = direction * (scores[levels[0]] - scores[levels[1]])
        if (scores[gaps] < -TOLERANCE).any().any():
            raise ValueError(
                f"Best-seen, best-final and deployed results out of order for {metric}"
            )
        metrics.extend([*levels, *gaps])

    keys = ["seed", "method", "k"]
    reuse = scores.checkpoint_block == scores.fit_block
    matched = scores.loc[reuse].groupby(keys)[metrics].mean()
    disjoint = scores.loc[~reuse].groupby(keys)[metrics].mean()
    difference = matched - disjoint
    check_zero(difference.xs(1, level="k").stack(), "K=1 swap symmetry")
    check_zero(
        difference.xs("mean", level="method")[["audit_brier", "audit_f1"]].stack(),
        "Arithmetic mean audit invariance",
    )
    return pd.concat(
        [
            record(matched, "reuse", keys),
            record(disjoint, "disjoint", keys),
            record(difference, "reuse_minus_disjoint", keys),
        ],
        ignore_index=True,
    ).dropna(subset="value")


def provider_rows(checkpoints):
    checkpoints = checkpoints.copy()
    checkpoints["checkpoint_block"] = checkpoints.checkpoint_block.str.upper()
    providers = checkpoints.provider.unique()
    if len(providers) != 8:
        raise ValueError("Expected eight providers")
    validate_cells(
        checkpoints,
        {
            "seed": SEEDS,
            "k": [1, 8],
            "checkpoint_block": ["A", "B"],
            "provider": providers,
        },
    )
    if not ((checkpoints.slot >= 0) & (checkpoints.slot < checkpoints.k)).all():
        raise ValueError("Selected checkpoint outside the released prefix")
    metrics = ["selection_f1", "opposite_f1", "audit_f1"]
    if not np.isfinite(checkpoints[metrics]).all().all():
        raise ValueError("Nonfinite checkpoint scores")
    means = checkpoints.groupby(["seed", "k"])[metrics].mean()
    means["selection_minus_opposite_f1"] = means.selection_f1 - means.opposite_f1
    means["selection_minus_audit_f1"] = means.selection_f1 - means.audit_f1
    check_zero(
        means.xs(1, level="k").selection_minus_opposite_f1, "K=1 provider swap symmetry"
    )
    gain = means.xs(8, level="k") - means.xs(1, level="k")
    gain = gain.assign(k=8).set_index("k", append=True)
    return pd.concat(
        [
            record(means, "level", ["seed", "k"]),
            record(gain, "k8_minus_k1", ["seed", "k"]),
        ],
        ignore_index=True,
    ).assign(method="provider_checkpoints")


def summarize(rows):
    # The same resampled seed indices are used for every estimand and every output.
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(
        len(SEEDS), size=(BOOTSTRAPS, len(SEEDS))
    )
    data = rows.pivot(index="seed", columns=KEYS, values="value").reindex(SEEDS)
    if not np.isfinite(data).all().all():
        raise ValueError("An estimand is missing a bank or has a nonfinite value")
    output = []
    for key, column in data.items():
        values = column.to_numpy()
        low, high = np.quantile(values[draws].mean(axis=1), [0.025, 0.975])
        output.append(
            {
                **dict(zip(KEYS, key)),
                "n_banks": len(values),
                "estimate": values.mean(),
                "ci_low": low,
                "ci_high": high,
            }
        )
    return pd.DataFrame(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", type=Path, default=Path(__file__).resolve().parent / "results"
    )
    args = parser.parse_args()
    fusion = fusion_rows(pd.read_csv(args.results / "scores.csv"))
    providers = provider_rows(pd.read_csv(args.results / "checkpoints.csv"))
    rows = pd.concat([fusion, providers], ignore_index=True)[["seed", *KEYS, "value"]]
    rows = rows.sort_values(["seed", *KEYS])
    summary = summarize(rows)
    rows.to_csv(args.results / "bank-contrasts.csv", index=False)
    summary.to_csv(args.results / "summary.csv", index=False)
    primary = summary.query(
        "method == 'gp_brier' and k == 8 and metric == 'audit_brier'"
        " and comparison == 'reuse_minus_disjoint'"
    ).iloc[0]
    print(
        f"Primary paired audit Brier difference {primary.estimate:+.6f} "
        f"(95% interval {primary.ci_low:+.6f}, {primary.ci_high:+.6f})"
    )


if __name__ == "__main__":
    main()

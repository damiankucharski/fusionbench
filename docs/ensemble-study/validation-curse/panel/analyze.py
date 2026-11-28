"""Paired, scenario-balanced summaries of the predeclared validation-curse panels."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def compute_interval(delta):
    """Resample banks within each fixed scenario; retain pairing between arms."""
    matrix = delta.unstack("rep").sort_index().to_numpy()
    n_scenarios, n_reps = matrix.shape
    if matrix.shape != (24, 6) or not np.isfinite(matrix).all():
        raise ValueError(
            "Expected the complete standard-v3 panel of 24 scenarios by six banks"
        )
    rng = np.random.default_rng(328456)
    selectors = rng.integers(n_reps, size=(5000, n_scenarios, n_reps))
    means = matrix[np.arange(n_scenarios)[None, :, None], selectors].mean((1, 2))
    return np.quantile(means, [0.025, 0.975])


def analyze(directory):
    f = pd.read_csv(directory / "scores.csv", float_precision="round_trip")
    matrices = f.pivot(
        index=["scenario", "rep"], columns=["stage", "domain", "method"], values="brier"
    )
    data = {}
    for label in ["frozen", "guarded_frozen"]:
        if label not in f.stage.unique():
            continue
        validation = (
            matrices[label, "internal_validation", "mean"]
            - matrices[label, "internal_validation", "internal_winner"]
        )
        audit = (
            matrices[label, "source_audit", "mean"]
            - matrices[label, "source_audit", "internal_winner"]
        )
        data.update(
            {
                f"{label}_validation_gain": validation,
                f"{label}_audit_gain": audit,
                f"{label}_relative_optimism": validation - audit,
                f"{label}_absolute_optimism": matrices[
                    label, "source_audit", "internal_winner"
                ]
                - matrices[label, "internal_validation", "internal_winner"],
            }
        )
    second = "guarded_frozen" in f.stage.unique()
    if second:
        for stat in ("relative_optimism", "absolute_optimism"):
            data["search_guard_minus_incumbent_" + stat] = (
                data["guarded_frozen_" + stat] - data["frozen_" + stat]
            )
        for domain in ("source_audit", "suite_test", "swapped_test"):
            for left, right in (
                ("search_guard_full", "incumbent_full"),
                ("outer_guard", "outer_raw"),
                ("outer_guard", "incumbent_full"),
                ("incumbent_inner", "incumbent_full"),
                ("outer_raw", "incumbent_inner"),
                ("outer_guard", "mean"),
            ):
                data[domain + "_" + left + "_minus_" + right] = (
                    matrices["deployment", domain, left]
                    - matrices["deployment", domain, right]
                )
    b = pd.DataFrame(data)
    b.to_csv(directory / "bank-summary.csv")
    b.groupby("scenario").mean().to_csv(directory / "scenario-summary.csv")
    summaries = []
    for name, v in b.items():
        low, high = compute_interval(v)
        summaries.append(
            {
                "measurement": name,
                "mean": v.mean(),
                "ci025": low,
                "ci975": high,
                "negative_banks": (v < -1e-12).sum(),
                "ties": (v.abs() <= 1e-12).sum(),
                "positive_banks": (v > 1e-12).sum(),
                "negative_scenario_means": (
                    v.groupby("scenario").mean() < -1e-12
                ).sum(),
            }
        )
    pd.DataFrame(summaries).to_csv(directory / "summary.csv", index=False)
    f.groupby(["stage", "domain", "method"])[["brier", "accuracy"]].mean().to_csv(
        directory / "method-means.csv"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    analyze(p.parse_args().directory)

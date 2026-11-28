"""Paired round-one contrasts, survivor diagnostics and fixed-decision design effects."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SEEDS = np.arange(287100, 287132)
SHARES = (0.0, 0.5, 1.0)
LAYOUTS = ((24, 8), (192, 1))
LAWS = ("equal", "signal")
METHODS = ("seeded", "mean", "top_n", "prob_lr", "regularized")
DRAWS = np.random.default_rng(97824).integers(len(SEEDS), size=(10000, len(SEEDS)))
EXPECTED = len(SEEDS) * len(LAWS) * len(SHARES) * len(LAYOUTS)
IDS = ["seed", "law", "scene_error_share", "n_scenes", "rows_per_scene"]


def interval(point, sampled):
    low, high = np.percentile(sampled, [2.5, 97.5])
    return {"estimate": float(point), "ci_low": float(low), "ci_high": float(high)}


def paired_interval(differences):
    return interval(differences.mean(), differences[DRAWS].mean(axis=1))


def design_effect(between, within):
    return 8 * between / (between + (7 / 8) * within)


def analyze(folder):
    scores = pd.read_csv(folder / "scores.csv")
    stages = pd.read_csv(folder / "trajectories.csv.gz")
    components = pd.read_csv(folder / "variance-components.csv")
    if (
        len(scores) != EXPECTED * len(METHODS)
        or scores.duplicated([*IDS, "method"]).any()
        or len(stages) != EXPECTED * 17
        or stages.duplicated([*IDS, "generation"]).any()
        or len(components) != len(SEEDS) * len(LAWS) * len(SHARES)
        or components.duplicated(IDS[:3]).any()
    ):
        raise ValueError(
            "Incomplete or duplicated scores, trajectories or variance components"
        )
    contrasts, summary, primary_banks = [], [], []
    metrics = [
        "fit_brier",
        "audit_brier",
        "audit_excess",
        "fit_f1",
        "audit_f1",
        "n_providers",
    ]
    levels = scores.groupby(
        ["law", "method", "scene_error_share", "n_scenes", "rows_per_scene"], sort=True
    )
    for (law, method, share, g, r), values in levels:
        if not np.array_equal(np.sort(values.seed), SEEDS):
            raise ValueError("Missing bank in score table")
        summary.append(
            {
                "law": law,
                "method": method,
                "scene_error_share": share,
                "n_scenes": g,
                "rows_per_scene": r,
                "banks": len(values),
                **values[metrics].mean().to_dict(),
                "n_selected_mean": int(values.selected_is_mean.eq(True).sum()),
                "n_selected_signal": int(values.selected_is_signal.eq(True).sum())
                if law == "signal"
                else None,
            }
        )
    for law in LAWS:
        for method in METHODS:
            part = scores[(scores.law == law) & (scores.method == method)]
            for metric in ("audit_excess", "audit_brier"):
                wide = part.pivot(
                    index="seed",
                    columns=["scene_error_share", "n_scenes", "rows_per_scene"],
                    values=metric,
                ).loc[SEEDS]
                gaps = {}
                for share in SHARES:
                    differences = (
                        wide[(share, 24, 8)] - wide[(share, 192, 1)]
                    ).to_numpy()
                    gaps[share] = differences
                    contrasts.append(
                        {
                            "law": law,
                            "method": method,
                            "metric": metric,
                            "contrast": "24x8_minus_192x1",
                            "scene_error_share": share,
                            "banks": len(SEEDS),
                            **paired_interval(differences),
                        }
                    )
                delta = gaps[1.0] - gaps[0.0]
                contrasts.append(
                    {
                        "law": law,
                        "method": method,
                        "metric": metric,
                        "contrast": "gap_at_1_minus_gap_at_0",
                        "scene_error_share": None,
                        "banks": len(SEEDS),
                        **paired_interval(delta),
                    }
                )
                if law == "signal" and method == "seeded" and metric == "audit_excess":
                    for i, seed in enumerate(SEEDS):
                        primary_banks.append(
                            {
                                "seed": seed,
                                "gap_share_0": gaps[0.0][i],
                                "gap_share_0_5": gaps[0.5][i],
                                "gap_share_1": gaps[1.0][i],
                                "interaction": delta[i],
                            }
                        )

    # Use all F scenes, pooling estimated variance components across banks before
    # forming the ratio. Every resample uses the same bank indices at all shares.
    variance = []
    for law in LAWS:
        group = components[components.law == law]
        for share in SHARES:
            rows = group[group.scene_error_share == share].set_index("seed").loc[SEEDS]
            within, between = (
                rows[col].to_numpy() for col in ("within", "between_means")
            )
            ratio = design_effect(between.mean(), within.mean())
            draws = design_effect(
                between[DRAWS].mean(axis=1), within[DRAWS].mean(axis=1)
            )
            variance.append(
                {
                    "law": law,
                    "scene_error_share": share,
                    "banks": len(SEEDS),
                    "scenes_per_bank": int(rows.n_pool_scenes.iloc[0]),
                    "within": within.mean(),
                    "between_means": between.mean(),
                    "variance_ratio": ratio,
                    "variance_ratio_ci_low": np.quantile(draws, 0.025),
                    "variance_ratio_ci_high": np.quantile(draws, 0.975),
                    "standard_error_ratio": np.sqrt(ratio),
                    "loss_difference_intraclass_correlation": (ratio - 1) / 7,
                    "icc_ci_low": (np.quantile(draws, 0.025) - 1) / 7,
                    "icc_ci_high": (np.quantile(draws, 0.975) - 1) / 7,
                }
            )

    retention = []
    for (law, share, g, r, generation), sub in stages.groupby(
        ["law", "scene_error_share", "n_scenes", "rows_per_scene", "generation"],
        sort=True,
    ):
        if not np.array_equal(np.sort(sub.seed), SEEDS):
            raise ValueError("Missing bank in trajectory table")
        row = {
            "law": law,
            "scene_error_share": share,
            "n_scenes": g,
            "rows_per_scene": r,
            "generation": generation,
            "banks": len(sub),
        }
        for key in (
            "fit_selected",
            "audit_selected",
            "audit_selected_excess",
            "audit_best_final_excess",
            "audit_best_seen_excess",
        ):
            row[key] = sub[key].mean()
        for suffix in ("mean_exact", "mean_equiv", "signal_exact", "signal_equiv"):
            if suffix.startswith("signal") and law == "equal":
                continue
            available = sub["n_active_" + suffix] > 0
            chosen = sub["selected_" + suffix].eq(True)
            if (chosen & ~available).any():
                raise ValueError(
                    "Selected program absent from the frozen candidate population"
                )
            row["n_available_" + suffix] = int(available.sum())
            row["n_selected_" + suffix] = int(chosen.sum())
            row["n_bypassed_" + suffix] = int((available & ~chosen).sum())
            row["n_absent_" + suffix] = int((~available).sum())
        retention.append(row)

    for name, rows in (
        ("summary", summary),
        ("paired-contrasts", contrasts),
        ("primary-by-bank", primary_banks),
        ("fixed-design-effects", variance),
        ("retention", retention),
    ):
        pd.DataFrame(rows).to_csv(folder / (name + ".csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", type=Path, default=Path(__file__).with_name("results")
    )
    analyze(parser.parse_args().results)

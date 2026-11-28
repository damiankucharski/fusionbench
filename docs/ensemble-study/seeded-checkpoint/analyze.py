"""Paired fresh-bank analysis for the frozen seeded-checkpoint study."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SEEDS = tuple(range(285100, 285132))
LAWS = ("equal", "signal")
METHODS = (
    "stock",
    "sham",
    "seeded",
    "protected",
    "carry",
    "mean",
    "top_n",
    "prob_lr",
    "regularized",
)
SEARCHES = ("stock", "sham", "seeded", "carry")
CELL = ["seed", "law", "checkpoint_block", "fit_block", "method"]
TOL = 1e-12


def check_grid(frame, dimensions):
    keys = list(dimensions)
    expected = pd.MultiIndex.from_product(dimensions.values(), names=keys)
    actual = pd.MultiIndex.from_frame(frame[keys])
    if (
        not actual.is_unique
        or len(expected.difference(actual))
        or len(actual.difference(expected))
    ):
        raise ValueError(f"Incomplete or duplicated study design for {keys}")


def prepare(frame, *, trajectories=False):
    frame = frame.copy()
    for key in ("checkpoint_block", "fit_block"):
        frame[key] = frame[key].str.upper()
    grid = {
        "seed": SEEDS,
        "law": LAWS,
        "checkpoint_block": ["A", "B"],
        "fit_block": ["A", "B"],
        "method": SEARCHES if trajectories else METHODS,
    }
    if trajectories:
        grid["generation"] = range(17)
    check_grid(frame, grid)
    frame["allocation"] = np.where(
        frame.checkpoint_block == frame.fit_block, "reuse", "disjoint"
    )
    return frame


def numeric(series):
    """Accept booleans read as either native values or strings when other rows are missing."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    strings = series.astype("string").replace({"True": "1", "False": "0"})
    return pd.to_numeric(strings, errors="raise").astype(float)


def paired_levels(frame, metrics, *, by=()):
    keys = ["seed", "law", "allocation", "method", *by]
    values = frame[metrics].apply(numeric)
    if np.isinf(values).any().any():
        raise ValueError("Infinite archive value")
    grouped = pd.concat([frame[keys], values], axis=1).groupby(keys, sort=True)
    available = grouped[metrics].count()
    if ((available != 0) & (available != 2)).any().any():
        raise ValueError("Only one swap available in a bank-level estimand")
    return (
        grouped[metrics]
        .mean()
        .stack()
        .rename("value")
        .rename_axis(keys + ["metric"])
        .reset_index()
    )


def scores_by_bank(frame):
    frame = prepare(frame)
    metrics = ["fit_brier", "audit_brier", "audit_f1", "audit_excess", "n_providers"]
    if frame[metrics].isna().any().any():
        raise ValueError("One or more required scores are missing")
    frame["optimism_brier"] = frame.audit_brier - frame.fit_brier
    metrics.append("optimism_brier")
    if "fit_f1" in frame:
        frame["optimism_f1"] = frame.fit_f1 - frame.audit_f1
        metrics.extend(["fit_f1", "optimism_f1"])
    for key in ("selected_is_mean", "selected_is_signal", "used_mean_finalist"):
        if key in frame:
            if key == "selected_is_signal":
                frame[key] = (
                    numeric(frame[key]).astype(float).where(frame.law != "equal")
                )
            metrics.append(key)
    metrics.extend(
        key for key in ("score_calls", "proposals", "selected_nodes") if key in frame
    )
    membership = ["audit_best_final_excess", "audit_best_seen_excess"]
    if all(key in frame for key in membership):
        frame["deployed_to_final_gap"] = (
            frame.audit_excess - frame.audit_best_final_excess
        )
        frame["final_to_seen_gap"] = (
            frame.audit_best_final_excess - frame.audit_best_seen_excess
        )
        if (frame.final_to_seen_gap < -TOL).any():
            raise ValueError("Final minimum is less than the all-seen minimum")
        # A protected choice may explicitly deploy the separate mean outside final survivors.
        if (
            frame.loc[frame.method.isin(SEARCHES), "deployed_to_final_gap"] < -TOL
        ).any():
            raise ValueError(
                "An unrestricted selected tree beats its survivors' minimum"
            )
        metrics.extend(membership + ["deployed_to_final_gap", "final_to_seen_gap"])
    return paired_levels(frame, metrics)


def arm_differences(levels):
    idx = ["seed", "law", "allocation", "metric"]
    table = levels.pivot(index=idx, columns="method", values="value")
    pairs = [
        ("carry", "protected"),
        ("carry", "seeded"),
        ("seeded", "sham"),
        ("seeded", "stock"),
        ("sham", "stock"),
        ("protected", "seeded"),
        ("protected", "sham"),
        ("protected", "stock"),
    ]
    pairs.extend(
        (arm, ref)
        for arm in (*SEARCHES, "protected", "regularized")
        for ref in ("mean", "top_n", "prob_lr")
    )
    pairs.extend((arm, "regularized") for arm in (*SEARCHES, "protected"))
    output = []
    for arm, reference in pairs:
        diff = (table[arm] - table[reference]).dropna().rename("value").reset_index()
        diff["method"], diff["reference"] = arm, reference
        output.append(diff)
    return pd.concat(output, ignore_index=True)


def retention_by_bank(trajectory):
    frame = trajectory.sort_values([*CELL, "generation"])
    rows = []
    for cell, stages in frame.groupby(CELL, sort=False):
        base = dict(zip(CELL, cell))
        row = {**base, "allocation": stages.allocation.iloc[0]}
        if base["method"] == "carry" and "carry_reinserted" in stages:
            restored = numeric(stages.carry_reinserted)
            if (
                not restored.isin([0, 1]).all()
                or (stages.n_active_mean_exact <= 0).any()
            ):
                raise ValueError("Carry failed to preserve the canonical mean")
            row["carry_reinsertions"] = restored.sum()
            row["carry_ever_reinserted"] = int(restored.any())
        for target in ("mean_exact", "mean_equiv", "signal_exact", "signal_equiv"):
            if target.startswith("signal") and base["law"] == "equal":
                continue
            active = numeric(stages["n_active_" + target]).to_numpy()
            if not np.isfinite(active).all():
                raise ValueError(f"Missing active membership for {target}")
            present = active > 0
            ever_present = np.maximum.accumulate(present)
            lost = ~present & ever_present
            after_loss = np.maximum.accumulate(lost)
            # Reappearance requires observed absence after an earlier presence.
            for event, value in {
                "present_initial": present[0],
                "ever_present": present.any(),
                "present_final": present[-1],
                "ever_lost": lost.any(),
                "reappeared_after_loss": (present & after_loss).any(),
                "chosen_final": numeric(stages["selected_" + target]).iloc[-1],
            }.items():
                row[f"{target}_{event}"] = int(value)
        rows.append(row)
    events = pd.DataFrame(rows)
    metrics = [k for k in events if k not in CELL + ["allocation"]]
    # Keep the generation curves separate from pathwise loss/reappearance events.
    curves = frame.copy()
    presence = []
    for target in ("mean_exact", "mean_equiv", "signal_exact", "signal_equiv"):
        key = target + "_present"
        curves[key] = numeric(curves["n_active_" + target]).gt(0).astype(float)
        if target.startswith("signal"):
            curves.loc[curves.law == "equal", key] = np.nan
        presence.append(key)
    return paired_levels(events, metrics), paired_levels(
        curves, presence, by=["generation"]
    )


def generation_performance(frame):
    """Give each bank equal weight after averaging its two fit/checkpoint orientations."""
    keys = ["law", "allocation", "method", "generation"]
    measures = ["fit_selected", "audit_selected", "audit_selected_excess"]
    fractions = []
    frame = frame.copy()
    for target in ("mean", "signal"):
        defined = (
            frame.law.eq("signal")
            if target == "signal"
            else np.ones(len(frame), dtype=bool)
        )
        for role in ("winner", "available"):
            output = f"{target}_{role}_fraction"
            if role == "winner":
                values = numeric(frame[f"selected_{target}_equiv"])
            else:
                counts = numeric(frame[f"n_active_{target}_equiv"])
                values = counts.gt(0).where(counts.notna())
            frame[output] = values.astype(float).where(defined)
            if not frame.loc[defined, output].isin([0, 1]).all():
                raise ValueError(f"Missing or invalid {target} {role} indicator")
            fractions.append(output)
    if not np.isfinite(frame[measures]).all().all():
        raise ValueError("Nonfinite trajectory performance")
    by_bank = frame.groupby([*keys, "seed"])[[*measures, *fractions]].mean()
    summary = by_bank.groupby(keys).mean()
    summary["n_banks"] = by_bank.groupby(keys).size()
    summary["n_fits"] = frame.groupby(keys).size()
    if not (summary.n_fits == 2 * summary.n_banks).all():
        raise ValueError("Missing checkpoint orientation")
    for fraction in fractions:
        name = fraction.removesuffix("_fraction") + "_count"
        summary[name] = (summary[fraction] * summary.n_fits).round().astype("Int64")
    return summary.reset_index()


def intervals(rows, keys, draws, *, count_signs=False):
    wide = rows.pivot(index="seed", columns=keys, values="value").reindex(SEEDS)
    if not np.isfinite(wide).all().all():
        raise ValueError("An estimand does not cover all 32 banks")
    output = []
    for index, series in wide.items():
        x = series.to_numpy()
        lo, hi = np.quantile(x[draws].mean(axis=1), [0.025, 0.975])
        row = dict(zip(keys, index))
        row.update(n_banks=len(x), estimate=x.mean(), ci_low=lo, ci_high=hi)
        if count_signs:
            row.update(
                n_lower=int((x < -TOL).sum()),
                n_tied=int((abs(x) <= TOL).sum()),
                n_higher=int((x > TOL).sum()),
            )
        output.append(row)
    return pd.DataFrame(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", type=Path, default=Path(__file__).resolve().parent / "results"
    )
    args = parser.parse_args()
    folder = args.results
    levels = scores_by_bank(pd.read_csv(folder / "scores.csv"))
    differences = arm_differences(levels)
    trajectories = prepare(
        pd.read_csv(folder / "trajectories.csv.gz"), trajectories=True
    )
    events, curves = retention_by_bank(trajectories)
    performance = generation_performance(trajectories)
    keys = ["law", "allocation", "method", "metric"]
    contrast_keys = ["law", "allocation", "method", "reference", "metric"]
    draws = np.random.default_rng(97822).integers(len(SEEDS), size=(10_000, len(SEEDS)))
    summary = intervals(levels, keys, draws)
    comparisons = intervals(differences, contrast_keys, draws, count_signs=True)
    retention = intervals(events, keys, draws)
    generations = intervals(curves, [*keys, "generation"], draws)
    for table, name in (
        (levels, "bank-levels.csv.gz"),
        (differences, "bank-comparisons.csv.gz"),
        (events, "bank-retention.csv.gz"),
    ):
        table.to_csv(
            folder / name, index=False, compression={"method": "gzip", "mtime": 0}
        )
    summary.to_csv(folder / "summary.csv", index=False)
    comparisons.to_csv(folder / "comparisons.csv", index=False)
    retention.to_csv(folder / "retention.csv", index=False)
    generations.to_csv(folder / "generations.csv", index=False)
    performance.to_csv(folder / "generation-performance.csv", index=False)
    primary = comparisons.query(
        "law == 'equal' and allocation == 'reuse' and method == 'seeded'"
        " and reference == 'sham' and metric == 'audit_excess'"
    ).iloc[0]
    print(
        f"Primary seeded minus sham excess Brier {primary.estimate:+.6f} "
        f"(95% interval {primary.ci_low:+.6f}, {primary.ci_high:+.6f})"
    )


if __name__ == "__main__":
    main()

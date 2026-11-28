"""Summarize paired full-data fits. Run this file with a round number from 1 to 5."""

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "results"
METRICS = ("brier", "fit_brier", "gap", "macro_f1", "average_precision")
N_BOOT = 10_000
TIE = 1e-12


def summarize(scores, r):
    """Return group levels, standard family levels and bank-paired differences."""
    scores = scores.copy()
    scores["gap"] = scores["brier"] - scores["fit_brier"]
    if set(scores["round"]) != {r}:
        raise ValueError("Score file must contain one matching round")
    keys = ["cohort", "family", "seed", "method"]
    if scores.duplicated(keys).any() or scores[keys].isna().any().any():
        raise ValueError("Every bank and method must have one unique record")
    if not np.isfinite(scores[["brier", "fit_brier"]]).all().all():
        raise ValueError("All methods require finite fit and audit Brier scores")
    methods = sorted(scores["method"].unique())
    if not {"mean", "top_n", "lr", "full"}.issubset(methods):
        raise ValueError("All four reference methods are required")
    counts = scores.groupby(["cohort", "family", "seed"]).size()
    if (counts != len(methods)).any():
        raise ValueError("Missing method(s) in a paired bank")
    scores["group"] = scores["family"].where(scores["cohort"] != "standard", "standard")
    standard = scores[scores["cohort"] == "standard"]
    if standard["family"].nunique() != 27:
        raise ValueError("The standard-v5 panel has 27 fixed families")

    levels = []
    comparisons = []
    for group, records in scores.groupby("group", sort=True):
        sizes = records.groupby("family")["seed"].nunique()
        if sizes.nunique() != 1 or records["cohort"].nunique() != 1:
            raise ValueError(f"Unbalanced or ambiguous group {group}")
        n_families, n_seeds = len(sizes), int(sizes.iloc[0])
        coh = records["cohort"].iat[0]
        # One array reused across metrics and method pairs. Every fixed family
        # gets equal weight in the suite macro even in bootstrap replicates.
        draws = None
        if coh != "digits":
            draws = np.random.default_rng(91700 + r).integers(
                n_seeds, size=(N_BOOT, n_families, n_seeds)
            )
        group_levels = {
            m: {
                "group": group,
                "cohort": coh,
                "method": m,
                "families": n_families,
                "seeds_per_family": n_seeds,
                "banks": n_families * n_seeds,
            }
            for m in methods
        }
        for metric in METRICS:
            cells = records.pivot(
                index=["family", "seed"], columns="method", values=metric
            )
            cells = cells.sort_index().loc[:, methods]
            if cells.isna().any().any():
                # E.g. AP is available for binary data only. No renormalizing the
                # mixed binary/multiclass suite after dropping unavailable families.
                if cells.isna().nunique(axis=1).max() > 1:
                    raise ValueError(
                        f"Availability varies by method for {group} {metric}"
                    )
                for row in group_levels.values():
                    row[metric] = np.nan
                continue
            if not np.isfinite(cells.to_numpy()).all():
                raise ValueError(f"Non-finite scores in {group} {metric}")
            for m in methods:
                group_levels[m][metric] = cells[m].mean()
            higher_better = metric in {"macro_f1", "average_precision"}
            for m, ref in combinations(methods, 2):
                v = (cells[m] - cells[ref]).to_numpy().reshape(n_families, n_seeds)
                mean = v.mean()
                lo, hi = np.nan, np.nan
                if draws is not None:
                    bootstrap = v[np.arange(n_families)[None, :, None], draws].mean(
                        axis=(1, 2)
                    )
                    lo, hi = np.quantile(bootstrap, [0.025, 0.975])
                benefit = v if higher_better else -v
                better, worse = int((benefit > TIE).sum()), int((benefit < -TIE).sum())
                common = {
                    "group": group,
                    "cohort": coh,
                    "metric": metric,
                    "total": v.size,
                    "tied_banks": v.size - better - worse,
                }
                comparisons.extend(
                    [
                        {
                            **common,
                            "method": m,
                            "reference": ref,
                            "delta": mean,
                            "lo": lo,
                            "hi": hi,
                            "better_banks": better,
                            "worse_banks": worse,
                        },
                        {
                            **common,
                            "method": ref,
                            "reference": m,
                            "delta": -mean,
                            "lo": -hi,
                            "hi": -lo,
                            "better_banks": worse,
                            "worse_banks": better,
                        },
                    ]
                )
        levels.extend(group_levels.values())
    standard_means = standard.groupby(["family", "method"], sort=True)
    families = standard_means[list(METRICS)].mean()
    families["banks"] = standard_means.size()
    full_brier = families.xs("full", level="method")["brier"]
    families["delta_full"] = families["brier"].sub(full_brier, level="family")
    return pd.DataFrame(levels), families.reset_index(), pd.DataFrame(comparisons)


def scene_count_contrasts(scores, r):
    """Pair scene16 and scene128 on the same target images and source pool seed."""
    records = scores[scores["cohort"] == "clustered"].copy()
    if set(records["family"]) != {"scene16", "scene128"}:
        raise ValueError("Expected both scene-count conditions")
    records["gap"] = records["brier"] - records["fit_brier"]
    methods = sorted(records["method"].unique())
    n_seeds = records["seed"].nunique()
    draws = np.random.default_rng(91700 + r).integers(n_seeds, size=(N_BOOT, n_seeds))
    rows = []
    for metric in METRICS:
        target = records.pivot(
            index="seed", columns=["family", "method"], values=metric
        )
        if not np.isfinite(target.to_numpy()).all():
            raise ValueError(f"Missing {metric} on paired scene draws")
        for method, ref in combinations(methods, 2):
            d16 = target["scene16", method] - target["scene16", ref]
            d128 = target["scene128", method] - target["scene128", ref]
            diff = (d16 - d128).to_numpy()
            lo, hi = np.quantile(diff[draws].mean(axis=1), [0.025, 0.975])
            for a, b, sign in ((method, ref, 1), (ref, method, -1)):
                rows.append(
                    {
                        "metric": metric,
                        "method": a,
                        "reference": b,
                        "delta_scene16": sign * d16.mean(),
                        "delta_scene128": sign * d128.mean(),
                        "difference": sign * diff.mean(),
                        "lo": lo if sign == 1 else -hi,
                        "hi": hi if sign == 1 else -lo,
                        "paired_seeds": n_seeds,
                    }
                )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("round", type=int, choices=range(1, 6))
    r = parser.parse_args().round
    scores = pd.read_csv(OUT / f"round{r}-scores.csv", float_precision="round_trip")
    tables = summarize(scores, r)
    if r == 5:
        scene_count_contrasts(scores, r).to_csv(
            OUT / "round5-scene-count.csv", index=False
        )
    for name, table in zip(("levels", "families", "contrasts"), tables, strict=True):
        table.to_csv(OUT / f"round{r}-{name}.csv", index=False)


if __name__ == "__main__":
    main()

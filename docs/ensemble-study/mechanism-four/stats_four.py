"""Paired summaries of the held-out scores, fixed families and crossed scene conditions."""

import argparse
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
CODES = tuple(product((0, 1), repeat=3))
CELLS = [f"h{h}c{c}s{s}" for h, c, s in CODES]
METRICS = (
    "brier",
    "expected_brier",
    "fit_brier",
    "gap",
    "macro_f1",
    "ap",
    "nodes",
    "providers",
)
NEW_FAMILIES = {"grouped_rare_event", "shared_error_bloc", "rare_scene_specialist"}
N_BOOT = 10_000
KEY = ["family", "seed"]


def weights(nf, ns, r):
    """One equal total weight for each fixed family, with replacement inside it."""
    ids = np.random.default_rng(84100 + r).integers(ns, size=(N_BOOT, nf, ns))
    # Each column corresponds to a specific (family, seed). Matrix multiplication
    # then shares identical bootstrap draws across every model and metric.
    counts = np.stack([(ids == i).sum(axis=2) for i in range(ns)], axis=2)
    return counts.reshape(N_BOOT, nf * ns) / (nf * ns)


def validate(frame, r):
    keys = ["cohort", "family", "seed", "method"]
    if set(frame["round"]) != {r} or frame.duplicated(keys).any():
        raise ValueError("Expected unique bank and method keys for a single round")
    if frame[keys].isna().any().any():
        raise ValueError("Missing bank or method key")
    required = {"mean", "top_n", "lr", "full", "pull_mean"}
    methods = sorted(frame.method.unique())
    if not required.issubset(methods):
        raise ValueError("Missing a baseline")
    if (frame.groupby(keys[:-1]).size() != len(methods)).any():
        raise ValueError("Methods differ from one bank to another")
    if not np.isfinite(frame[["brier", "fit_brier", "macro_f1"]].to_numpy()).all():
        raise ValueError("Missing empirical performance scores")
    fac = frame.query("cohort == 'factorial'")
    if (
        set(fac.family) != set(CELLS)
        or not (fac.groupby("seed").family.nunique() == 8).all()
    ):
        raise ValueError("Each factorial seed must include all eight cells")
    names = set(frame.query("cohort == 'standard'").family)
    if len(names) != 30 or not NEW_FAMILIES.issubset(names):
        raise ValueError("Every round must include all 30 standard-v6 families")


def grouped(frame):
    standard = frame[frame.cohort == "standard"]
    yield "standard", standard, standard.family.nunique()
    legacy = standard[~standard.family.isin(NEW_FAMILIES)]
    yield "legacy27", legacy, legacy.family.nunique()
    factorial = frame[frame.cohort == "factorial"]
    # The same latent scenes and labels occur in every factorial cell for a seed.
    means = factorial.groupby(["seed", "method"])[list(METRICS)].mean().reset_index()
    means["family"] = "factorial"
    yield "factorial", means, 8
    for group, rows in frame[frame.cohort != "standard"].groupby("family"):
        yield group, rows, 1


def summarize(frame, r):
    validate(frame, r)
    frame = frame.assign(gap=frame.brier - frame.fit_brier)
    means, contrasts = [], []
    for group, records, n_families in grouped(frame):
        ns_by_family = records.groupby("family").seed.nunique()
        if ns_by_family.nunique() != 1:
            raise ValueError(f"Unbalanced seed count in {group}")
        ns = int(ns_by_family.iloc[0])
        w = weights(len(ns_by_family), ns, r)
        for method, part in records.groupby("method"):
            row = {
                "group": group,
                "method": method,
                "families": n_families,
                "seeds_per_family": ns,
            }
            for metric in METRICS:
                row[metric] = (
                    part[metric].mean() if part[metric].notna().all() else np.nan
                )
            means.append(row)
        for metric in METRICS:
            table = records.pivot(
                index=KEY, columns="method", values=metric
            ).sort_index()
            table = table.dropna(axis=1)
            if not np.isfinite(table.to_numpy()).all():
                raise ValueError(f"Nonfinite {group} {metric}")
            values = table.to_numpy()
            boot = w @ values
            for i, j in combinations(range(len(table.columns)), 2):
                d = values[:, i] - values[:, j]
                lo, hi = np.quantile(boot[:, i] - boot[:, j], [0.025, 0.975])
                for m, ref, delta, l, h in (
                    (table.columns[i], table.columns[j], d.mean(), lo, hi),
                    (table.columns[j], table.columns[i], -d.mean(), -hi, -lo),
                ):
                    contrasts.append(
                        {
                            "group": group,
                            "metric": metric,
                            "method": m,
                            "reference": ref,
                            "delta": delta,
                            "lo": l,
                            "hi": h,
                            "resampling_units": len(d),
                        }
                    )
    families = frame[frame.cohort == "standard"].groupby(["family", "method"])
    family_tab = families[list(METRICS)].mean()
    family_tab["seeds"] = families.size()
    incumbent = family_tab.xs("pull_mean", level="method").brier
    family_tab["delta_pull_mean"] = family_tab.brier.sub(incumbent, level="family")
    return pd.DataFrame(means), pd.DataFrame(contrasts), family_tab.reset_index()


def factorial_effects(frame, r):
    """High minus low differences and iterated differences, averaging other factors."""
    fac = frame[frame.cohort == "factorial"]
    rows = []
    methods = sorted(fac.method.unique())
    n = fac.seed.nunique()
    w = weights(1, n, r)
    codes = np.array(CODES)
    for metric in ("brier", "expected_brier", "macro_f1", "ap"):
        table = fac.pivot(index=["family", "seed"], columns="method", values=metric)
        if table.isna().any().any():
            raise ValueError(f"Missing {metric} for a fully crossed condition")
        cells = np.stack([table.loc[cell, methods].sort_index() for cell in CELLS])
        for order in (1, 2, 3):
            for factors in combinations(range(3), order):
                signs = (2 * codes[:, factors] - 1).prod(axis=1) / (2 ** (3 - order))
                label = "".join("HCS"[k] for k in factors)
                # shape [seed, method], preserving all eight matched worlds in each seed.
                effect = np.einsum("c,csm->sm", signs, cells)
                boot = w @ effect
                for i, method in enumerate(methods):
                    refs = [(None, "absolute")]
                    refs += [
                        (methods.index(ref), ref)
                        for ref in ("mean", "pull_mean", "lr")
                        if ref != method
                    ]
                    for j, ref in refs:
                        obs = effect[:, i] if j is None else effect[:, i] - effect[:, j]
                        draws = boot[:, i] if j is None else boot[:, i] - boot[:, j]
                        lo, hi = np.quantile(draws, [0.025, 0.975])
                        rows.append(
                            {
                                "factor": label,
                                "metric": metric,
                                "method": method,
                                "reference": ref,
                                "delta": obs.mean(),
                                "lo": lo,
                                "hi": hi,
                                "seeds": n,
                            }
                        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("round", type=int, choices=range(1, 5))
    r = parser.parse_args().round
    scores = pd.read_csv(OUT / f"r{r}-scores.csv", float_precision="round_trip")
    with threadpool_limits(limits=1):
        levels, contrasts, families = summarize(scores, r)
        effects = factorial_effects(scores, r)
    for name, table in (
        ("levels", levels),
        ("contrasts", contrasts),
        ("families", families),
        ("factorial-effects", effects),
    ):
        table.to_csv(OUT / f"r{r}-{name}.csv", index=False)


if __name__ == "__main__":
    main()

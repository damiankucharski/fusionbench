"""Compare source interventions at the same generator seeds and on the same target laws."""

import numpy as np
import pandas as pd


def profile_effects(raw, r):
    focus = raw[raw.cohort == "focus"]
    wide = focus.pivot(
        index=["family", "seed"], columns=["method", "world"], values="brier"
    )
    laws = (
        focus.groupby(["family", "seed", "world"]).oracle_brier.first().unstack("world")
    )

    def risks(family, metric):
        part = wide.loc[family]
        if metric in laws.columns:
            return part.xs(metric, axis=1, level="world")
        lo = part.xs("s0", axis=1, level="world")
        hi = part.xs("s100", axis=1, level="world")
        if metric == "max_regret":
            lo = lo.sub(laws.loc[family, "s0"], axis=0)
            hi = hi.sub(laws.loc[family, "s100"], axis=0)
        return lo.combine(hi, np.maximum)

    results = []
    for family in sorted(set(focus.family) - {"mixed"}):
        for metric in (
            "worst_brier",
            "max_regret",
            "s0",
            "s100",
            "s50",
            "specialist80",
        ):
            alt, base = risks(family, metric), risks("mixed", metric)
            if not alt.index.equals(base.index):
                raise ValueError("Source-design comparisons must pair the same seeds")
            n = len(alt)
            idx = np.random.default_rng(103000 + r).integers(n, size=(10_000, n))
            change = alt - base
            for name, d in (
                ("absolute", change),
                ("relative_to_native", change.sub(change.native, axis=0)),
            ):
                draws = d.to_numpy()[idx].mean(axis=1)
                lo, hi = np.quantile(draws, [0.025, 0.975], axis=0)
                results.append(
                    pd.DataFrame({"delta": d.mean(), "lo": lo, "hi": hi})
                    .reset_index()
                    .assign(
                        profile=family,
                        reference_profile="mixed",
                        world=metric,
                        contrast=name,
                        paired_sources=n,
                    )
                )
    return pd.concat(results, ignore_index=True)


def stage_comparisons(raw, r):
    methods = {3: ("discovery_only",), 4: ("selection_only", "refit_only")}[r]
    one = raw.drop_duplicates(["cohort", "family", "seed", "method"]).copy()
    groups = [("standard", one[one.cohort == "standard"])]
    groups.extend(one[one.cohort == "focus"].groupby("family"))
    out = []
    for group, sources in groups:
        index = ["cohort", "family", "seed"]
        for method in methods:
            target = sources[sources.method == method].set_index(index).sort_index()
            for reference in ("native", "full"):
                ref = sources[sources.method == reference].set_index(index).sort_index()
                if not ref.index.equals(target.index):
                    raise ValueError("Unpaired native stage intervention")
                out.append(
                    {
                        "group": group,
                        "method": method,
                        "reference": reference,
                        "banks": len(ref),
                        **{
                            k + "_changed": int((target[k] != ref[k]).sum())
                            for k in (
                                "path_hash",
                                "tree_hash",
                                "chosen",
                                "fit_hash",
                                "test_hash",
                            )
                        },
                    }
                )
    return pd.DataFrame(out)

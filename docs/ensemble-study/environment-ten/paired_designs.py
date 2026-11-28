"""Matched changes in detection and in fusion loss under interventions on the source."""

import numpy as np
import pandas as pd
from paired_stats import intervals, robust_panel, weights


def design_pairs(raw, r):
    focus = raw[raw.cohort == "focus"]
    comparisons = [
        ("varied", "homogeneous"),
        ("case_mix", "homogeneous"),
        ("mixed", "case_mix"),
        ("mixed", "varied"),
    ]
    base = {"varied", "homogeneous", "case_mix", "mixed"}
    comparisons.extend((family, "mixed") for family in sorted(set(focus.family) - base))
    return pd.concat(
        [
            summarize_pair(focus, change, reference, r)
            for change, reference in comparisons
        ]
    )


def summarize_pair(focus, change, reference, r):
    rows = []
    a = focus[focus.family == change]
    b = focus[focus.family == reference]
    choices = [part.drop_duplicates(["seed", "method"]) for part in (a, b)]
    indices = np.sort(a.seed.unique())
    w = weights(
        pd.MultiIndex.from_arrays(
            [np.repeat("pair", len(indices)), indices], names=["family", "seed"]
        ),
        r,
    )

    def record(metric, world, left, right):
        left, right = left.sort_index(), right.sort_index()
        if (
            not left.index.equals(right.index)
            or left.isna().any().any()
            or right.isna().any().any()
        ):
            raise ValueError("Unpaired source intervention")
        res = intervals(w, left - right).reset_index()
        rows.append(
            res.assign(
                profile=change,
                reference_profile=reference,
                metric=metric,
                world=world,
                paired_seeds=len(indices),
            )
        )

    for event in ("any", "shortcut", "specialist"):
        flags = []
        for source in choices:
            drop = source["drop"]
            value = drop.notna() if event == "any" else drop.eq(source["true_" + event])
            flags.append(
                source.assign(value=value.astype(float)).pivot(
                    index="seed", columns="method", values="value"
                )
            )
        record(event + "_drop_rate", "source", *flags)
    for world in [*sorted(set(a.world)), "worst_endpoint"]:
        losses = []
        for part in (a, b):
            if world == "worst_endpoint":
                loss = robust_panel(part, part.family.iloc[0]).droplevel("family")
            else:
                loss = part[part.world == world].pivot(
                    index="seed", columns="method", values="brier"
                )
            losses.append(loss)
        record("brier", world, *losses)
        record(
            "brier_delta_native",
            world,
            *(risk.sub(risk.native, axis=0) for risk in losses),
        )
    return pd.concat(rows)

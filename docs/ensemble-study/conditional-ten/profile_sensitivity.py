"""Describe R10 standard rankings under omissions and two predeclared research profiles."""

import json
from itertools import combinations, pairwise
from pathlib import Path

import numpy as np
import pandas as pd

from fusionbench.taxonomy import (
    APPLICATION_PROFILES,
    MECHANISM_GROUPS,
    TAXONOMY_VERSION,
    scenario_weights,
    weighted_score,
)

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
METHODS = ("native", "guarded", "single", "half", "replicated", "mean", "top_n", "lr")
TIE = 1e-12


class Panel:
    def __init__(self, rows):
        standard = rows[(rows.cohort == "standard") & (rows.world == "test")]
        matrix = standard.pivot(
            index=["family", "seed"], columns="method", values="brier"
        )
        if (
            set(matrix.columns) != set(METHODS)
            or not np.isfinite(matrix.to_numpy()).all()
        ):
            raise ValueError("Every source must have scores for all eight methods")
        matrix = matrix.loc[:, list(METHODS)].sort_index()
        self.means = matrix.groupby(level="family").mean()
        self.families = self.means.index.tolist()
        count = matrix.groupby(level="family").size()
        if set(self.families) != set(scenario_weights()) or not (count == 8).all():
            raise ValueError(
                "Expected all 30 standard-v6 families and eight paired banks per family"
            )
        cube = matrix.to_numpy().reshape(30, 8, len(METHODS))
        # Pair methods within a bank. Stratify by fixed scenario, resampling the
        # eight datasets separately inside each. Reuse these draws across all
        # omissions and blends to keep uncertainty comparisons paired.
        picks = np.random.default_rng(113010).integers(8, size=(10_000, 30, 8))
        self.resamples = np.stack(
            [cube[i][picks[:, i]].mean(axis=1) for i in range(30)], axis=1
        )

    def aggregate(self, weights):
        vector = np.asarray([weights.get(f, 0) for f in self.families])
        public = np.array(
            [weighted_score(self.means[m].to_dict(), weights) for m in METHODS]
        )
        np.testing.assert_allclose(
            public, vector @ self.means.to_numpy(), rtol=0, atol=2e-14
        )
        bootstrap = np.einsum("s,bsm->bm", vector, self.resamples)
        return public, bootstrap


def leaders(risk):
    return [
        m for m, value in zip(METHODS, risk, strict=True) if value <= risk.min() + TIE
    ]


def checked(panel, archive, group, weights):
    raw, boot = panel.aggregate(weights)
    official = archive.loc[group, list(METHODS)].to_numpy(dtype=float)
    np.testing.assert_allclose(raw, official, rtol=0, atol=2e-14)
    return raw, boot


def groups_and_omissions(panel, archive):
    records = []
    for group, members in MECHANISM_GROUPS.items():
        risk, boot = checked(
            panel, archive, f"cluster/{group}", scenario_weights(groups=group)
        )
        top = leaders(risk)
        for removed in (None, *members):
            score, draws = risk, boot
            if removed is not None:
                subset = [s for s in members if s != removed]
                score, draws = panel.aggregate(
                    scenario_weights(groups=group, scenarios=subset)
                )
            winners = leaders(score)
            for method in top:
                i = METHODS.index(method)
                others = [j for j in range(len(METHODS)) if j != i]
                gap = score[others].min() - score[i]
                distrib = draws[:, others].min(axis=1) - draws[:, i]
                lo, hi = np.quantile(distrib, (0.025, 0.975))
                records.append(
                    {
                        "group": group,
                        "scenarios": len(members),
                        "omitted": removed or "",
                        "full_group_leader": method,
                        "leaders_after_omission": "|".join(winners),
                        "risk": score[i],
                        "advantage_over_best_other": gap,
                        "margin_lo": lo,
                        "margin_hi": hi,
                        "top_or_tied": method in winners,
                        "bootstrap_top_or_tied": np.mean(distrib >= -TIE),
                    }
                )
    return pd.DataFrame(records)


def blend_weights(site, rules, share):
    return {
        s: w
        for s in sorted(set(site) | set(rules))
        if (w := (1 - share) * site.get(s, 0) + share * rules.get(s, 0)) > 0
    }


def envelope(site, rules):
    """Find all rank changes on the lower envelope of affine method risks."""
    slope = rules - site
    points = {0.0, 1.0}
    for i, j in combinations(range(len(METHODS)), 2):
        d = slope[i] - slope[j]
        if d:
            crossing = (site[j] - site[i]) / d
            if 0 < crossing < 1:
                points.add(float(crossing))
    ticks = []
    for x in sorted(points):
        if not ticks or x - ticks[-1] > TIE:
            ticks.append(x)
    ranges = []
    ticks[-1] = 1.0
    for a, b in pairwise(ticks):
        during = "|".join(leaders(site + slope * ((a + b) / 2)))
        if ranges and ranges[-1]["leaders_inside"] == during:
            ranges[-1]["end"] = b
        else:
            ranges.append({"start": a, "end": b, "leaders_inside": during})
    for seg in ranges:
        for name in ("start", "end"):
            seg["leaders_at_" + name] = "|".join(leaders(site + slope * seg[name]))
    return pd.DataFrame(ranges)


def blends(panel, archive):
    site = scenario_weights(group_weights=APPLICATION_PROFILES["multi_site"])
    rules = scenario_weights(group_weights=APPLICATION_PROFILES["rule_design"])
    a, a_draw = checked(panel, archive, "multi_site", site)
    b, b_draw = checked(panel, archive, "rule_design", rules)
    records = []
    for x in (0.0, 0.25, 0.5, 0.75, 1.0):
        risk, boot = panel.aggregate(blend_weights(site, rules, x))
        np.testing.assert_allclose(risk, (1 - x) * a + x * b, rtol=0, atol=2e-14)
        np.testing.assert_allclose(
            boot, (1 - x) * a_draw + x * b_draw, rtol=0, atol=2e-14
        )
        top = leaders(risk)
        for i, method in enumerate(METHODS):
            others = [j for j in range(len(METHODS)) if j != i]
            gap = boot[:, others].min(axis=1) - boot[:, i]
            lo, hi = np.quantile(gap, (0.025, 0.975))
            records.append(
                {
                    "rule_design_share": x,
                    "method": method,
                    "risk": risk[i],
                    "leads_or_ties": method in top,
                    "advantage_over_best_other": risk[others].min() - risk[i],
                    "margin_lo": lo,
                    "margin_hi": hi,
                    "bootstrap_top_or_tied": np.mean(gap >= -TIE),
                }
            )
    segments = envelope(a, b)
    for k in range(len(segments) - 1):
        left = segments.at[k, "leaders_inside"].split("|")
        right = segments.at[k + 1, "leaders_inside"].split("|")
        # Identical lines can have several names. Use the first method on
        # each side when describing uncertainty for this pair's intersection.
        i, j = METHODS.index(left[0]), METHODS.index(right[0])
        d0 = a_draw[:, i] - a_draw[:, j]
        d1 = b_draw[:, i] - b_draw[:, j]
        crosses = (d0 < -TIE) & (d1 > TIE)
        segments.loc[k, "end_pair_crossing_frequency"] = crosses.mean()
        if crosses.any():
            where = d0[crosses] / (d0[crosses] - d1[crosses])
            segments.loc[k, ["end_pair_conditional_lo", "end_pair_conditional_hi"]] = (
                np.quantile(where, (0.025, 0.975))
            )
    return pd.DataFrame(records), segments


def scenario_effects(panel, locked):
    records = []
    for i, family in enumerate(panel.families):
        for m in ("native", "guarded"):
            a, b = METHODS.index(locked), METHODS.index(m)
            delta = panel.means.iloc[i, a] - panel.means.iloc[i, b]
            lo, hi = np.quantile(
                panel.resamples[:, i, a] - panel.resamples[:, i, b], (0.025, 0.975)
            )
            records.append(
                {
                    "scenario": family,
                    "locked": locked,
                    "reference": m,
                    "delta": delta,
                    "lo": lo,
                    "hi": hi,
                    "contribution_to_standard_delta": delta / 30,
                }
            )
    return pd.DataFrame(records)


def signed(x):
    return "0" if abs(x) <= TIE else f"{x:+.3g}"


def note(omissions, grid, segments, effects, locked):
    lines = [
        "# Sensitivity to the mix of benchmark scenarios",
        "",
        f"Round 10 used eight fresh draws for each of 30 fixed scenarios. Rankings below use mean full-class Brier, with lower values preferred. Every ranking includes all five native configurations, the arithmetic mean, Top-N and logistic stacking (LR). The method locked in round 9 was `{locked}`.",
        "",
        "## Does the group winner depend on one scenario?",
        "",
        "A positive margin is the reduction in Brier relative to the best other method. Each omission reweights the remaining scenarios equally. Methods within 10⁻¹² in mean risk share the lead. An omitted-scenario range extending below zero means an alternative overtakes the full-group leader.",
        "",
        "| Mechanism | Full-group leader | Margin [95% interval] | Retains or shares lead after omission | Margin range after omission |",
        "|---|---|---:|---:|---:|",
    ]
    grouped_rows = {}
    for (group, method), rows in omissions.groupby(
        ["group", "full_group_leader"], sort=False
    ):
        all_ = rows[rows.omitted == ""].iloc[0]
        omitted = rows[rows.omitted != ""]
        values = (
            f"{signed(all_.advantage_over_best_other)} [{signed(all_.margin_lo)}, {signed(all_.margin_hi)}]",
            f"{omitted.top_or_tied.sum()}/{len(omitted)}",
            f"{signed(omitted.advantage_over_best_other.min())} to {signed(omitted.advantage_over_best_other.max())}",
        )
        grouped_rows.setdefault((group, *values), []).append(method)
    for (group, margin, kept, limits), names in grouped_rows.items():
        lines.append(f"| {group} | {', '.join(names)} | {margin} | {kept} | {limits} |")
    changes = omissions[(omissions.omitted != "") & ~omissions.top_or_tied]
    if not changes.empty:
        lines += [
            "",
            "The three omissions that cost a full-group leader the most Brier were",
            "",
        ]
        cases = changes.sort_values("advantage_over_best_other").drop_duplicates(
            ["group", "omitted"]
        )
        for v in cases.head(3).itertuples():
            winners = v.leaders_after_omission.replace("|", ", ")
            lines.append(
                f"- Omitting `{v.omitted}` from {v.group} puts {winners} first. The original leader trails by {abs(v.advantage_over_best_other):.3g}."
            )
    lines += [
        "",
        "## What changes across the two illustrative profiles?",
        "",
        "λ is the fraction of the fixed rule-design profile. The balance goes to the multi-site profile. These two profiles describe examples of research priorities. Applying a profile to deployment would require evidence about its scenario frequencies.",
        "",
        "| λ | Leading methods | Mean Brier | Margin [95% interval] |",
        "|---:|---|---:|---:|",
    ]
    for v in grid[grid.leads_or_ties].itertuples():
        lines.append(
            f"| {v.rule_design_share:g} | {v.method} | {v.risk:.5f} | {signed(v.advantage_over_best_other)} [{signed(v.margin_lo)}, {signed(v.margin_hi)}] |"
        )
    lines += [
        "",
        "Each method’s risk changes linearly with λ. We calculated all pairwise intersections and compared the corresponding risks between consecutive intersections.",
        "",
    ]
    for s in segments.itertuples():
        if (
            s.start == 0
            and s.end == 1
            and s.leaders_at_start == s.leaders_at_end == s.leaders_inside
        ):
            lines.append(
                f"The point leader throughout 0 ≤ λ ≤ 1 is {s.leaders_inside.replace('|', ', ')}."
            )
            continue
        lines.append(
            f"- {s.start:.6g} < λ < {s.end:.6g} — {s.leaders_inside.replace('|', ', ')}. At the left endpoint {s.leaders_at_start.replace('|', ', ')}. At the right endpoint {s.leaders_at_end.replace('|', ', ')}."
        )
        if s.end < 1 and np.isfinite(getattr(s, "end_pair_conditional_lo", np.nan)):
            lines.append(
                f"  The pair swapped order in this direction in {s.end_pair_crossing_frequency:.0%} of seed resamples. Conditional on that swap, its intersection ranged from {s.end_pair_conditional_lo:.3f} to {s.end_pair_conditional_hi:.3f} across the central 95%. Other methods may lead there."
            )
    lines += [
        "",
        "## Which scenarios move the locked comparison?",
        "",
        "Signed changes below are locked minus reference Brier within each scenario. Dividing a change by 30 gives its contribution to the standard macro. Shown are the largest absolute changes in either direction.",
        "",
    ]
    for ref, rows in effects.groupby("reference", sort=False):
        order = rows[rows.delta.abs() > TIE].sort_values(
            "delta", key=lambda s: s.abs(), ascending=False
        )
        desc = ", ".join(
            f"{r.scenario} {signed(r.delta)}" for r in order.head(4).itertuples()
        )
        lines.append(
            f"- Against `{ref}`, {desc or 'all 30 scenario means tie'}. Standard macro change {signed(rows.contribution_to_standard_delta.sum())}."
        )
    lines += [
        "",
        "Intervals are the 2.5th and 97.5th percentiles of 10,000 paired seed resamples within each fixed scenario. We recompute the closest competitor in each resample. These descriptive intervals condition on the reported leader and are unadjusted for multiple comparisons. Omissions measure sensitivity to group membership. Crossings describe the risks observed on this panel.",
        "",
        "[Omissions by scenario](results/r10-profile-omissions.csv) · [All eight methods at each blend](results/r10-profile-grid.csv) · [Continuous intervals](results/r10-profile-envelope.csv) · [All scenario effects](results/r10-profile-drivers.csv)",
        "",
    ]
    return "\n".join(lines)


def main():
    raw = pd.read_csv(OUT / "r10.csv", float_precision="round_trip")
    if set(raw["round"]) != {10}:
        raise ValueError("Expected the locked R10 confirmation panel")
    panel = Panel(raw)
    archive = pd.read_csv(OUT / "r10-levels.csv").pivot(
        index="group", columns="method", values="risk"
    )
    locked = json.loads((HERE / "selection.json").read_text())["chosen"]
    if TAXONOMY_VERSION != "mechanisms-v1":
        raise ValueError("The frozen taxonomy changed")
    checked(panel, archive, "standard", scenario_weights())
    omissions = groups_and_omissions(panel, archive)
    grid, segments = blends(panel, archive)
    effects = scenario_effects(panel, locked)
    for name, df in {
        "omissions": omissions,
        "grid": grid,
        "envelope": segments,
        "drivers": effects,
    }.items():
        df.to_csv(OUT / f"r10-profile-{name}.csv", index=False)
    (HERE / "profile-sensitivity.md").write_text(
        note(omissions, grid, segments, effects, locked)
    )


if __name__ == "__main__":
    main()

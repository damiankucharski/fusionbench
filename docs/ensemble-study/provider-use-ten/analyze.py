"""Paired source-bank comparisons, masked-provider diagnostics and frozen confirmation."""

import argparse
import gzip
import json
import subprocess
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from gate_pairs import gate_pairs

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
LOCK = HERE / "selection.json"
BASES = ("native", "mean", "top_n", "lr")
REFS = (*BASES, "hard", "half")
CHOOSE = ("protect", "adaptive", "half")
CORE = (*BASES, "hard", "half", "protect", "adaptive", "shuffled_evidence")
EXTRA = {
    2: ("permuted_env",),
    4: ("quarter", "three_quarter", "full"),
    5: ("half_d2", "half_d32"),
    8: ("direct_only", "routing_only"),
}
PROFILES = {
    2: ("varied",),
    6: ("narrow",),
    7: ("small128",),
    8: ("homogeneous",),
    9: ("reversed",),
    10: ("varied", "narrow", "small128", "reversed", "homogeneous"),
}
KEY = ["cohort", "family", "seed", "world"]
MEASURES = ("brier", "macro_f1", "providers")
RISK_LIMITS = {
    "mixed/max_regret": 0.0,
    "standard": 0.005,
    "mixed/s100": 0.005,
    "standard/gated_experts": 0.010,
}


def read(r):
    return pd.read_csv(OUT / f"r{r:02d}.csv", float_precision="round_trip")


def check_input(raw, r):
    if set(raw["round"]) != {r} or raw.duplicated([*KEY, "method"]).any():
        raise ValueError("Duplicate bank scores or incorrect round")
    methods = set(CORE) | set(EXTRA.get(r, ()))
    if (
        set(raw.method) != methods
        or not raw.groupby(KEY).method.nunique().eq(len(methods)).all()
    ):
        raise ValueError("Methods do not share identical bank-world conditions")
    if (
        not np.isfinite(raw[list(MEASURES)]).all().all()
        or raw[["test_hash", "fit_hash"]].isna().any().any()
    ):
        raise ValueError("Missing score or forecast identity")
    identity = [
        "source_x_hash",
        "source_y_hash",
        "test_x_hash",
        "target_hash",
        "n_fit",
        "n_screen",
    ]
    if not raw.groupby(KEY)[identity].nunique().eq(1).all().all():
        raise ValueError("Compared methods used different source or target data")
    standard = raw[raw.cohort == "standard"]
    focus = raw[raw.cohort == "focus"]
    gate = raw[raw.cohort == "gate_control"]
    if (
        standard.family.nunique() != 30
        or set(standard.world) != {"test"}
        or "gated_experts" not in set(standard.family)
    ):
        raise ValueError("Standard test panel is incomplete")
    if set(focus.family) != {"mixed", "case_mix", *PROFILES.get(r, ())} or set(
        focus.world
    ) != {"s0", "s50", "s100", "specialist80"}:
        raise ValueError("Unexpected focus panel")
    if set(gate.family) != (
        {"gated_original", "gated_shuffled"} if r in (3, 10) else set()
    ) or (len(gate) > 0 and set(gate.world) != {"test"}):
        raise ValueError("Missing or unregistered gate control")
    if len(focus) + len(standard) + len(gate) != len(raw):
        raise ValueError("Unknown cohort")
    for (cohort, family), rows in raw.groupby(["cohort", "family"]):
        offset = {"standard": 400, "focus": 100, "gate_control": 700}[cohort]
        n = (
            {9: 4, 10: 8}.get(r, 2)
            if cohort == "standard"
            else {9: 12, 10: 24}.get(r, 6)
        )
        start = 940000 + r * 1000 + offset
        if set(rows.seed) != set(range(start, start + n)):
            raise ValueError(f"Incorrect draws for {cohort}/{family}")
    if not focus.groupby(["family", "seed", "method"]).world.nunique().eq(4).all():
        raise ValueError("Missing focal worlds")
    if (
        focus.oracle_brier.isna().any()
        or not focus.groupby(KEY).oracle_brier.nunique().eq(1).all()
    ):
        raise ValueError("Oracle risk must be available and shared on each focal world")
    if (
        not raw[raw.cohort != "focus"].oracle_brier.isna().all()
        or (focus.brier - focus.oracle_brier).min() < -1e-10
    ):
        raise ValueError("Invalid oracle Brier")
    consistency = [
        "drop",
        "strength",
        "fit_hash",
        "test_hash",
        "n_fit",
        "true_shortcut",
        "true_specialist",
        "direct_gain",
        "routing_gain",
    ]
    if (
        not raw.groupby(["cohort", "family", "seed", "method"])[consistency]
        .nunique(dropna=False)
        .eq(1)
        .all()
        .all()
    ):
        raise ValueError(
            "Procedure, provider evidence or forecast changed across target worlds"
        )
    if (
        focus[["true_shortcut", "true_specialist"]].isna().any().any()
        or focus.true_shortcut.eq(focus.true_specialist).any()
    ):
        raise ValueError("The two focal roles must be present and distinct")
    return {
        "sources": len(raw.drop_duplicates(["cohort", "family", "seed"])),
        "bank_world_conditions": len(raw.drop_duplicates(KEY)),
        "score_rows": len(raw),
    }


def trace_audit(raw, r):
    """Rebuild every choice from frozen source evidence and check equivalent forecasts."""
    with gzip.open(OUT / f"r{r:02d}-source.jsonl.gz", "rt") as f:
        traces = {(j["cohort"], j["family"], j["seed"]): j for j in map(json.loads, f)}
    source = raw.drop_duplicates(["cohort", "family", "seed", "method"])
    n = 0
    for key, rows in source.groupby(["cohort", "family", "seed"]):
        info = traces[key]
        for row in rows.itertuples():
            if row.method in BASES:
                if not pd.isna(row.drop) or row.strength != 0:
                    raise ValueError("Baseline must not screen")
                continue
            provider = info["flags"][
                "permuted" if row.method == "permuted_env" else "original"
            ]
            if pd.isna(row.drop) != (provider is None) or (
                provider is not None and row.drop != provider
            ):
                raise ValueError("Archived flag differs from selected provider")
            if provider is None:
                if row.strength != 0 or not (
                    pd.isna(row.direct_gain) and pd.isna(row.routing_gain)
                ):
                    raise ValueError("Unflagged input must retain the native forecast")
            else:
                mode = "shuffled" if row.method == "shuffled_evidence" else "real"
                ev = info["evidence"][f"{provider}/{mode}"]
                if (
                    row.direct_gain != ev["direct_gain"]
                    or row.routing_gain != ev["routing_gain"]
                ):
                    raise ValueError(
                        "Archived cross-fitted gain differs from policy metadata"
                    )
                d, g = row.direct_gain > 0.02, row.routing_gain > 0.02
                setting = {
                    "hard": 1.0,
                    "half": 0.5,
                    "quarter": 0.25,
                    "three_quarter": 0.75,
                    "full": 1.0,
                    "half_d2": 0.5,
                    "half_d32": 0.5,
                    "protect": 0.0 if d or g else 0.5,
                    "direct_only": 0.0 if d else 0.5,
                    "routing_only": 0.0 if g else 0.5,
                }
                expected = setting.get(row.method, 0.0 if g else 0.25 if d else 0.75)
                if row.strength != expected:
                    raise ValueError(
                        f"{row.method} differs from registered attenuation"
                    )
            n += 1
    if len(traces) != source.groupby(["cohort", "family", "seed"]).ngroups:
        raise ValueError("Source traces do not cover all banks")
    native = raw[raw.method == "native"].set_index(KEY)
    keep = raw[(~raw.method.isin(BASES)) & (raw.strength == 0)]
    expected = native.reindex(pd.MultiIndex.from_frame(keep[KEY]))
    for col in ("fit_hash", "test_hash", *MEASURES):
        if not np.array_equal(expected[col].to_numpy(), keep[col].to_numpy()):
            raise ValueError("Zero attenuation must reproduce native")
    soft = raw[~raw.method.isin((*BASES, "hard")) & raw["drop"].notna()].copy()
    soft["donors"] = soft.method.map({"half_d2": 2, "half_d32": 32}).fillna(8)
    same = soft.groupby([*KEY, "drop", "strength", "donors"], dropna=False)
    if not same[["fit_hash", "test_hash", "brier"]].nunique().eq(1).all().all():
        raise ValueError("Equivalent soft deployment produced distinct forecasts")
    return {"source_decisions_checked": n, "unmodified_forecasts_checked": len(keep)}


def bootstrap_weights(index, r):
    nf = index.get_level_values("family").nunique()
    n = len(index) // nf
    if (
        not index.is_monotonic_increasing
        or not pd.Series(1, index=index).groupby(level="family").sum().eq(n).all()
    ):
        raise ValueError("Expect a balanced panel ordered by family and seed")
    picks = np.random.default_rng(93000 + r).integers(n, size=(10_000, nf, n))
    counts = np.stack([(picks == i).sum(2) for i in range(n)], axis=2)
    return counts.reshape(10_000, len(index)) / len(index)


def sampled(rows):
    return rows.pivot(
        index=["family", "seed"], columns="method", values="brier"
    ).sort_index()


def endpoints(rows):
    sub = rows[rows.world.isin(("s0", "s100"))].copy()
    sub["regret"] = sub.brier - sub.oracle_brier
    grp = sub.groupby(["family", "seed", "method"])
    if not grp.world.nunique().eq(2).all():
        raise ValueError("Each focal seed needs both endpoints")
    return {
        key: grp[field].max().unstack().sort_index()
        for key, field in (("worst_brier", "brier"), ("max_regret", "regret"))
    }


def panels(raw):
    std = raw[raw.cohort == "standard"]
    yield "standard", "standard", "all", "test", "brier", sampled(std), std
    for (cohort, family), part in raw.groupby(["cohort", "family"]):
        if cohort == "standard":
            yield (
                "standard/" + family,
                cohort,
                family,
                "test",
                "brier",
                sampled(part),
                part,
            )
        else:
            for world, rows in part.groupby("world"):
                yield (
                    family + "/" + world,
                    cohort,
                    family,
                    world,
                    "brier",
                    sampled(rows),
                    rows,
                )
            if cohort == "focus":
                for metric, values in endpoints(part).items():
                    yield (
                        family + "/" + metric,
                        cohort,
                        family,
                        metric,
                        metric,
                        values,
                        None,
                    )


def estimates(raw, r):
    levels, contrasts = [], []
    for group, cohort, family, world, metric, risk, rows in panels(raw):
        labels = {
            "group": group,
            "cohort": cohort,
            "family": family,
            "world": world,
            "metric": metric,
            "families": risk.index.get_level_values("family").nunique(),
            "banks": len(risk),
        }
        level = risk.mean().rename("risk").to_frame()
        if rows is not None:
            level = level.join(
                rows.groupby("method")[[*MEASURES, "oracle_brier"]].mean()
            )
        elif metric == "worst_brier":
            level["brier"] = level.risk
        levels.append(level.reset_index().assign(**labels))
        w = bootstrap_weights(risk.index, r)
        for ref in REFS:
            if ref not in risk:
                continue
            d = risk.sub(risk[ref], axis=0).drop(columns=ref)
            lo, hi = np.quantile(w @ d.to_numpy(), [0.025, 0.975], axis=0)
            contrasts.append(
                pd.DataFrame({"delta": d.mean(), "lo": lo, "hi": hi})
                .reset_index()
                .assign(reference=ref, **labels)
            )
    return pd.concat(levels, ignore_index=True), pd.concat(contrasts, ignore_index=True)


def diagnostic_summary(raw):
    source_key = ["cohort", "family", "seed"]
    unique = raw.drop_duplicates([*source_key, "method"]).copy()
    native = unique[unique.method == "native"].set_index(source_key)["test_hash"]
    unique["changed_forecast"] = (
        unique.test_hash.to_numpy()
        != native.reindex(pd.MultiIndex.from_frame(unique[source_key])).to_numpy()
    )
    groups = [("standard", unique[unique.cohort == "standard"])]
    groups.extend(
        (f"{c}/{f}", rows) for (c, f), rows in unique.groupby(["cohort", "family"])
    )
    result = []
    for group, part in groups:
        for m, per in part.groupby("method"):
            drop = per["drop"]
            flag = drop.notna()
            strength = per.strength
            flat = per[flag]
            labeled = per.true_shortcut.notna()
            dat = {
                "group": group,
                "method": m,
                "banks": len(per),
                "flagged": int(flag.sum()),
                "flag_rate": flag.mean(),
                "changed_forecasts": int(per.changed_forecast.sum()),
                "mean_strength_if_flagged": flat.strength.mean(),
                "direct_gain_if_flagged": flat.direct_gain.mean(),
                "routing_gain_if_flagged": flat.routing_gain.mean(),
                "n_flagged_direct_above_002": int((flat.direct_gain > 0.02).sum()),
                "n_flagged_routing_above_002": int((flat.routing_gain > 0.02).sum()),
                "zero_strength_if_flagged": int((flat.strength == 0).sum()),
                "full_strength_if_flagged": int((flat.strength == 1).sum()),
            }
            for name in ("shortcut", "specialist"):
                matches = drop.eq(per["true_" + name])
                dat[name + "_flags"] = int(matches.sum()) if labeled.any() else np.nan
                dat[name + "_positive_strength"] = (
                    int((matches & (strength > 0)).sum()) if labeled.any() else np.nan
                )
                dat[name + "_mean_strength_when_flagged"] = strength[matches].mean()
            result.append(dat)
    return pd.DataFrame(result)


def binding_endpoint(raw):
    """Count which endpoint governs each source-level absolute- or excess-risk maximum."""
    rows = raw[(raw.cohort == "focus") & raw.world.isin(("s0", "s100"))].copy()
    rows["regret"] = rows.brier - rows.oracle_brier
    summary = []
    for (profile, m), group in rows.groupby(["family", "method"]):
        for metric in ("brier", "regret"):
            pair = group.pivot(index="seed", columns="world", values=metric)
            diff = pair.s0 - pair.s100
            # Count near-equal values as ties only in this descriptive table.
            # Selection and confirmation always use the actual numeric maximum.
            summary.append(
                {
                    "family": profile,
                    "method": m,
                    "metric": metric,
                    "banks": len(diff),
                    "s0_higher": int((diff > 1e-12).sum()),
                    "s100_higher": int((diff < -1e-12).sum()),
                    "effectively_tied": int((diff.abs() <= 1e-12).sum()),
                }
            )
    return pd.DataFrame(summary)


def role_summary(raw):
    # Repeated target worlds share the same training decision.
    source = (
        raw[(raw.cohort == "focus") & raw["drop"].notna()]
        .drop_duplicates(["family", "seed", "method"])
        .copy()
    )
    source["role"] = np.select(
        [
            source["drop"] == source.true_shortcut,
            source["drop"] == source.true_specialist,
        ],
        ["shortcut", "specialist"],
        default="other",
    )
    direct = source.direct_gain > 0.02
    routing = source.routing_gain > 0.02
    source["evidence"] = np.select(
        [direct & routing, direct, routing],
        ["both", "direct_only", "routing_only"],
        default="neither",
    )
    return (
        source.groupby(["family", "method", "role", "evidence"])
        .agg(
            flagged_banks=("seed", "size"),
            mean_direct_gain=("direct_gain", "mean"),
            mean_routing_gain=("routing_gain", "mean"),
            mean_strength=("strength", "mean"),
        )
        .reset_index()
    )


def select(raw):
    standard = sampled(raw[raw.cohort == "standard"])
    risks = endpoints(raw[(raw.cohort == "focus") & (raw.family == "mixed")])[
        "max_regret"
    ]
    std = standard.sub(standard.native, axis=0).mean()
    safety = {m: float(std[m]) for m in CHOOSE}
    worst = {m: float(risks[m].mean()) for m in CHOOSE}
    eligible = [m for m in CHOOSE if safety[m] <= 0.005]
    return {
        "chosen": min(eligible, key=worst.get) if eligible else "native",
        "mixed_max_regret": worst,
        "standard_delta_native": safety,
        "eligible": eligible,
        "r09_sha256": sha256((OUT / "r09.csv").read_bytes()).hexdigest(),
    }


def committed_selection():
    root = Path(
        subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], cwd=HERE, text=True
        ).strip()
    )
    blob = subprocess.check_output(
        ["git", "show", f"HEAD:{LOCK.relative_to(root)}"], cwd=root
    )
    record = json.loads(blob)
    if LOCK.read_bytes() != blob or select(read(9)) != record:
        raise RuntimeError("Selection or its evidence differs from the committed lock")
    return record


def confirm(table):
    locked = committed_selection()
    chosen = locked["chosen"]
    part = table[(table.method == chosen) & (table.reference == "native")].set_index(
        "group"
    )
    tests = {}
    for group, limit in RISK_LIMITS.items():
        diff = (
            {k: 0.0 for k in ("delta", "lo", "hi")}
            if chosen == "native"
            else part.loc[group, ["delta", "lo", "hi"]].astype(float).to_dict()
        )
        passed = diff["hi"] < 0 if group == "mixed/max_regret" else diff["hi"] <= limit
        tests[group] = {**diff, "upper_limit": limit, "passed": bool(passed)}
    other = table[
        (table.method == chosen)
        & (table.reference == "hard")
        & (table.group.isin(RISK_LIMITS))
    ].set_index("group")
    return {
        "chosen": chosen,
        "r09_sha256": locked["r09_sha256"],
        "primary_pass": tests["mixed/max_regret"]["passed"],
        "all_guardrails_pass": all(
            v["passed"] for k, v in tests.items() if k != "mixed/max_regret"
        ),
        "all_pass": all(v["passed"] for v in tests.values()),
        "checks": tests,
        "chosen_minus_hard": other[["delta", "lo", "hi"]].to_dict("index"),
    }


def save(path, record):
    path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("round", type=int, choices=range(1, 11))
    r = p.parse_args().round
    raw = read(r)
    audit = check_input(raw, r)
    audit.update(trace_audit(raw, r))
    with threadpool_limits(limits=1):
        levels, contrasts = estimates(raw, r)
    for name, frame in {
        "levels": levels,
        "contrasts": contrasts,
        "families": levels[(levels.cohort == "standard") & (levels.family != "all")],
        "diagnostics": diagnostic_summary(raw),
        "role-diagnostics": role_summary(raw),
        "binding": binding_endpoint(raw),
    }.items():
        frame.to_csv(OUT / f"r{r:02d}-{name}.csv", index=False)
    if r in (3, 10):
        source, paired = gate_pairs(OUT / f"r{r:02d}-source.jsonl.gz", r)
        source.to_csv(OUT / f"r{r:02d}-gate-utility-sources.csv", index=False)
        paired.to_csv(OUT / f"r{r:02d}-gate-utility.csv", index=False)
    save(OUT / f"r{r:02d}-integrity.json", audit)
    if r == 9:
        lock = select(raw)
        if LOCK.exists() and json.loads(LOCK.read_text()) != lock:
            raise RuntimeError("A different decision is already present")
        save(LOCK, lock)
    if r == 10:
        save(OUT / "confirmation.json", confirm(contrasts))


if __name__ == "__main__":
    main()

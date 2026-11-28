"""Fixed-family seed comparisons and committed rule selection for the case-mix study."""

import argparse
import json
import subprocess
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
from matched_profiles import profile_effects, stage_comparisons
from threadpoolctl import threadpool_limits
from weight_audit import audit as audit_weights

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
LOCK = HERE / "selection.json"
CORE = ("native", "mean", "top_n", "lr", "full", "cap2", "guarded")
CHEAP = ("mean", "top_n", "lr")
CANDIDATES = ("guarded", "cap2", "full")
EXTRA = {
    2: ("shuffled",),
    3: ("discovery_only",),
    4: ("selection_only", "refit_only"),
    5: ("cap15",),
    8: ("permuted_weights",),
}
PROFILES = {
    2: ("varied",),
    3: ("reversed",),
    5: ("extreme",),
    6: ("small128",),
    7: ("noisy_context",),
    8: ("homogeneous",),
    9: ("reversed",),
    10: ("varied", "reversed", "extreme", "small128", "noisy_context", "homogeneous"),
}
KEY = ["cohort", "family", "seed", "world"]
SOURCE = KEY[:3]
METRICS = ["brier", "macro_f1", "providers"]
LIMITS = {
    "mixed/max_regret": 0.0,
    "standard": 0.005,
    "case_mix/specialist80": 0.010,
    "mixed/s0": 0.010,
}


def read(r):
    return pd.read_csv(OUT / f"r{r:02d}.csv", float_precision="round_trip")


def validate(raw, r):
    if set(raw["round"]) != {r} or raw.duplicated([*KEY, "method"]).any():
        raise ValueError("Duplicate or incorrectly labeled round rows")
    methods = set(CORE) | set(EXTRA.get(r, ()))
    if (
        set(raw.method) != methods
        or not raw.groupby(KEY).method.nunique().eq(len(methods)).all()
    ):
        raise ValueError("Unbalanced method-by-bank panel")
    if (
        set(raw.cohort) != {"standard", "focus"}
        or not np.isfinite(raw[METRICS]).all().all()
    ):
        raise ValueError("Missing scores or unknown cohort")
    if raw[["fit_hash", "test_hash"]].isna().any().any():
        raise ValueError("Every method must have source and target forecast identities")
    std = raw[raw.cohort == "standard"]
    focus = raw[raw.cohort == "focus"]
    if (
        std.family.nunique() != 30
        or set(std.world) != {"test"}
        or set(focus.family) != {"mixed", "case_mix", *PROFILES.get(r, ())}
    ):
        raise ValueError("Unexpected registered family panel")
    if (
        set(focus.world) != {"s0", "s50", "s100", "specialist80"}
        or not focus.groupby([*SOURCE, "method"]).world.nunique().eq(4).all()
    ):
        raise ValueError("Incomplete focal target panel")
    for (cohort, family), rows in raw.groupby(["cohort", "family"]):
        n = (
            {9: 3, 10: 6}.get(r, 2)
            if cohort == "standard"
            else {9: 10, 10: 20}.get(r, 6)
        )
        a = 1040000 + r * 1000 + (400 if cohort == "standard" else 100)
        if set(rows.seed) != set(range(a, a + n)):
            raise ValueError(f"Incorrect data seeds for {cohort}/{family}")
    if (
        not std.oracle_brier.isna().all()
        or focus.oracle_brier.isna().any()
        or not focus.groupby(KEY).oracle_brier.nunique().eq(1).all()
    ):
        raise ValueError("Missing or inconsistent focal Bayes risk")
    if (focus.brier - focus.oracle_brier).min() < -1e-10:
        raise ValueError("Focal Brier falls below the true conditional risk")
    if not focal_sources_stable(focus):
        raise ValueError("Fitted decision or forecast depends on the focal target")
    identity = [
        c
        for c in (
            "n_fit",
            "source_x_hash",
            "source_y_hash",
            "test_x_hash",
            "target_hash",
        )
        if c in raw
    ]
    if (
        identity
        and not raw.groupby(KEY)[identity].nunique(dropna=False).eq(1).all().all()
    ):
        raise ValueError("Compared methods used different input or target banks")
    # For fixed forecasts the target joint distribution is affine in shortcut activity.
    panel = focus[focus.world.isin(["s0", "s50", "s100"])].pivot(
        index=[*SOURCE, "method"], columns="world", values="brier"
    )
    err = float((panel.s50 - 0.5 * (panel.s0 + panel.s100)).abs().max())
    if err > 3e-12:
        raise ValueError(
            "Exact intermediate target risk must match endpoint interpolation"
        )
    return {
        "sources": raw[SOURCE].drop_duplicates().shape[0],
        "target_conditions": raw[KEY].drop_duplicates().shape[0],
        "score_rows": len(raw),
        "maximum_midpoint_error": err,
    }


def focal_sources_stable(focus):
    fields = [
        "fit_hash",
        "test_hash",
        "ess",
        "weight_max",
        "reason",
        "stage",
        "chosen",
        "path_hash",
    ]
    fields += [
        c
        for c in ("tree_hash", "source_x_hash", "source_y_hash", "test_x_hash")
        if c in focus
    ]
    return (
        focus.groupby([*SOURCE, "method"])[fields]
        .nunique(dropna=False)
        .eq(1)
        .all()
        .all()
    )


def risk_matrix(rows):
    return rows.pivot(
        index=["family", "seed"], columns="method", values="brier"
    ).sort_index()


def endpoints(focus):
    ep = focus[focus.world.isin(("s0", "s100"))].copy()
    ep["regret"] = ep.brier - ep.oracle_brier
    grouped = ep.groupby(["family", "seed", "method"])
    if not grouped.world.nunique().eq(2).all():
        raise ValueError("Endpoint pair missing")
    return {
        "worst_brier": grouped.brier.max().unstack(),
        "max_regret": grouped.regret.max().unstack(),
    }


def panels(raw):
    std = raw[raw.cohort == "standard"]
    yield "standard", "standard", "all", "test", "brier", risk_matrix(std), std
    for (c, f), part in raw.groupby(["cohort", "family"]):
        if c == "standard":
            yield f"standard/{f}", c, f, "test", "brier", risk_matrix(part), part
            continue
        for world, rows in part.groupby("world"):
            yield f"{f}/{world}", c, f, world, "brier", risk_matrix(rows), rows
        for metric, val in endpoints(part).items():
            yield f"{f}/{metric}", c, f, metric, metric, val, None


def bootstrap(idx, r):
    nf = idx.get_level_values("family").nunique()
    n = len(idx) // nf
    if (
        not idx.is_monotonic_increasing
        or not pd.Series(1, index=idx).groupby(level="family").sum().eq(n).all()
    ):
        raise ValueError("Expected ordered and equally weighted fixed families")
    draws = np.random.default_rng(103000 + r).integers(n, size=(10_000, nf, n))
    return np.stack([(draws == i).sum(-1) for i in range(n)], axis=-1).reshape(
        10_000, -1
    ) / len(idx)


def aggregate(raw, r):
    levels = []
    contrasts = []
    for group, cohort, family, world, metric, risk, rows in panels(raw):
        info = {
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
                rows.groupby("method")[[*METRICS, "oracle_brier"]].mean()
            )
        elif metric == "worst_brier":
            level["brier"] = level.risk
        levels.append(level.reset_index().assign(**info))
        w = bootstrap(risk.index, r)
        for ref in ("native", *CHEAP, "full", "cap2", "guarded"):
            diff = risk.sub(risk[ref], axis=0).drop(columns=ref)
            lo, hi = np.quantile(w @ diff.to_numpy(), [0.025, 0.975], axis=0)
            contrasts.append(
                pd.DataFrame({"delta": diff.mean(), "lo": lo, "hi": hi})
                .reset_index()
                .assign(**info, reference=ref)
            )
    return pd.concat(levels, ignore_index=True), pd.concat(contrasts, ignore_index=True)


def source_summary(raw):
    data = raw.drop_duplicates([*SOURCE, "method"]).copy()
    index = pd.MultiIndex.from_frame(data[SOURCE])
    native = data[data.method == "native"].set_index(SOURCE).reindex(index)
    for col, label in (
        ("fit_hash", "fit"),
        ("test_hash", "forecast"),
        ("chosen", "component"),
        ("path_hash", "path"),
        ("tree_hash", "tree"),
    ):
        if col in native:
            observed = data[col].notna().to_numpy() & native[col].notna().to_numpy()
            if label in {"component", "path", "tree"}:
                observed &= ~data.method.isin(CHEAP).to_numpy()
            data["changed_" + label] = np.where(
                observed, data[col].to_numpy() != native[col].to_numpy(), np.nan
            )
    if "n_fit" not in data:
        data["n_fit"] = np.where(data.family == "small128", 128, 512)
    data["ess_fraction"] = data.ess / data.n_fit
    data["group_ess_fraction"] = data.group_ess / data.groups
    result = []
    reasons = []
    groups = [("standard", data[data.cohort == "standard"])]
    groups.extend(
        ((f"standard/{f}" if c == "standard" else f), part)
        for (c, f), part in data.groupby(["cohort", "family"])
    )
    for group, part in groups:
        for method, rows in part.groupby("method"):
            desc = {
                "group": group,
                "method": method,
                "banks": len(rows),
                "mean_ess": rows.ess.mean(),
                "mean_group_ess": rows.group_ess.mean(),
                "mean_group_ess_fraction": rows.group_ess_fraction.mean(),
                "mean_ess_fraction": rows.ess_fraction.mean(),
                "min_ess_fraction": rows.ess_fraction.min(),
                "mean_max_weight": rows.weight_max.mean(),
                "max_weight": rows.weight_max.max(),
            }
            for col in (k for k in rows if k.startswith("changed_")):
                desc[col] = rows[col].sum(min_count=1)
            result.append(desc)
            counts = (
                rows.groupby(["policy", "stage", "reason"], dropna=False)
                .size()
                .rename("count")
                .reset_index()
            )
            reasons.append(counts.assign(group=group, method=method, banks=len(rows)))
    return pd.DataFrame(result), pd.concat(reasons, ignore_index=True)


def scene_groups(raw, r):
    """Standard scenarios with real training-scene IDs, separated from row-level banks."""
    std = raw[raw.cohort == "standard"].copy()
    std["ess_fraction"] = std.ess / std.n_fit
    std["group_ess_fraction"] = std.group_ess / std.groups
    families = {"grouped_rare_event", "shared_error_bloc", "rare_scene_specialist"}
    out = []
    for name, part in (
        ("grouped3", std[std.family.isin(families)]),
        ("other27", std[~std.family.isin(families)]),
    ):
        wide = risk_matrix(part)
        w = bootstrap(wide.index, r)
        d = wide.sub(wide.native, axis=0)
        lo, hi = np.quantile(w @ d.to_numpy(), [0.025, 0.975], axis=0)
        t = part.groupby("method")[
            [
                "brier",
                "ess",
                "ess_fraction",
                "groups",
                "group_ess",
                "group_ess_fraction",
            ]
        ].mean()
        out.append(
            t.join(pd.DataFrame({"delta_native": d.mean(), "lo": lo, "hi": hi}))
            .reset_index()
            .assign(group=name, banks=len(wide))
        )
    return pd.concat(out, ignore_index=True)


def select(raw):
    std = risk_matrix(raw[raw.cohort == "standard"])
    cases = risk_matrix(
        raw[
            (raw.cohort == "focus")
            & (raw.family == "case_mix")
            & (raw.world == "specialist80")
        ]
    )
    worst = endpoints(raw[(raw.cohort == "focus") & (raw.family == "mixed")])[
        "max_regret"
    ]
    std_effect = std.sub(std.native, axis=0).mean()
    case_effect = cases.sub(cases.native, axis=0).mean()
    s = {m: float(std_effect[m]) for m in CANDIDATES}
    c = {m: float(case_effect[m]) for m in CANDIDATES}
    score = {m: float(worst[m].mean()) for m in CANDIDATES}
    eligible = [m for m in CANDIDATES if s[m] <= 0.005 and c[m] <= 0.010]
    return {
        "chosen": min(eligible, key=score.get) if eligible else "native",
        "mixed_max_regret": score,
        "standard_delta_native": s,
        "case_mix_specialist80_delta_native": c,
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
    if blob != LOCK.read_bytes() or record != select(read(9)):
        raise RuntimeError("Selection differs from committed evidence")
    return record


def confirmation(contrasts):
    lock = committed_selection()
    m = lock["chosen"]
    table = contrasts[
        (contrasts.reference == "native") & (contrasts.method == m)
    ].set_index("group")
    checks = {}
    for name, limit in LIMITS.items():
        e = (
            {k: 0.0 for k in ("delta", "lo", "hi")}
            if m == "native"
            else table.loc[name, ["delta", "lo", "hi"]].astype(float).to_dict()
        )
        criterion = e["hi"] < 0 if name == "mixed/max_regret" else e["hi"] <= limit
        checks[name] = {**e, "upper_limit": limit, "passed": bool(criterion)}
    return {
        "chosen": m,
        "r09_sha256": lock["r09_sha256"],
        "checks": checks,
        "primary_pass": checks["mixed/max_regret"]["passed"],
        "all_guardrails_pass": all(
            c["passed"] for name, c in checks.items() if name != "mixed/max_regret"
        ),
        "all_pass": all(c["passed"] for c in checks.values()),
    }


def save(path, values):
    path.write_text(json.dumps(values, indent=2, allow_nan=False) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("round", type=int, choices=range(1, 11))
    r = p.parse_args().round
    raw = read(r)
    audit = validate(raw, r)
    support, weight_checks = audit_weights(raw, OUT / f"r{r:02d}-source.jsonl.gz")
    audit.update(weight_checks)
    with threadpool_limits(limits=1):
        levels, contrasts = aggregate(raw, r)
    weights, reasons = source_summary(raw)
    tables = {
        "levels": levels,
        "contrasts": contrasts,
        "families": levels[(levels.cohort == "standard") & (levels.family != "all")],
        "weights": weights,
        "support": support,
        "reasons": reasons,
    }
    tables["profile-pairs"] = profile_effects(raw, r)
    with threadpool_limits(limits=1):
        tables["scene-groups"] = scene_groups(raw, r)
    if r in (3, 4):
        tables["stage-pairs"] = stage_comparisons(raw, r)
    for name, frame in tables.items():
        frame.to_csv(OUT / f"r{r:02d}-{name}.csv", index=False)
    save(OUT / f"r{r:02d}-integrity.json", audit)
    if r == 9:
        decision = select(raw)
        if LOCK.exists() and json.loads(LOCK.read_text()) != decision:
            raise RuntimeError("Different selection already exists")
        save(LOCK, decision)
    if r == 10:
        save(OUT / "confirmation.json", confirmation(contrasts))


if __name__ == "__main__":
    main()

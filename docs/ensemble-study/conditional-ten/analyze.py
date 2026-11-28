"""Evaluate conditional weighting on paired sources and the published scenario taxonomy."""

import argparse
import json
import subprocess
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta
from threadpoolctl import threadpool_limits

from fusionbench.taxonomy import (
    APPLICATION_PROFILES,
    MECHANISM_GROUPS,
    PURPOSE_TAGS,
    scenario_weights,
)

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
LOCK = HERE / "selection.json"
CORE = ("native", "mean", "top_n", "lr", "guarded", "single", "half", "replicated")
REFS = ("native", "mean", "top_n", "lr", "guarded", "single")
CANDIDATES = ("half", "replicated", "single")
EXTRA = {2: ("raw",), 3: ("gate_sham",), 8: ("random_gate",)}
EXTRA_FOCUS = {
    2: ("varied",),
    3: ("reversed",),
    4: ("small128",),
    5: ("extreme",),
    6: ("narrow",),
    7: ("noisy_context",),
    8: ("homogeneous",),
    9: ("reversed",),
    10: (
        "varied",
        "reversed",
        "small128",
        "extreme",
        "narrow",
        "noisy_context",
        "homogeneous",
    ),
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
        raise ValueError("Duplicate bank scores or incorrect round")
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
        raise ValueError("Unknown cohort or missing outcome")
    if raw[["fit_hash", "test_hash"]].isna().any().any():
        raise ValueError(
            "Every fitted method requires source and target forecast identities"
        )
    std = raw[raw.cohort == "standard"]
    focus = raw[raw.cohort == "focus"]
    if set(std.family) != set(scenario_weights()) or set(std.world) != {"test"}:
        raise ValueError("Missing standard-v6 scenario")
    if set(focus.family) != {"mixed", "case_mix", *EXTRA_FOCUS.get(r, ())} or set(
        focus.world
    ) != {"s0", "s50", "s100", "specialist80"}:
        raise ValueError("Unexpected focal coverage")
    if not focus.groupby([*SOURCE, "method"]).world.nunique().eq(4).all():
        raise ValueError("Missing paired focal world")
    for (cohort, family), rows in raw.groupby(["cohort", "family"]):
        n = (
            {9: 4, 10: 8}.get(r, 2)
            if cohort == "standard"
            else {9: 12, 10: 24}.get(r, 6)
        )
        start = 1140000 + 1000 * r + (400 if cohort == "standard" else 100)
        if set(rows.seed) != set(range(start, start + n)):
            raise ValueError(f"Incorrect source seeds for {cohort}/{family}")
    if (
        not std.oracle_brier.isna().all()
        or focus.oracle_brier.isna().any()
        or not focus.groupby(KEY).oracle_brier.nunique().eq(1).all()
    ):
        raise ValueError("Missing or inconsistent focal oracle risk")
    if (focus.brier - focus.oracle_brier).min() < -2e-12:
        raise ValueError("Expected risk below Bayes risk")
    for group, cols in [
        (
            [*SOURCE, "method"],
            [
                "gate_active",
                "gate_provider",
                "applied",
                "weight_hash",
                "fit_hash",
                "test_hash",
                "ess",
                "policy",
            ],
        ),
        (
            KEY,
            ["source_x_hash", "source_y_hash", "test_x_hash", "target_hash", "n_fit"],
        ),
    ]:
        if (
            not raw.groupby(group)[[c for c in cols if c in raw]]
            .nunique(dropna=False)
            .eq(1)
            .all()
            .all()
        ):
            raise ValueError(
                "Source choices, forecasts or data identities changed between matched rows"
            )
    ep = focus.pivot(index=[*SOURCE, "method"], columns="world", values="brier")
    err = float((ep.s50 - 0.5 * (ep.s0 + ep.s100)).abs().max())
    if err > 3e-12:
        raise ValueError(
            "Intermediate focal risk does not integrate over the endpoint mixture"
        )
    return {
        "sources": len(raw[SOURCE].drop_duplicates()),
        "target_conditions": len(raw[KEY].drop_duplicates()),
        "score_rows": len(raw),
        "max_endpoint_interpolation_error": err,
    }


def standard_weights():
    yield "standard", scenario_weights()
    for group in MECHANISM_GROUPS:
        yield "cluster/" + group, scenario_weights(groups=[group])
    for tag in PURPOSE_TAGS:
        yield "purpose/" + tag, scenario_weights(purposes=[tag])
    yield (
        "equal_cluster",
        scenario_weights(group_weights={g: 1 for g in MECHANISM_GROUPS}),
    )
    for name, weights in APPLICATION_PROFILES.items():
        yield name, scenario_weights(group_weights=weights)


def risk_matrix(rows):
    return rows.pivot(
        index=["family", "seed"], columns="method", values="brier"
    ).sort_index()


def endpoints(focus):
    ep = focus[focus.world.isin(("s0", "s100"))].copy()
    ep["regret"] = ep.brier - ep.oracle_brier
    by = ep.groupby(["family", "seed", "method"])
    if not by.world.nunique().eq(2).all():
        raise ValueError("A source is missing an endpoint")
    return {
        "max_regret": by.regret.max().unstack(),
        "worst_brier": by.brier.max().unstack(),
    }


def panels(raw):
    std = raw[raw.cohort == "standard"]
    for name, mass in standard_weights():
        rows = std[std.family.isin(mass)]
        yield name, "standard", "all", "test", "brier", mass, risk_matrix(rows), rows
    for (cohort, family), part in raw.groupby(["cohort", "family"]):
        mass = {family: 1.0}
        if cohort == "standard":
            yield (
                "standard/" + family,
                cohort,
                family,
                "test",
                "brier",
                mass,
                risk_matrix(part),
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
                    mass,
                    risk_matrix(rows),
                    rows,
                )
            for metric, risks in endpoints(part).items():
                yield (
                    family + "/" + metric,
                    cohort,
                    family,
                    metric,
                    metric,
                    mass,
                    risks,
                    None,
                )


def bootstrap(index, shares, r):
    names = index.get_level_values("family").unique()
    nf = len(names)
    n = len(index) // nf
    if (
        not index.is_monotonic_increasing
        or not pd.Series(1, index=index).groupby(level="family").sum().eq(n).all()
    ):
        raise ValueError("Expected sorted balanced source panel")
    mass = np.array([shares[k] for k in names])
    if not np.all(mass > 0) or not np.isclose(mass.sum(), 1, atol=1e-12):
        raise ValueError("Positive family weights must sum to one")
    draws = np.random.default_rng(113000 + r).integers(n, size=(10_000, nf, n))
    freq = np.stack([(draws == i).sum(axis=2) for i in range(n)], axis=2)
    return (freq * (mass / n)[None, :, None]).reshape(10_000, -1), np.repeat(
        mass / n, n
    )


def estimates(raw, r):
    levels, contrasts = [], []
    for group, c, f, world, metric, shares, risk, rows in panels(raw):
        meta = {
            "group": group,
            "cohort": c,
            "family": f,
            "world": world,
            "metric": metric,
            "families": len(shares),
            "banks": len(risk),
        }
        w, point = bootstrap(risk.index, shares, r)
        score = pd.Series(
            point @ risk.to_numpy(), index=risk.columns, name="risk"
        ).to_frame()
        if rows is not None:
            for name in (*METRICS, "oracle_brier"):
                wide = rows.pivot(
                    index=["family", "seed"], columns="method", values=name
                ).reindex(index=risk.index, columns=risk.columns)
                score[name] = point @ wide.to_numpy()
        elif metric == "worst_brier":
            score["brier"] = score.risk
        levels.append(score.reset_index().assign(**meta))
        for ref in REFS:
            diff = risk.sub(risk[ref], axis=0).drop(columns=ref)
            low, high = np.quantile(w @ diff.to_numpy(), [0.025, 0.975], axis=0)
            contrasts.append(
                pd.DataFrame(
                    {
                        "method": diff.columns,
                        "delta": point @ diff.to_numpy(),
                        "lo": low,
                        "hi": high,
                    }
                ).assign(reference=ref, **meta)
            )
    return pd.concat(levels, ignore_index=True), pd.concat(contrasts, ignore_index=True)


def exact_bounds(k, n):
    if not n:
        return None, None
    return (
        0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1),
        1.0 if k == n else beta.ppf(0.975, k + 1, n - k),
    )


def gate_summary(raw, r):
    gates = (
        raw[~raw.method.isin(("native", "mean", "top_n", "lr", "guarded"))]
        .drop_duplicates([*SOURCE, "method"])
        .copy()
    )
    # The base policies use all providers; the identity of a diagnosed provider is evaluated retrospectively.
    groups = [("standard", gates[gates.cohort == "standard"])]
    groups.extend(
        ("cluster/" + g, gates[(gates.cohort == "standard") & gates.family.isin(names)])
        for g, names in MECHANISM_GROUPS.items()
    )
    groups.extend(
        ((f if c == "focus" else "standard/" + f), rows)
        for (c, f), rows in gates.groupby(["cohort", "family"])
    )
    records = []
    rates = []
    paired = []
    for group, frame in groups:
        for method, rows in frame.groupby("method"):
            active = rows.gate_active.eq(True)
            applied = rows.applied.eq(True)
            labeled = rows.true_shortcut.notna()
            shortcut = active & rows.gate_provider.eq(rows.true_shortcut)
            specialist = active & rows.gate_provider.eq(rows.true_specialist)
            record = {
                "group": group,
                "method": method,
                "banks": len(rows),
                "active": int(active.sum()),
                "applied": int(applied.sum()),
                "mean_ess": rows.ess.mean(),
                "other_or_unassigned": int((active & ~shortcut & ~specialist).sum()),
                "shortcut": int(shortcut.sum()) if labeled.any() else None,
                "specialist": int(specialist.sum()) if labeled.any() else None,
            }
            records.append(record)
            # Source seeds within a single focal design are independent Bernoulli trials.
            # Pooling the 30 fixed families would impose exchangeability across different generators.
            if rows.cohort.eq("focus").all():
                events = {
                    "activation": active,
                    "application": applied,
                    "correct_shortcut": shortcut,
                    "specialist_flag": specialist,
                    "abstention": ~active,
                }
                for event, y in events.items():
                    count = int(y.sum())
                    lo, hi = exact_bounds(count, len(y))
                    rates.append(
                        {
                            "group": group,
                            "method": method,
                            "event": event,
                            "count": count,
                            "banks": len(y),
                            "rate": count / len(y),
                            "lo": lo,
                            "hi": hi,
                        }
                    )
    focal = gates[gates.cohort == "focus"]
    for m, rows in focal.groupby("method"):
        for y, values in (
            ("activation", rows.gate_active.eq(True)),
            ("application", rows.applied.eq(True)),
            (
                "correct_shortcut",
                rows.gate_active.eq(True) & rows.gate_provider.eq(rows.true_shortcut),
            ),
        ):
            panel = (
                rows.assign(event=values.astype(float))
                .pivot(index="seed", columns="family", values="event")
                .sort_index()
            )
            if panel[["mixed", "case_mix"]].isna().any().any():
                raise ValueError("Unpaired mixed and case mix gate decisions")
            d = (panel.mixed - panel.case_mix).to_numpy()
            idx = np.random.default_rng(113000 + r).integers(
                len(d), size=(10_000, len(d))
            )
            lo, hi = np.quantile(d[idx].mean(1), [0.025, 0.975])
            paired.append(
                {
                    "method": m,
                    "event": y,
                    "mixed": panel.mixed.mean(),
                    "case_mix": panel.case_mix.mean(),
                    "delta": d.mean(),
                    "lo": lo,
                    "hi": hi,
                    "paired_sources": len(d),
                }
            )
    return pd.DataFrame(records), pd.DataFrame(rates), pd.DataFrame(paired)


def decision_effects(raw):
    """Decompose achieved risk changes by actual application, one row per source."""
    std = raw[raw.cohort == "standard"]
    focus = raw[raw.cohort == "focus"]
    metrics = [("standard", risk_matrix(std))]
    for group, families in MECHANISM_GROUPS.items():
        metrics.append(
            ("cluster/" + group, risk_matrix(std[std.family.isin(families)]))
        )
    for family, rows in focus.groupby("family"):
        metrics.append((family + "/max_regret", endpoints(rows)["max_regret"]))
        for world, part in rows.groupby("world"):
            metrics.append((family + "/" + world, risk_matrix(part)))
    per_source = raw.drop_duplicates([*SOURCE, "method"])
    d = per_source.set_index(["method", "family", "seed"]).sort_index()
    output = []
    for group, risk in metrics:
        for method in sorted(set(risk.columns) - {"mean", "top_n", "lr", "native"}):
            target = d.loc[method].reindex(risk.index)
            base = d.loc["native"].reindex(risk.index)
            diff = risk[method] - risk.native
            applied = target.applied
            if not applied.any():
                active_delta = None
            else:
                active_delta = float(diff[applied].mean())
            if not diff[~applied].eq(0).all():
                raise ValueError("An unweighted native fit changed measured risk")
            changed = {
                field.removesuffix("_hash") + "_changed": int(
                    target[field].ne(base[field]).sum()
                )
                for field in ("path_hash", "tree_hash", "fit_hash", "test_hash")
            }
            if changed["test_changed"] > int(applied.sum()):
                raise ValueError("Unweighted fit changed test probabilities")
            output.append(
                {
                    "group": group,
                    "method": method,
                    "banks": len(diff),
                    "active": int(target.gate_active.sum()),
                    "applied": int(applied.sum()),
                    "improved_if_applied": int((diff[applied] < -1e-12).sum()),
                    "worsened_if_applied": int((diff[applied] > 1e-12).sum()),
                    "unchanged_if_applied": int((diff[applied].abs() <= 1e-12).sum()),
                    "mean_delta_if_applied": active_delta,
                    "mean_delta": float(diff.mean()),
                    **changed,
                }
            )
    return pd.DataFrame(output)


def select(raw):
    std = risk_matrix(raw[raw.cohort == "standard"])
    focus = raw[raw.cohort == "focus"]
    cm = risk_matrix(
        focus[(focus.family == "case_mix") & (focus.world == "specialist80")]
    )
    mix = focus[focus.family == "mixed"]
    s0 = risk_matrix(mix[mix.world == "s0"])
    scores = endpoints(mix)["max_regret"].mean()

    def effect(panel):
        d = panel.sub(panel.native, axis=0).mean()
        return {m: float(d[m]) for m in CANDIDATES}

    a, b, c = effect(std), effect(cm), effect(s0)
    objective = {m: float(scores[m]) for m in CANDIDATES}
    eligible = [
        m for m in CANDIDATES if a[m] <= 0.005 and b[m] <= 0.010 and c[m] <= 0.010
    ]
    return {
        "chosen": min(eligible, key=objective.get) if eligible else "native",
        "eligible": eligible,
        "mixed_max_regret": objective,
        "standard_delta_native": a,
        "case_mix_specialist80_delta_native": b,
        "mixed_s0_delta_native": c,
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
    decision = json.loads(blob)
    if blob != LOCK.read_bytes() or decision != select(read(9)):
        raise RuntimeError("R9 selection differs from its committed evidence")
    return decision


def confirm(contrasts):
    lock = committed_selection()
    method = lock["chosen"]
    rows = contrasts[contrasts.method == method]
    table = rows[rows.reference == "native"].set_index("group")
    checks = {}
    for group, limit in LIMITS.items():
        vals = (
            {k: 0.0 for k in ("delta", "lo", "hi")}
            if method == "native"
            else table.loc[group, ["delta", "lo", "hi"]].astype(float).to_dict()
        )
        passed = vals["hi"] < 0 if group == "mixed/max_regret" else vals["hi"] <= limit
        checks[group] = {**vals, "upper_limit": limit, "passed": bool(passed)}
    groups = [
        "case_mix/max_regret",
        "case_mix/specialist80",
        "mixed/max_regret",
        *["cluster/" + g for g in MECHANISM_GROUPS],
    ]
    versus = rows[(rows.reference == "guarded") & rows.group.isin(groups)].set_index(
        "group"
    )
    return {
        "chosen": method,
        "r09_sha256": lock["r09_sha256"],
        "checks": checks,
        "primary_pass": checks["mixed/max_regret"]["passed"],
        "all_guardrails_pass": all(
            t["passed"] for g, t in checks.items() if g != "mixed/max_regret"
        ),
        "all_pass": all(t["passed"] for t in checks.values()),
        "secondary_vs_guarded": versus[["delta", "lo", "hi"]].to_dict("index"),
    }


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("round", type=int, choices=range(1, 11))
    r = parser.parse_args().round
    raw = read(r)
    audit = validate(raw, r)
    with threadpool_limits(limits=1):
        levels, contrasts = estimates(raw, r)
        gates, rates, paired = gate_summary(raw, r)
    table = {
        "levels": levels,
        "contrasts": contrasts,
        "gates": gates,
        "decision-effects": decision_effects(raw),
        "rates": rates,
        "paired-gates": paired,
        "families": levels[(levels.cohort == "standard") & (levels.family != "all")],
    }
    for name, data in table.items():
        data.to_csv(OUT / f"r{r:02d}-{name}.csv", index=False)
    save(OUT / f"r{r:02d}-integrity.json", audit)
    if r == 9:
        record = select(raw)
        if LOCK.exists() and record != json.loads(LOCK.read_text()):
            raise RuntimeError("R9 selection already exists with a different input")
        save(LOCK, record)
    if r == 10:
        save(OUT / "confirmation.json", confirm(contrasts))


if __name__ == "__main__":
    main()

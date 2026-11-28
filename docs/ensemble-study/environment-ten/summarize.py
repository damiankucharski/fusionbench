"""Paired environment-study estimates; smaller Brier is better."""

import argparse
import json
import subprocess
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
from paired_designs import design_pairs
from paired_stats import intervals, robust_panel, weights
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
LOCK = HERE / "selection.json"
CANDIDATES = ("adjusted", "strict", "raw")  # Predeclared tie order.
REFS = ("native", "mean", "top_n", "lr")
KEY = ["cohort", "family", "seed", "world"]
METRICS = ["brier", "macro_f1", "providers"]


def read(r):
    return pd.read_csv(OUT / f"r{r:02d}.csv", float_precision="round_trip")


def panels(frame):
    standard = frame[frame.cohort == "standard"]
    if standard.family.nunique() != 30 or set(standard.world) != {"test"}:
        raise ValueError(
            "The standard panel requires all 30 families on their test sets"
        )
    yield (
        {"group": "standard", "cohort": "standard", "family": "all", "world": "test"},
        standard,
    )
    focus = frame[frame.cohort == "focus"]
    for (family, world), rows in focus.groupby(["family", "world"]):
        yield (
            {
                "group": f"{family}/{world}",
                "cohort": "focus",
                "family": family,
                "world": world,
            },
            rows,
        )


def validate(raw, r):
    if set(raw["round"]) != {r} or raw.duplicated([*KEY, "method"]).any():
        raise ValueError("Wrong round or duplicate method risk per bank")
    if set(raw.cohort) != {"standard", "focus"} or not set(REFS) <= set(raw.method):
        raise ValueError("Missing cohort or cheap comparator")
    intended = set(REFS) | set(CANDIDATES) | ({"sham"} if r >= 2 else set())
    if set(raw.method) != intended:
        raise ValueError("Unexpected decision rules")
    extra = {
        5: {"four_env"},
        6: {"narrow"},
        7: {"small128"},
        8: {"original_scale", "noisy_context"},
        10: {"reversed", "narrow", "small128"},
    }
    focals = {"homogeneous", "varied", "case_mix", "mixed"} | extra.get(r, set())
    worlds = (
        {"s0", "s50", "s100"}
        | ({"specialist80"} if r in (3, 4, 9, 10) else set())
        | ({"s25", "s75"} if r == 10 else set())
    )
    focus = raw[raw.cohort == "focus"]
    if set(focus.family) != focals or set(focus.world) != worlds:
        raise ValueError("Incomplete focal designs or target laws")
    if (
        not focus.groupby(["family", "seed", "method"])
        .world.nunique()
        .eq(len(worlds))
        .all()
    ):
        raise ValueError("Missing paired focal world")
    for (cohort, family), rows in raw.groupby(["cohort", "family"]):
        if cohort == "standard":
            offset, n = 400, (8 if r == 10 else 4 if r == 9 else 2)
        else:
            offset, n = 100, (24 if r == 10 else 12 if r == 9 else 8)
        start = 840000 + 1000 * r + offset
        if set(rows.seed) != set(range(start, start + n)):
            raise ValueError(f"Incomplete or wrong source draws in {cohort}/{family}")
    reps = raw.groupby(KEY).method.nunique()
    if not reps.eq(len(intended)).all():
        raise ValueError("All methods must share all banks")
    if not np.isfinite(raw[METRICS]).all().all():
        raise ValueError("Missing or non-finite primary or secondary outcome")
    if raw[["fit_hash", "test_hash"]].isna().any().any():
        raise ValueError("Missing prediction identities")
    dropped = raw["drop"].dropna()
    if (dropped < 0).any() or (dropped != dropped.astype(int)).any():
        raise ValueError("A drop must name a provider")
    if raw[raw.method.isin(REFS)]["drop"].notna().any():
        raise ValueError(
            "The reference methods must not be assigned diagnostic exclusions"
        )
    source = raw.groupby(["cohort", "family", "seed", "method"], dropna=False)
    for field in ("drop", "fit_hash", "n_fit", "test_x_hash", "test_hash"):
        if source[field].nunique(dropna=False).gt(1).any():
            raise ValueError(
                f"A source decision or forecast changed across target worlds: {field}"
            )
    # The filtering rule delegates to the native baseline on the original bank when it abstains.
    joined = raw[~raw.method.isin(REFS) & raw["drop"].isna()].merge(
        raw[raw.method == "native"],
        on=KEY,
        suffixes=("", "_native"),
        validate="many_to_one",
    )
    for field in ("fit_hash", "test_hash", *METRICS):
        if not joined[field].eq(joined[field + "_native"]).all():
            raise ValueError(f"Abstaining rule differs from native on {field}")
    return {
        "score_rows": len(raw),
        "bank_world_conditions": len(reps),
        "abstention_rows_checked": len(joined),
    }


def estimates(raw, r):
    levels, contrasts, drops = [], [], []
    families = (
        raw.groupby(["cohort", "family", "world", "method"])[METRICS]
        .mean()
        .reset_index()
    )
    for labels, part in panels(raw):
        wide = part.pivot(
            index=["family", "seed"], columns="method", values="brier"
        ).sort_index()
        w = weights(wide.index, r)
        levels.append(
            part.groupby("method")[METRICS]
            .mean()
            .reset_index()
            .assign(**labels, families=part.family.nunique(), banks=len(wide))
        )
        for ref in (*REFS, *CANDIDATES, "sham"):
            if ref not in wide:
                continue
            d = wide.sub(wide[ref], axis=0).drop(columns=ref)
            contrasts.append(
                intervals(w, d)
                .reset_index()
                .assign(**labels, reference=ref, banks=len(wide))
            )
        for method, rows in part.groupby("method"):
            chosen = rows["drop"]
            removed = chosen.notna()
            special = rows.true_specialist.notna()
            shortcut = rows.true_shortcut.notna()

            drops.append(
                {
                    **labels,
                    "method": method,
                    "banks": len(rows),
                    "dropped": int(removed.sum()),
                    "drop_rate": float(removed.mean()),
                    "shortcut_present": int(shortcut.sum()),
                    "specialist_present": int(special.sum()),
                    "shortcut_drops": int(chosen.eq(rows.true_shortcut).sum())
                    if shortcut.any()
                    else np.nan,
                    "specialist_drops": int(chosen.eq(rows.true_specialist).sum())
                    if special.any()
                    else np.nan,
                    "shortcut_drop_rate": float(
                        chosen[shortcut].eq(rows.true_shortcut[shortcut]).mean()
                    )
                    if shortcut.any()
                    else np.nan,
                    "specialist_drop_rate": float(
                        chosen[special].eq(rows.true_specialist[special]).mean()
                    )
                    if special.any()
                    else np.nan,
                }
            )
    return {
        "levels": pd.concat(levels),
        "contrasts": pd.concat(contrasts),
        "families": families,
        "drops": pd.DataFrame(drops),
    }


def robust_summaries(out, raw, r):
    for family in sorted(set(raw.loc[raw.cohort == "focus", "family"])):
        risks = robust_panel(raw, family)
        flags = {
            "group": family + "/worst_endpoint",
            "cohort": "focus",
            "family": family,
            "world": "worst_endpoint",
        }
        level = (
            risks.mean()
            .rename("brier")
            .reset_index()
            .assign(**flags, families=1, banks=len(risks))
        )
        out["levels"] = pd.concat([out["levels"], level], ignore_index=True)
        w = weights(risks.index, r)
        items = []
        for ref in risks:
            d = risks.sub(risks[ref], axis=0).drop(columns=ref)
            items.append(
                intervals(w, d)
                .reset_index()
                .assign(**flags, reference=ref, banks=len(risks))
            )
        out["contrasts"] = pd.concat([out["contrasts"], *items], ignore_index=True)


def drop_comparisons(raw, r):
    """One decision per source regardless of the number of scored target worlds."""
    source = raw.drop_duplicates(["cohort", "family", "seed", "method"])
    groups = [("standard", source[source.cohort == "standard"])]
    groups.extend(source[source.cohort == "focus"].groupby("family"))
    transitions, rates = [], []
    for group, rows in groups:
        index = rows.pivot(
            index=["family", "seed"], columns="method", values="drop"
        ).sort_index()
        w = weights(index.index, r)
        base = (
            rows[rows.method == "native"].set_index(["family", "seed"]).loc[index.index]
        )
        labels = {}
        for m in (*CANDIDATES, "sham"):
            if m not in index:
                continue
            role = pd.Series("other", index=index.index)
            if group == "standard":
                role[:] = "unlabelled"
            role[index[m].eq(base.true_shortcut)] = "shortcut"
            role[index[m].eq(base.true_specialist)] = "specialist"
            role[index[m].isna()] = "abstain"
            labels[m] = role
        for a, b in (("adjusted", "raw"), ("strict", "adjusted"), ("adjusted", "sham")):
            if b not in labels:
                continue
            transitions.extend(
                {
                    "group": group,
                    "method": a,
                    "reference": b,
                    "reference_role": ref,
                    "selected_role": target,
                    "count": int(count),
                    "banks": len(base),
                }
                for (ref, target), count in pd.DataFrame(
                    {"ref": labels[b], "target": labels[a]}
                )
                .value_counts()
                .items()
            )
            contrasts = {
                "any": index[a].notna().astype(float) - index[b].notna().astype(float)
            }
            for role in ("shortcut", "specialist"):
                if group != "standard":
                    contrasts[role] = labels[a].eq(role).astype(float) - labels[b].eq(
                        role
                    ).astype(float)
            effects = intervals(w, pd.DataFrame(contrasts))
            rates.extend(
                {
                    "group": group,
                    "method": a,
                    "reference": b,
                    "event": event,
                    "banks": len(base),
                    **row,
                }
                for event, row in effects.to_dict("index").items()
            )
    return pd.DataFrame(transitions), pd.DataFrame(rates)


def selection():
    raw = read(9)
    risks = robust_panel(raw, "mixed")
    if len(risks) != 12:
        raise ValueError("Selection expects 12 fresh mixed-mechanism draws")
    scores = {m: float(risks[m].mean()) for m in CANDIDATES}
    return {
        "chosen": min(CANDIDATES, key=scores.get),
        "mixed_mean_worst_endpoint_brier": scores,
        "r09_sha256": sha256((OUT / "r09.csv").read_bytes()).hexdigest(),
    }


def committed_selection():
    root = Path(
        subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], cwd=HERE, text=True
        ).strip()
    )
    raw = subprocess.check_output(
        ["git", "show", f"HEAD:{LOCK.relative_to(root)}"], cwd=root
    )
    record = json.loads(raw)
    if LOCK.read_bytes() != raw or record != selection():
        raise RuntimeError("Round-nine decision or evidence differs from its commit")
    return record


def confirm(contrasts):
    record = committed_selection()
    method = record["chosen"]
    table = contrasts[
        (contrasts.method == method) & (contrasts.reference == "native")
    ].set_index("group")
    outcomes = {}
    for key, limit in [
        ("mixed/worst_endpoint", 0.0),
        ("case_mix/specialist80", 0.005),
        ("standard", 0.005),
        ("mixed/s100", 0.005),
    ]:
        row = table.loc[key, ["delta", "lo", "hi"]].astype(float).to_dict()
        passed = row["hi"] < 0 if key == "mixed/worst_endpoint" else row["hi"] <= limit
        outcomes[key] = {**row, "upper_limit": limit, "passed": bool(passed)}
    return {
        "chosen": method,
        "r09_sha256": record["r09_sha256"],
        "checks": outcomes,
        "primary_pass": outcomes["mixed/worst_endpoint"]["passed"],
        "all_guardrails_pass": all(
            o["passed"]
            for name, o in outcomes.items()
            if name != "mixed/worst_endpoint"
        ),
        "all_pass": all(o["passed"] for o in outcomes.values()),
        "unseen_reversed_worst_endpoint": table.loc[
            "reversed/worst_endpoint", ["delta", "lo", "hi"]
        ]
        .astype(float)
        .to_dict(),
    }


def save_json(path, record):
    path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("round", type=int, choices=range(1, 11))
    r = parser.parse_args().round
    raw = read(r)
    checks = validate(raw, r)
    with threadpool_limits(limits=1):
        out = estimates(raw, r)
        robust_summaries(out, raw, r)
        out["drop-pairs"], out["drop-effects"] = drop_comparisons(raw, r)
        out["design-pairs"] = design_pairs(raw, r)
    for name, df in out.items():
        df.to_csv(OUT / f"r{r:02d}-{name}.csv", index=False)
    save_json(OUT / f"r{r:02d}-integrity.json", checks)
    if r == 9:
        new = selection()
        if LOCK.exists() and new != json.loads(LOCK.read_text()):
            raise RuntimeError("A different selection already exists")
        save_json(LOCK, new)
    if r == 10:
        save_json(OUT / "confirmation.json", confirm(out["contrasts"]))


if __name__ == "__main__":
    main()

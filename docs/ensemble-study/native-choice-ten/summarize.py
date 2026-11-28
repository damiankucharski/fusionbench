"""Paired scenario summaries, matched interventions and a committed round-nine decision."""

import argparse
import json
import subprocess
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
LOCK = HERE / "selection.json"
KEYS = ["family", "seed"]
HELD = {
    "grouped_rare_event",
    "shared_error_bloc",
    "rare_scene_specialist",
    "cloned_shortcut",
    "parity",
    "similar_classes_38",
}
CANDIDATES = ("cv", "one_se", "paired", "sign", "stress", "augmented")
CHEAP = ("mean", "top_n", "lr")
REFS = ("native", *CHEAP)
ARMS = {
    1: ("apparent",),
    2: ("cv",),
    3: ("cv", "one_se"),
    4: ("cv", "paired", "sign"),
    5: ("cv", "paired", "stress"),
    6: ("augmented",),
}
CONTROLS = {"weak", "strong", "scene16", "scene128"}
SHORTCUTS = ("shortcut_100", "shortcut_050", "shortcut_000")
METRICS = ("brier", "fit_brier", "macro_f1", "providers")


def table(r):
    return pd.read_csv(OUT / f"r{r:02d}.csv", float_precision="round_trip")


def panels(primary):
    standard = primary[primary.cohort == "standard"]
    names = set(standard.family)
    if len(names) != 30 or not HELD <= names:
        raise ValueError("The standard panel must contain all 30 families")
    yield "standard", standard
    yield "seen24", standard[~standard.family.isin(HELD)]
    yield "held6", standard[standard.family.isin(HELD)]
    controls = primary[primary.cohort == "control"]
    wanted = CONTROLS | (set(SHORTCUTS) if set(primary["round"]) <= {5, 10} else set())
    if set(controls.family) != wanted:
        raise ValueError("Unexpected or missing control banks")
    yield from controls.groupby("family")


def weights(index, r):
    counts = pd.Series(1, index=index).groupby(level="family").sum()
    if counts.nunique() != 1 or list(index) != sorted(index):
        raise ValueError("Each family must have the same number of sorted draws")
    nf, n = len(counts), counts.iloc[0]
    draws = np.random.default_rng(81000 + r).integers(n, size=(10_000, nf, n))
    freq = np.stack([(draws == i).sum(axis=-1) for i in range(n)], axis=-1)
    return freq.reshape(10_000, nf * n) / (nf * n)


def interval(w, matrix):
    lo, hi = np.quantile(w @ matrix.to_numpy(), [0.025, 0.975], axis=0)
    return pd.DataFrame({"delta": matrix.mean(), "lo": lo, "hi": hi})


def validate_rows(raw, r):
    key = ["variant", "cohort", *KEYS]
    if set(raw["round"]) != {r} or raw.duplicated([*key, "method"]).any():
        raise ValueError("Duplicate or wrongly labeled bank scores")
    if (
        raw[[*METRICS, "n_fit", "n_test", "fit_hash", "test_hash", "action"]]
        .isna()
        .any()
        .any()
    ):
        raise ValueError("Missing score, prediction identity or action")
    if not np.isfinite(raw[list(METRICS)].to_numpy()).all():
        raise ValueError("Non-finite score")
    variants = {7: {"large", "primary"}, 8: {"seed0", "primary"}}.get(r, {"primary"})
    if set(raw.variant) != variants:
        raise ValueError("Unexpected round variants")
    arms = set(ARMS.get(r, CANDIDATES)) | set(REFS)
    size = raw.groupby(key).method.nunique()
    if set(raw.method) != arms or not size.eq(len(arms)).all():
        raise ValueError(f"Incomplete method-by-bank panel for round {r}")
    metadata = [
        "n_fit",
        "n_test",
        "source_x_hash",
        "source_y_hash",
        "test_x_hash",
        "target_hash",
    ]
    if not raw.groupby(key)[metadata].nunique().eq(1).all().all():
        raise ValueError("Compared methods used different banks")

    for (variant, cohort, family), group in raw.groupby(
        ["variant", "cohort", "family"]
    ):
        base = 720000 + 1000 * r
        if cohort == "standard":
            offset, expected = 400, 10 if r == 10 else 5 if r == 9 else 3
        else:
            offset = (
                300
                if family in SHORTCUTS
                else 200
                if family in {"weak", "strong"}
                else 100
            )
            expected = 16 if r == 10 else 8
        if set(group.seed) != set(range(base + offset, base + offset + expected)):
            raise ValueError(
                f"Missing or incorrect draw in {variant}/{cohort}/{family}"
            )

    # Standalone arithmetic uses NumPy summation. The optional augmented search
    # can realize the same arithmetic operation using native tree reductions.
    policies = raw[~raw.method.isin(REFS)]
    matched = policies.merge(
        raw[raw.method.isin(REFS)],
        left_on=[*key, "action"],
        right_on=[*key, "method"],
        how="left",
        validate="many_to_one",
        suffixes=("", "_base"),
    )
    if (
        not matched.action.isin(("native", "mean", "top_n")).all()
        or matched.method_base.isna().any()
    ):
        raise ValueError("A deployable rule selected an unknown action")
    compiled = (matched.method == "augmented") & matched.action.isin(("mean", "top_n"))
    for label in ["fit_hash", "test_hash", *METRICS]:
        both = matched.loc[~compiled, [label, label + "_base"]]
        if not both.iloc[:, 0].eq(both.iloc[:, 1]).all():
            raise ValueError(f"Routed row disagrees with the claimed action on {label}")
    for label in ("brier", "fit_brier", "providers"):
        atol = 0 if label == "providers" else 1e-6
        if not np.allclose(
            matched.loc[compiled, label],
            matched.loc[compiled, label + "_base"],
            rtol=0,
            atol=atol,
        ):
            raise ValueError(
                f"Native arithmetic tree disagrees with {label} in the selected action"
            )
    return {
        "bank_conditions": len(size),
        "exact_action_rows": int((~compiled).sum()),
        "compiled_arithmetic_action_rows": int(compiled.sum()),
    }


def summarize(primary, r):
    families = primary.groupby(["cohort", "family", "method"])[list(METRICS)].mean()
    families["delta_native"] = families.brier.sub(
        families.xs("native", level="method")
        .brier.reindex(families.index.droplevel("method"))
        .to_numpy()
    )
    levels, contrasts, choices = [], [], []
    for group, rows in panels(primary):
        wide = rows.pivot(index=KEYS, columns="method", values="brier").sort_index()
        w = weights(wide.index, r)
        means = rows.groupby("method")[list(METRICS)].mean()
        levels.append(
            means.assign(
                group=group, families=rows.family.nunique(), banks=len(wide)
            ).reset_index()
        )
        for ref in REFS:
            part = interval(w, wide.sub(wide[ref], axis=0)).drop(index=ref)
            contrasts.append(
                part.assign(group=group, reference=ref, units=len(wide)).reset_index()
            )
        policies = rows[~rows.method.isin(REFS)]
        lookup = rows[rows.method == "native"].set_index(KEYS).test_hash
        policies = policies.copy()
        policies["different_native_forecast"] = (
            policies.test_hash.to_numpy()
            != lookup.reindex(pd.MultiIndex.from_frame(policies[KEYS])).to_numpy()
        )
        action_counts = (
            policies.groupby(["method", "action"])
            .agg(
                banks=("seed", "size"),
                different_native_forecast=("different_native_forecast", "sum"),
            )
            .reset_index()
        )
        action_counts["fraction"] = action_counts.banks / len(wide)
        choices.append(action_counts.assign(group=group))
    return {
        "families": families.reset_index(),
        "levels": pd.concat(levels),
        "contrasts": pd.concat(contrasts),
        "actions": pd.concat(choices),
    }


def switch_diagnostics(primary):
    """Count procedural switches, forecast changes and their contribution to held-out loss."""
    records = []
    for group, rows in panels(primary):
        raw = rows.pivot(index=KEYS, columns="method", values="brier")
        ident = rows.pivot(index=KEYS, columns="method", values="test_hash")
        actions = rows.pivot(index=KEYS, columns="method", values="action")
        oracle = raw[["mean", "top_n", "native"]].min(axis=1)
        for rule in CANDIDATES:
            changed = ident[rule] != ident.native
            switched = actions[rule] != "native"
            differences = raw[rule] - raw.native
            d = differences[switched]
            records.append(
                {
                    "group": group,
                    "method": rule,
                    "banks": len(raw),
                    "action_switches": int(switched.sum()),
                    "forecast_changes": int(changed.sum()),
                    "improvements": int((d < -1e-12).sum()),
                    "regressions": int((d > 1e-12).sum()),
                    "ties": int((d.abs() <= 1e-12).sum()),
                    "net_delta_per_switch": d.mean(),
                    "net_delta_per_changed_forecast": differences[changed].mean(),
                    "mean_gain_per_improved_bank": -d[d < -1e-12].mean(),
                    "mean_cost_per_regressed_bank": d[d > 1e-12].mean(),
                    "native_brier": raw.native.mean(),
                    "oracle_brier": oracle.mean(),
                    "headroom": (raw.native - oracle).mean(),
                    "gain_over_native": (raw.native - raw[rule]).mean(),
                }
            )
    return pd.DataFrame(records)


def scenario_headroom(primary):
    """Locate each standard family's contribution to hindsight gains."""
    standard = primary[primary.cohort == "standard"]
    by_bank = standard.pivot(index=KEYS, columns="method", values="brier")
    oracle = by_bank[["native", "mean", "top_n"]].min(axis=1)
    gain = by_bank.native - oracle
    by_family = pd.DataFrame(
        {
            "native_brier": by_bank.native,
            "oracle_brier": oracle,
            "headroom": gain,
            "draws_with_strict_gain": gain > 1e-12,
        }
    )
    out = by_family.groupby(level="family").agg(
        {
            "native_brier": "mean",
            "oracle_brier": "mean",
            "headroom": "mean",
            "draws_with_strict_gain": "sum",
        }
    )
    total = out.headroom.sum()
    out["fraction_of_total_headroom"] = out.headroom / total if total > 0 else np.nan
    return out.reset_index()


def shortcut_intervention(primary, r):
    """Each fitted forecast faces three target laws on the identical feature support."""
    sample = primary[primary.family.isin(SHORTCUTS)]
    grouped = sample.groupby(["seed", "method"])
    if (
        not grouped.family.nunique().eq(3).all()
        or not grouped[
            [
                "action",
                "fit_hash",
                "test_hash",
                "source_x_hash",
                "source_y_hash",
                "test_x_hash",
            ]
        ]
        .nunique()
        .eq(1)
        .all()
        .all()
    ):
        raise ValueError("Forecast or deployed action changed with unseen target law")
    table = sample.pivot(
        index="seed", columns=["family", "method"], values="brier"
    ).sort_index()
    base = table[SHORTCUTS[0]]
    index = pd.MultiIndex.from_arrays(
        [np.repeat("shortcut", len(table)), table.index], names=KEYS
    )
    w = weights(index, r)
    rows = []
    for law in SHORTCUTS[1:]:
        compared = table[law]
        delta = compared.sub(compared.native, axis=0) - base.sub(base.native, axis=0)
        rows.append(
            interval(w, delta)
            .assign(target_law=law, reference_law=SHORTCUTS[0], pairs=len(delta))
            .reset_index()
        )
    return pd.concat(rows)


def matched_variant(raw, r):
    label = "large" if r == 7 else "seed0"
    standard = raw[raw.cohort == "standard"]
    if (raw.loc[raw.variant == label, "cohort"] != "standard").any():
        raise ValueError("Intervention must have only matched standard families")
    key = [*KEYS, "method"]
    left = standard[standard.variant == "primary"].set_index(key).sort_index()
    right = standard[standard.variant == label].set_index(key).sort_index()
    if not left.index.equals(right.index) or not left.n_test.eq(right.n_test).all():
        raise ValueError("Unpaired intervention banks")
    for field in ("test_x_hash", "target_hash"):
        if not left[field].equals(right[field]):
            raise ValueError(f"The intervention changed the test bank: {field}")
    if r == 7 and not (right.n_fit.eq(512) & left.n_fit.eq(128)).all():
        raise ValueError("The large source intervention must increase the fit size")
    if r == 8:
        if not left[["n_fit", "source_x_hash", "source_y_hash"]].equals(
            right[["n_fit", "source_x_hash", "source_y_hash"]]
        ):
            raise ValueError("Optimizer intervention changed the fit size")
        for method in CHEAP:
            for digest in ("fit_hash", "test_hash"):
                if not left.xs(method, level="method")[digest].equals(
                    right.xs(method, level="method")[digest]
                ):
                    raise ValueError(
                        f"Optimizer intervention changed {method} predictions"
                    )

    def delta(frame):
        wide = frame.brier.unstack("method")
        return wide.sub(wide.native, axis=0)

    a, b = delta(left), delta(right)
    w = weights(a.index, r)
    data = interval(w, a - b)
    primary = left.brier.unstack("method")
    other = right.brier.unstack("method")
    direct = interval(w, primary - other)
    data["primary_brier"], data["other_brier"] = primary.mean(), other.mean()
    data["direct_primary_minus_other"] = direct.delta
    data["direct_lo"], data["direct_hi"] = direct.lo, direct.hi
    data["primary_delta_native"] = a.mean()
    data["other_delta_native"] = b.mean()
    return data.assign(other_variant=label, units=len(a)).reset_index()


def selection(raw):
    part = raw[
        (raw.variant == "primary") & (raw.cohort == "standard") & ~raw.family.isin(HELD)
    ]
    risk = part.groupby(["family", "method"]).brier.mean().unstack()[list(CANDIDATES)]
    if len(risk) != 24 or risk.isna().any().any():
        raise ValueError("Missing candidate on the 24-family selection screen")
    mean = risk.mean().to_dict()
    return {
        "chosen": min(CANDIDATES, key=mean.get),
        "seen24_brier": mean,
        "r09_sha256": sha256((OUT / "r09.csv").read_bytes()).hexdigest(),
    }


def lock():
    record = selection(table(9))
    if LOCK.exists() and json.loads(LOCK.read_text()) != record:
        raise RuntimeError("Different selection already present")
    LOCK.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


def committed_selection():
    root = Path(
        subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], cwd=HERE, text=True
        ).strip()
    )
    path = LOCK.relative_to(root)
    git_bytes = subprocess.check_output(["git", "show", f"HEAD:{path}"], cwd=root)
    record = json.loads(git_bytes)
    if LOCK.read_bytes() != git_bytes or record != selection(table(9)):
        raise RuntimeError("Round-nine selection differs from the committed evidence")
    return record


def confirm(contrasts):
    record = committed_selection()
    chosen = record["chosen"]
    part = contrasts[
        (contrasts.method == chosen) & (contrasts.reference == "native")
    ].set_index("group")
    limits = {"standard": 0.0, "held6": 0.005, "weak": 0.005}
    outcomes = {}
    for group, limit in limits.items():
        delta = part.loc[group, ["delta", "lo", "hi"]].astype(float).to_dict()
        # A strict superiority endpoint, a strict held-family bound, and an inclusive weak bound.
        passed = delta["hi"] <= limit if group == "weak" else delta["hi"] < limit
        outcomes[group] = {**delta, "upper_limit": limit, "passed": bool(passed)}
    return {
        "chosen": chosen,
        "r09_sha256": record["r09_sha256"],
        "checks": outcomes,
        "all_pass": all(c["passed"] for c in outcomes.values()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("round", type=int, choices=range(1, 11))
    r = parser.parse_args().round
    raw = table(r)
    check = validate_rows(raw, r)
    primary = raw[raw.variant == "primary"]
    with threadpool_limits(limits=1):
        result = summarize(primary, r)
        if r in (5, 10):
            result["shortcut"] = shortcut_intervention(primary, r)
        if r == 10:
            result["switches"] = switch_diagnostics(primary)
            result["headroom"] = scenario_headroom(primary)
        if r in (7, 8):
            result["intervention"] = matched_variant(raw, r)
    for name, frame in result.items():
        frame.to_csv(OUT / f"r{r:02d}-{name}.csv", index=False)
    (OUT / f"r{r:02d}-verification.json").write_text(json.dumps(check, indent=2) + "\n")
    if r == 9:
        lock()
    if r == 10:
        (OUT / "confirmation.json").write_text(
            json.dumps(confirm(result["contrasts"]), indent=2, allow_nan=False) + "\n"
        )


if __name__ == "__main__":
    main()

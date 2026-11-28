"""Audit source-only provider decisions, prediction replay and comparison panels."""

import argparse
import gzip
import json
import subprocess
from hashlib import sha256
from pathlib import Path
from runpy import run_path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from fusionbench import generate_dataset
from fusionbench.methods.averaging import SimpleAverage
from fusionbench.methods.native_pattern import RegularizedNativePatternOKAPI
from fusionbench.scenarios.source_environments import source_environment_bank
from fusionbench.transfer import load_preset

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
OLD = run_path(str(HERE.parent / "full-data-five" / "verify_full.py"))
KEY = ["cohort", "family", "seed"]
WORLD = {
    "s0": (0.5, 0),
    "s50": (0.5, 0.5),
    "s100": (0.5, 1),
    "specialist80": (0.8, 0.5),
}
METHODS = ("native", "mean", "top_n", "lr", "hard", "half", "protect", "adaptive")


def digest(arr):
    return sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def eq(a, b):
    np.testing.assert_allclose(
        np.asarray(a, dtype=float), np.asarray(b, dtype=float), atol=5e-12, rtol=0
    )


def table(name):
    return pd.read_csv(OUT / name, float_precision="round_trip")


def independent_joint(x, order, prevalence, activity):
    """Integrate class and region first; only model votes are observed by the Bayes rule."""
    signs = x[:, np.argsort(order), 1] > 0.5
    joint = np.zeros((len(x), 2))
    for y in (0, 1):
        for region in (0, 1):
            rows = signs[:, 2] == region
            successes = signs[rows][:, [0, 1, 3]] == y
            accuracies = [0.75, 0.95 if region else 0.5, 0.5 + 0.4 * activity]
            prob = np.prod(
                np.where(successes, accuracies, 1 - np.array(accuracies)), axis=1
            )
            joint[rows, y] = 0.5 * (prevalence if region else 1 - prevalence) * prob
    eq(joint.sum(), 1)
    return joint


def make_examples(r):
    base = 940000 + 1000 * r
    seed = base + 400
    dataset = generate_dataset("gated_experts", seed=seed, n_per_split=512)
    x = dataset.fit_preds
    mean = x.mean(1)
    disagreement = ((x - mean[:, None]) ** 2).sum(2).mean(1)
    confidence = mean.max(1)
    yield SimpleNamespace(
        key=("standard", "gated_experts", seed),
        x=x,
        y=dataset.fit_y,
        idx=np.arange(len(x)),
        env=(disagreement > np.median(disagreement)).astype(int),
        ctx=(confidence > np.median(confidence)).astype(int),
        shortcut=None,
        specialist=None,
        worlds=[("test", dataset.test_preds, dataset.test_y, None)],
    )
    seed = base + 100
    bank = source_environment_bank(
        region=(0.15, 0.85), shortcut=(0, 1), n_rows=512, seed=seed
    )
    worlds = []
    for name, (p, a) in WORLD.items():
        target = bank.target(p, a)
        joint = independent_joint(target.x, bank.provider_order, p, a)
        eq(joint.sum(1), target.mass)
        eq(joint[:, 1] / joint.sum(1), target.q)
        worlds.append((name, target.x, None, joint))
    yield SimpleNamespace(
        key=("focus", "mixed", seed),
        x=bank.x,
        y=bank.y,
        idx=np.arange(len(bank.y)),
        env=bank.env,
        ctx=bank.context,
        shortcut=bank.shortcut_index,
        specialist=bank.specialist_index,
        worlds=worlds,
    )


def gains(x, y, fold, candidate):
    """Out-of-fold loss differences, refitting each median and choice inside the fold."""
    target = np.eye(x.shape[-1])[y]
    losses = np.square(x - target[:, None, :]).sum(2)
    others = [i for i in range(x.shape[1]) if i != candidate]
    direct, routing = np.zeros(len(y)), np.zeros(len(y))
    for k in (0, 1):
        train, held = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        prevalence = target[train].mean(0)
        direct[held] = (
            np.square(target[held] - prevalence).sum(1) - losses[held, candidate]
        )
        fixed = min(others, key=lambda i: losses[train, i].sum())
        best = None
        for c in [1] if x.shape[2] == 2 else range(x.shape[2]):
            threshold = np.median(x[train, candidate, c])
            low = x[train, candidate, c] <= threshold
            if min(low.sum(), (~low).sum()) < 8:
                continue
            left, right = train[low], train[~low]
            low_id = min(others, key=lambda i: losses[left, i].sum())
            high_id = min(others, key=lambda i: losses[right, i].sum())
            cost = losses[left, low_id].sum() + losses[right, high_id].sum()
            if best is None or cost < best[0]:
                best = cost, c, threshold, low_id, high_id
        if best is not None:
            _, c, t, low_id, high_id = best
            routed = np.where(x[held, candidate, c] <= t, low_id, high_id)
            routing[held] = losses[held, fixed] - losses[held, routed]
    return direct.mean(), routing.mean()


def source_screen(b):
    x, y = b.x[b.idx], b.y[b.idx]
    loss = np.square(x.astype(float) - np.eye(x.shape[-1])[y, None, :]).sum(2)
    levels, strata = np.unique(b.env), np.unique(b.ctx)
    if len(levels) < 2:
        return None
    props = np.array([np.mean(b.ctx == c) for c in strata])
    means, variances = [], []
    for e in levels:
        cells = [loss[(b.env == e) & (b.ctx == c)] for c in strata]
        if min(map(len, cells)) < 8:
            return None
        means.append(sum(v.mean(0) * w for v, w in zip(cells, props, strict=True)))
        variances.append(
            sum(
                w * w * v.var(0, ddof=1) / len(v)
                for v, w in zip(cells, props, strict=True)
            )
        )
    score = np.abs(means[0] - means[1]) - 1.96 * np.sqrt(variances[0] + variances[1])
    highest = np.flatnonzero(score == score.max())
    return int(highest[0]) if score.max() > 0.1 and len(highest) == 1 else None


def native(x, y):
    model = (
        RegularizedNativePatternOKAPI(seed=0, split_seed=0, gp_validation_top_k=8)
        if x.shape[1] > 1
        else SimpleAverage()
    )
    return model.fit(
        x, y, n_classes=x.shape[2], task="binary" if x.shape[2] == 2 else "multiclass"
    )


def marginal(model, x, d, src):
    """Reconstruct donor selection by sorting Python tuples, preserving entire reports."""
    votes = sorted(map(tuple, src[:, d].astype(float)))
    donors = [votes[(2 * j + 1) * len(votes) // 16] for j in range(8)]
    out = np.zeros((len(x), x.shape[2]))
    for donor in donors:
        replaced = np.asarray(x, dtype=float).copy()
        replaced[:, d] = donor
        out += model.predict(replaced)
    return out / len(donors)


def brier(pred, labels, joint):
    if labels is not None:
        return float(np.square(np.eye(pred.shape[1])[labels] - pred).sum(1).mean())
    return float(
        sum(
            joint[:, y] @ np.square(pred - np.eye(pred.shape[1])[y]).sum(1)
            for y in range(pred.shape[1])
        )
    )


def native_bank_predictions(b, drop, direct, routing, raw_methods):
    x, y = b.x, b.y
    full = native(x, y)
    hard = None
    if drop is not None:
        keep = np.delete(np.arange(x.shape[1]), drop)
        hard = native(np.ascontiguousarray(x[:, keep]), y)
    strengths = {
        "half": 0.5,
        "protect": 0 if direct > 0.02 or routing > 0.02 else 0.5,
        "adaptive": 0 if routing > 0.02 else 0.25 if direct > 0.02 else 0.75,
    }
    queries = [x, *(w[1] for w in b.worlds)]
    predicted = []
    cheap = OLD["controls"](x, y, queries)
    for q, cheap_pred in zip(queries, cheap, strict=True):
        ordinary = np.asarray(full.predict(q), dtype=float)
        h = (
            ordinary
            if hard is None
            else np.asarray(hard.predict(np.ascontiguousarray(q[:, keep])), dtype=float)
        )
        m = ordinary if drop is None else marginal(full, q, drop, x)
        pred = {"native": ordinary, "hard": h, **cheap_pred}
        for name, strength in strengths.items():
            s = strength if drop is not None else 0.0
            pred[name] = (1 - s) * ordinary + s * m
        predicted.append({k: v for k, v in pred.items() if k in raw_methods})
    return predicted


def layout(r, raw):
    assert (
        raw["round"].eq(r).all() and not raw.duplicated([*KEY, "world", "method"]).any()
    )
    base = 940000 + 1000 * r
    n_std = 8 if r == 10 else 4 if r == 9 else 2
    n_focal = 24 if r == 10 else 12 if r == 9 else 6
    std = set(load_preset("standard-v6")["scenarios"])
    assert len(std) == 30
    focal = {"mixed", "case_mix"}
    extra = {
        2: {"varied"},
        6: {"narrow"},
        7: {"small128"},
        8: {"homogeneous"},
        9: {"reversed"},
        10: {"varied", "narrow", "small128", "reversed", "homogeneous"},
    }
    cohorts = [
        ("standard", std, 400, n_std, {"test"}),
        ("focus", focal | extra.get(r, set()), 100, n_focal, set(WORLD)),
    ]
    if r in (3, 10):
        cohorts.append(
            (
                "gate_control",
                {"gated_original", "gated_shuffled"},
                700,
                n_focal,
                {"test"},
            )
        )
    expected = {
        (c, f, base + shift + i, w)
        for c, fs, shift, n, ws in cohorts
        for f in fs
        for i in range(n)
        for w in ws
    }
    universe = raw.groupby([*KEY, "world"]).method
    assert (
        set(universe.groups) == expected
        and universe.nunique().eq(raw.method.nunique()).all()
    )
    assert set(METHODS) <= set(raw.method)
    focus = raw[raw.cohort == "focus"].groupby(["family", "seed", "method"])
    for feature in ("fit_hash", "test_hash"):
        assert focus[feature].nunique(dropna=False).eq(1).all(), feature
    return len(expected), len(raw.groupby(KEY))


def paired(frame, r, method, reference):
    frame = frame.sort_index()
    d = (frame[method] - frame[reference]).to_numpy()
    f = len(frame.index.unique("family"))
    n = len(d) // f
    i = np.random.default_rng(93000 + r).integers(n, size=(10_000, f, n))
    i += (np.arange(f) * n)[None, :, None]
    return [d.mean(), *np.quantile(d[i].mean(axis=(1, 2)), [0.025, 0.975])]


def endpoint_tables(raw, bayes):
    std = raw[raw.cohort == "standard"].pivot(
        index=["family", "seed"], columns="method", values="brier"
    )
    result = {"standard": std, "standard/gated_experts": std.loc[["gated_experts"]]}
    for f, sub in raw[raw.cohort == "focus"].groupby("family"):
        models = {}
        for w, block in sub.groupby("world"):
            models[w] = block.pivot(
                index=["family", "seed"], columns="method", values="brier"
            )
            result[f"{f}/{w}"] = models[w]
        pair = [models[w] for w in ("s0", "s100")]
        result[f"{f}/worst_brier"] = pair[0].combine(pair[1], np.maximum)
        result[f"{f}/max_regret"] = (pair[0] - bayes["s0"]).combine(
            pair[1] - bayes["s100"], np.maximum
        )
    return result


def audit_all_decisions(raw, r):
    with gzip.open(OUT / f"r{r:02}-source.jsonl.gz", "rt") as stream:
        records = [json.loads(line) for line in stream]
    traces = {tuple(rec[k] for k in KEY): rec for rec in records}
    all_banks = raw.groupby(KEY, sort=False)
    assert len(traces) == len(records) and set(traces) == set(all_banks.groups)
    checked = 0
    for key, banks in all_banks:
        t = traces[key]
        sample = banks.drop_duplicates("method").set_index("method")
        ordinary = sample.loc["native"]
        for method in sample.index.difference(["native", "mean", "top_n", "lr"]):
            row = sample.loc[method]
            flagged = t["flags"]["permuted" if method == "permuted_env" else "original"]
            if flagged is None:
                assert pd.isna(row["drop"]) and row.strength == 0
                assert pd.isna(row.direct_gain) and pd.isna(row.routing_gain)
            else:
                assert row["drop"] == flagged
                used = "shuffled" if method == "shuffled_evidence" else "real"
                e = t["evidence"][f"{flagged}/{used}"]
                eq(
                    [row.direct_gain, row.routing_gain],
                    [e["direct_gain"], e["routing_gain"]],
                )
                d, g = row.direct_gain > 0.02, row.routing_gain > 0.02
                strength = {
                    "hard": 1.0,
                    "quarter": 0.25,
                    "half": 0.5,
                    "three_quarter": 0.75,
                    "full": 1.0,
                    "half_d2": 0.5,
                    "half_d32": 0.5,
                    "direct_only": 0 if d else 0.5,
                    "routing_only": 0 if g else 0.5,
                    "protect": 0 if (g or d) else 0.5,
                }
                expect = strength.get(method, 0 if g else 0.25 if d else 0.75)
                eq(row.strength, expect)
            if row.strength == 0:
                assert (row.fit_hash, row.test_hash) == (
                    ordinary.fit_hash,
                    ordinary.test_hash,
                )
            checked += 1
    return checked


def audit_minimax():
    from scipy.optimize import minimize_scalar

    rec = json.loads((OUT / "endpoint-regret-oracle.json").read_text())
    votes = np.array([row["votes"] for row in rec["report_patterns"]])
    x = np.stack([1 - votes, votes], axis=2)
    endpoints = [independent_joint(x, np.arange(4), 0.5, a) for a in (0, 1)]
    masses = np.array([j.sum(axis=1) for j in endpoints])
    q0, q1 = [j[:, 1] / j.sum(axis=1) for j in endpoints]
    # Dual of the maximum endpoint regret: integrate out the shared prediction for
    # each pattern, then maximize the resulting one-dimensional concave function.
    m0, m1 = masses

    def lower_bound(a):
        mix = (1 - a) * m0 + a * m1
        return 2 * a * (1 - a) * np.sum(m0 * m1 / mix * (q1 - q0) ** 2)

    opt = minimize_scalar(
        lambda a: -lower_bound(a),
        bounds=(0, 1),
        method="bounded",
        options={"xatol": 1e-12},
    )
    assert opt.success
    eq(-opt.fun, rec["minimax_regret"]["max_excess"])
    assert abs(opt.x - rec["minimax_regret"]["active_prior"]) < 1e-7
    fixed = np.array([row["minimax_regret_forecast"] for row in rec["report_patterns"]])
    eq(
        [2 * np.sum(m0 * (fixed - q0) ** 2), 2 * np.sum(m1 * (fixed - q1) ** 2)],
        [-opt.fun, -opt.fun],
    )
    return float(-opt.fun)


def check_examples(r, raw):
    counts = {
        "independent_native_and_cheap_bank_replays": 0,
        "primary_policy_hash_pairs": 0,
        "cheap_full_brier_replays": 0,
        "independent_focal_joint_rows": 0,
    }
    bayes = {}
    for b in make_examples(r):
        rows = raw.set_index(KEY).sort_index().loc[b.key]
        idx = b.idx
        drop = source_screen(b)
        data = rows.drop_duplicates("method").set_index("method")
        assert rows.source_x_hash.eq(digest(b.x)).all()
        assert rows.source_y_hash.eq(digest(b.y)).all()
        order = np.random.default_rng([b.key[-1], 817]).permutation(np.unique(idx))
        fold = np.isin(idx, order[::2]).astype(int)
        direct, routing = (
            (0, 0) if drop is None else gains(b.x[idx], b.y[idx], fold, drop)
        )
        for name in ("hard", "half", "protect", "adaptive"):
            if drop is None:
                assert pd.isna(data.at[name, "drop"]) and data.at[name, "strength"] == 0
            else:
                eq(
                    data.loc[name, ["drop", "direct_gain", "routing_gain"]],
                    [drop, direct, routing],
                )
        forecasts = native_bank_predictions(
            b, drop, direct, routing, rows.method.unique()
        )
        fit = forecasts[0]
        for (world, query, labels, joint), result in zip(
            b.worlds, forecasts[1:], strict=True
        ):
            saved = rows[rows.world == world].set_index("method")
            assert saved.test_x_hash.eq(digest(query)).all()
            if joint is not None:
                bayes[world] = np.sum(2 * joint[:, 0] * joint[:, 1] / joint.sum(1))
                eq(saved.oracle_brier, np.repeat(bayes[world], len(saved)))
                counts["independent_focal_joint_rows"] += len(joint)
            for m, p in result.items():
                eq(saved.at[m, "brier"], brier(p, labels, joint))
                if m in ("lr", "top_n", "mean"):
                    counts["cheap_full_brier_replays"] += 1
                else:
                    assert digest(fit[m]) == saved.at[m, "fit_hash"], (
                        r,
                        b.key,
                        m,
                        "fit",
                    )
                    assert (
                        digest(np.asarray(p, dtype=float)) == saved.at[m, "test_hash"]
                    ), (r, b.key, m, "test")
                    counts["primary_policy_hash_pairs"] += 1
        counts["independent_native_and_cheap_bank_replays"] += 1
        if b.key[0] == "focus":
            # The deployment routine sees a source-only facade and unlabelled query
            # reports. Forging the audit selector tests that role-only diagnostic reads
            # cannot change a choice or prediction. No target field is present here.
            from run import forecast_bank

            source = SimpleNamespace(
                x=b.x,
                y=b.y,
                idx=idx,
                groups=idx,
                env=b.env,
                ctx=b.ctx,
                seed=b.key[-1],
                selector=0,
            )
            query = b.worlds[0][1]
            counterfactual, choices, _ = forecast_bank(
                r, source, np.concatenate([b.x, query], axis=0)
            )
            saved = rows[rows.world == "s0"].set_index("method")
            for m, forecast in counterfactual.items():
                assert digest(forecast[: len(b.x)]) == saved.at[m, "fit_hash"]
                assert digest(forecast[len(b.x) :]) == saved.at[m, "test_hash"]
                eq(choices[m]["strength"], saved.at[m, "strength"])
            counts["guarded_source_only_hash_pairs"] = len(counterfactual)
    return counts, bayes


def check_summary(raw, r, bayes):
    prefix = f"r{r:02d}"
    family = (
        table(prefix + "-families.csv").set_index(["family", "method"]).sort_index()
    )
    std = raw[raw.cohort == "standard"].groupby(["family", "method"])
    assert (
        set(std.groups) == set(family.index)
        and len(family.index.unique("family")) == 30
    )
    cols = ["brier", "macro_f1", "providers"]
    eq(std[cols].mean().sort_index(), family[cols])
    focus = raw[raw.cohort == "focus"]
    for w, v in bayes.items():
        rows = focus[focus.world == w]
        eq(rows.oracle_brier, np.full(len(rows), v))
        assert (rows.brier >= v - 1e-12).all()
    panels = endpoint_tables(raw, bayes)
    levels = table(prefix + "-levels.csv").set_index(["group", "method"])
    contrasts = table(prefix + "-contrasts.csv").set_index(
        ["group", "method", "reference"]
    )
    checked = 0
    for g in (
        "standard",
        "standard/gated_experts",
        "mixed/s0",
        "mixed/s50",
        "mixed/s100",
        "mixed/specialist80",
        "mixed/worst_brier",
        "mixed/max_regret",
    ):
        mat = panels[g]
        for m in mat:
            eq(levels.at[(g, m), "risk"], mat[m].mean())
            for ref in ("native", "hard"):
                if m == ref:
                    continue
                eq(
                    contrasts.loc[(g, m, ref), ["delta", "lo", "hi"]],
                    paired(mat, r, m, ref),
                )
                checked += 1
    return panels, checked


def lock_and_confirmation(r, panels, bayes):
    lock_path = HERE / "selection.json"
    lock = json.loads(lock_path.read_text())
    rr9 = panels if r == 9 else endpoint_tables(table("r09.csv"), bayes)
    candidates = ["protect", "adaptive", "half"]
    diffs = rr9["standard"].subtract(rr9["standard"]["native"], axis=0).mean()
    regret = rr9["mixed/max_regret"].mean()
    eligible = [m for m in candidates if diffs[m] <= 0.005]
    assert eligible == lock["eligible"]
    chosen = min(eligible, key=regret.get) if eligible else "native"
    assert chosen == lock["chosen"]
    for m in candidates:
        eq(
            [lock["standard_delta_native"][m], lock["mixed_max_regret"][m]],
            [diffs[m], regret[m]],
        )
    assert sha256((OUT / "r09.csv").read_bytes()).hexdigest() == lock["r09_sha256"]
    result = {"chosen_based_on_round9": chosen}
    if r == 9:
        return result
    root = Path(
        subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], cwd=HERE, text=True
        ).strip()
    )
    name = str(lock_path.relative_to(root))
    commit = subprocess.check_output(
        ["git", "log", "-1", "--format=%H", "--diff-filter=A", "--", name],
        cwd=root,
        text=True,
    ).strip()
    assert (
        commit
        and subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=root)
        == lock_path.read_bytes()
    )
    assert (
        subprocess.run(
            [
                "git",
                "cat-file",
                "-e",
                f"{commit}:{(OUT / 'r10.csv').relative_to(root)}",
            ],
            cwd=root,
            check=False,
            capture_output=True,
        ).returncode
        != 0
    )
    confirmation = json.loads((OUT / "confirmation.json").read_text())
    passed = []
    limits = {
        "mixed/max_regret": 0,
        "standard": 0.005,
        "mixed/s100": 0.005,
        "standard/gated_experts": 0.010,
    }
    for g, t in limits.items():
        v = paired(panels[g], 10, chosen, "native")
        record = confirmation["checks"][g]
        eq([record[k] for k in ("delta", "lo", "hi", "upper_limit")], [*v, t])
        p = v[2] < 0 if g == "mixed/max_regret" else v[2] <= t
        assert p == record["passed"]
        passed.append(p)
    assert confirmation["primary_pass"] == passed[0] and confirmation[
        "all_guardrails_pass"
    ] == all(passed[1:])
    assert confirmation["all_pass"] == all(passed) and confirmation["chosen"] == chosen
    result.update(selection_commit=commit, all_pass=bool(all(passed)))
    return result


def main(r):
    prefix = f"r{r:02d}"
    raw = table(prefix + ".csv")
    conditions, sources = layout(r, raw)
    counts, bayes = check_examples(r, raw)
    counts["source_trace_decisions"] = audit_all_decisions(raw, r)
    counts["exact_minimax_regret"] = audit_minimax()
    panels, paired_n = check_summary(raw, r, bayes)
    counts.update(
        round=r,
        score_rows=len(raw),
        audited_conditions=conditions,
        audited_sources=sources,
        standard_families=30,
        independent_paired_intervals=paired_n,
        exact_oracle_brier=bayes,
    )
    if r >= 9:
        counts.update(lock_and_confirmation(r, panels, bayes))
    path = OUT / f"{prefix}-verification.json"
    path.write_text(json.dumps(counts, indent=2, allow_nan=False) + "\n")
    print(path.name, counts, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("round", type=int, choices=range(1, 11))
    with threadpool_limits(limits=1):
        main(parser.parse_args().round)

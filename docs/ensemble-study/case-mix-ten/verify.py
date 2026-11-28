"""Independent source-weight, native stage, focal law and paired outcome audit."""

import argparse
import gzip
import json
import subprocess
from collections import Counter
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from runpy import run_path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from fusionbench import generate_dataset
from fusionbench.eval.suite_point_gp import predict_tree
from fusionbench.methods.native_pattern import RegularizedNativePatternOKAPI
from fusionbench.scenarios.source_environments import source_environment_bank
from fusionbench.transfer import load_preset

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
PRIOR = run_path(str(HERE.parent / "provider-use-ten" / "verify.py"))
BASE = PRIOR["OLD"]
KEY = ["cohort", "family", "seed"]
WORLDS = PRIOR["WORLD"]
PROFILES = {
    "mixed": ((0.15, 0.85), (0, 1)),
    "case_mix": ((0.15, 0.85), (0.5, 0.5)),
    "varied": ((0.5, 0.5), (0, 1)),
    "reversed": ((0.85, 0.15), (0, 1)),
    "extreme": ((0.035, 0.965), (0, 1)),
    "homogeneous": ((0.5, 0.5), (0.5, 0.5)),
    "small128": ((0.15, 0.85), (0, 1)),
    "noisy_context": ((0.15, 0.85), (0, 1)),
}
EXTRAS = {
    2: ("varied",),
    3: ("reversed",),
    5: ("extreme",),
    6: ("small128",),
    7: ("noisy_context",),
    8: ("homogeneous",),
    9: ("reversed",),
    10: ("varied", "reversed", "extreme", "small128", "noisy_context", "homogeneous"),
}


def close(a, b, tol=5e-12):
    np.testing.assert_allclose(
        np.asarray(a, dtype=float), np.asarray(b, dtype=float), atol=tol, rtol=0
    )


def digest(x):
    return sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def read(name):
    return pd.read_csv(OUT / name, float_precision="round_trip")


def source_weights(env, ctx, cap=None):
    """Empirical r(e,c)=p(e)p(c)/p(e,c) from categorical counts in source row order."""
    n = len(env)
    by_e, by_c, fields = Counter(env), Counter(ctx), list(zip(env, ctx, strict=True))
    ncell = Counter(fields)
    missing = len(by_e) * len(by_c) - len(ncell)
    minimum = min(ncell.values())
    raw = np.array([by_e[e] * by_c[c] / (n * ncell[e, c]) for e, c in fields])
    reason = (
        "single_environment"
        if len(by_e) == 1
        else "single_context"
        if len(by_c) == 1
        else "missing_overlap"
        if missing
        else "insufficient_cell_count"
        if minimum < 4
        else "weighted"
    )
    if reason != "weighted":
        w = np.ones(n)
    elif cap is None:
        w = raw.copy()
    else:
        w = np.minimum(raw, cap)
        w /= w.mean()
    if reason == "weighted" and cap is None:
        # Direct check of both marginal frequencies and the requested independence.
        env, ctx = np.asarray(env), np.asarray(ctx)
        for e, ne in by_e.items():
            for c, nc in by_c.items():
                close(w[(env == e) & (ctx == c)].sum() / n, (ne / n) * (nc / n))
    close(w.mean(), 1)
    return w, {
        "reason": reason,
        "raw_max": raw.max(),
        "final_max": w.max(),
        "effective_n": w.sum() ** 2 / (w @ w),
        "overlap": not missing,
        "missing_cells": missing,
        "min_observed_cell": minimum,
        "clipped_rows": 0
        if cap is None or reason != "weighted"
        else int((raw > cap).sum()),
    }


def group_banks(r):
    base = 1040000 + 1000 * r
    for scenario in ("gated_experts", "grouped_rare_event"):
        d = generate_dataset(scenario, seed=base + 400, n_per_split=512)
        x = d.fit_preds
        mean = x.mean(1)
        var = np.square(x - mean[:, None, :]).sum(axis=2).mean(1)
        confidence = mean.max(1)
        yield SimpleNamespace(
            key=("standard", scenario, base + 400),
            x=x,
            y=d.fit_y,
            env=(var > np.median(var)).astype(int),
            ctx=(confidence > np.median(confidence)).astype(int),
            worlds=[("test", d.test_preds, d.test_y, None)],
        )
    for f in ("mixed", "case_mix", *EXTRAS.get(r, ())):
        region, shortcut = PROFILES[f]
        seed = base + 100
        b = source_environment_bank(
            region=region,
            shortcut=shortcut,
            n_rows=128 if f == "small128" else 512,
            seed=seed,
            context_noise=0.25 if f == "noisy_context" else 0,
        )
        worlds = []
        for name, (p, a) in WORLDS.items():
            target = b.target(p, a)
            joint = PRIOR["independent_joint"](target.x, b.provider_order, p, a)
            close(joint.sum(1), target.mass)
            close(joint[:, 1] / joint.sum(1), target.q)
            worlds.append((name, target.x, None, joint))
        yield SimpleNamespace(
            key=("focus", f, seed),
            x=b.x,
            y=b.y,
            env=b.env,
            ctx=b.context,
            worlds=worlds,
        )


def native_fit(b, w, stages):
    kw = {
        "task": "binary" if b.x.shape[2] == 2 else "multiclass",
        "n_classes": b.x.shape[2],
    }
    model = RegularizedNativePatternOKAPI(seed=0, split_seed=0, gp_validation_top_k=8)
    if w is not None:
        kw.update(sample_weight=w, weight_stages=stages)
    model.fit(b.x, b.y, **kw)
    return model


def audit_stages(b, m, w, stages):
    order = np.random.default_rng(np.random.SeedSequence([0, 51])).permutation(len(b.y))
    d, v = np.split(order, [len(order) * 7 // 10])
    np.testing.assert_array_equal(d, m.discovery_indices_)
    np.testing.assert_array_equal(v, m.validation_indices_)
    assert set(d).isdisjoint(v) and len(d) + len(v) == len(b.y)
    assert m.counts_["gp_selection_scale"] == "unit_brier_3_decimals"

    def skill(tree, rows, stage):
        p = np.asarray(predict_tree(tree, b.x[rows]))
        loss = np.square(p - np.eye(b.x.shape[2])[b.y[rows]]).sum(1)
        return 1 - 0.5 * np.average(
            loss, weights=w[rows] if w is not None and stage in stages else None
        )

    for name, tree in m.discovery_trees_.items():
        close(skill(tree, v, "validation"), m.counts_["validation_scores"][name])
    assert (
        max(m.counts_["validation_scores"], key=m.counts_["validation_scores"].get)
        == m.chosen_
    )
    for stage, rows, family in (
        ("discovery", d, m.discovery_trees_),
        ("refit", np.arange(len(b.y)), m.trees_),
    ):
        for name in ("flat", "augmented"):
            rec = m.counts_["head_fit_scores"][name + "_" + stage]
            close(skill(family[name], rows, stage), rec["raw_score"], tol=4e-7)
            close(rec["raw_score"] - rec["penalty"], rec["penalized_score"])
    return m.counts_["gp_path_digest"]


def coverage(r, raw):
    ns = 6 if r == 10 else 3 if r == 9 else 2
    nf = 20 if r == 10 else 10 if r == 9 else 6
    std = set(load_preset("standard-v6")["scenarios"])
    assert len(std) == 30
    families = {
        "standard": (std, ns, 400, ("test",)),
        "focus": (("mixed", "case_mix", *EXTRAS.get(r, ())), nf, 100, WORLDS),
    }
    base = 1040000 + 1000 * r
    expect = {
        (cohort, f, base + shift + i, world)
        for cohort, (fs, n, shift, worlds) in families.items()
        for f in fs
        for i in range(n)
        for world in worlds
    }
    groups = raw.groupby([*KEY, "world"]).method
    assert (
        set(groups.groups) == expect and groups.nunique().eq(raw.method.nunique()).all()
    )
    assert (
        raw["round"].eq(r).all() and not raw.duplicated([*KEY, "world", "method"]).any()
    )
    assert {"native", "mean", "top_n", "lr", "full", "guarded", "cap2"} <= set(
        raw.method
    )
    return len(expect)


@lru_cache(maxsize=4)
def draws(r, f, n):
    i = np.random.default_rng(103000 + r).integers(n, size=(10000, f, n))
    i += (np.arange(f) * n)[None, :, None]
    return i


def paired(df, r, m, ref):
    d = (df[m] - df[ref]).sort_index()
    f = d.index.unique("family").size
    i = draws(r, f, len(d) // f)
    return [d.mean(), *np.quantile(d.to_numpy()[i].mean((1, 2)), [0.025, 0.975])]


def all_panels(raw, bayes):
    panels = PRIOR["endpoint_tables"](raw, bayes)
    return panels


def panel_audit(r, raw, bayes):
    fam = read(f"r{r:02}-families.csv").set_index(["family", "method"]).sort_index()
    std = raw[raw.cohort == "standard"].groupby(["family", "method"])
    assert set(std.groups) == set(fam.index)
    cols = ["brier", "macro_f1", "providers"]
    close(fam[cols], std[cols].mean().sort_index())
    for w, risk in bayes.items():
        focal = raw[(raw.world == w) & (raw.cohort == "focus")]
        close(focal.oracle_brier, np.full(len(focal), risk))
        assert (focal.brier >= risk - 1e-12).all()
    panels = all_panels(raw, bayes)
    level = read(f"r{r:02}-levels.csv").set_index(["group", "method"])
    contrast = read(f"r{r:02}-contrasts.csv").set_index(
        ["group", "method", "reference"]
    )
    count = 0
    for g in (
        "standard",
        "mixed/max_regret",
        "mixed/worst_brier",
        "mixed/s0",
        "mixed/s100",
        "case_mix/specialist80",
    ):
        mat = panels[g]
        for method in mat:
            close(level.at[(g, method), "risk"], mat[method].mean())
            for ref in ("native", "full"):
                if ref == method:
                    continue
                close(
                    contrast.loc[(g, method, ref), ["delta", "lo", "hi"]],
                    paired(mat, r, method, ref),
                )
                count += 1
    return count, panels


def registered_outcome(r, panels, bayes):
    lockpath = HERE / "selection.json"
    locked = json.loads(lockpath.read_text())
    r9 = panels if r == 9 else all_panels(read("r09.csv"), bayes)
    std = r9["standard"].subtract(r9["standard"].native, axis=0).mean()
    case = (
        r9["case_mix/specialist80"]
        .subtract(r9["case_mix/specialist80"].native, axis=0)
        .mean()
    )
    worst = r9["mixed/max_regret"].mean()
    allowed = []
    for m in ("guarded", "cap2", "full"):
        close(
            [
                locked["standard_delta_native"][m],
                locked["case_mix_specialist80_delta_native"][m],
                locked["mixed_max_regret"][m],
            ],
            [std[m], case[m], worst[m]],
        )
        if std[m] <= 0.005 and case[m] <= 0.01:
            allowed.append(m)
    chosen = min(allowed, key=worst.get) if allowed else "native"
    assert locked["eligible"] == allowed and locked["chosen"] == chosen
    assert sha256((OUT / "r09.csv").read_bytes()).hexdigest() == locked["r09_sha256"]
    results = {"chosen_on_r9": chosen}
    if r == 10:
        root = Path(
            subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"], text=True, cwd=HERE
            ).strip()
        )
        path = str(lockpath.relative_to(root))
        commit = subprocess.check_output(
            ["git", "log", "-1", "--diff-filter=A", "--format=%H", "--", path],
            cwd=root,
            text=True,
        ).strip()
        assert (
            commit
            and subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=root)
            == lockpath.read_bytes()
        )
        assert subprocess.run(
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
        confirm = json.loads((OUT / "confirmation.json").read_text())
        flags = []
        for group, limit in (
            ("mixed/max_regret", 0),
            ("standard", 0.005),
            ("case_mix/specialist80", 0.01),
            ("mixed/s0", 0.01),
        ):
            actual = paired(panels[group], r, chosen, "native")
            rec = confirm["checks"][group]
            close(
                [rec[k] for k in ("delta", "lo", "hi", "upper_limit")], [*actual, limit]
            )
            ok = actual[2] < limit if limit == 0 else actual[2] <= limit
            assert rec["passed"] == ok
            flags.append(ok)
        assert (
            confirm["primary_pass"] == flags[0]
            and confirm["all_guardrails_pass"] == all(flags[1:])
            and confirm["all_pass"] == all(flags)
        )
        results.update(selection_commit=commit, all_pass=bool(all(flags)))
    return results


def check_tables(b, saved, weights):
    reports = b.x[:, :, 1] > 0.5
    totals = 0
    for name, w in weights.items():
        if name not in ("native", "full", "cap2"):
            continue
        w = np.ones(len(b.y)) if w is None else w / w.mean()
        for world, x, _, joint in b.worlds:
            posterior = []
            for pattern in x[:, :, 1] > 0.5:
                ix = (reports == pattern).all(1)
                posterior.append(
                    (0.5 + w[ix & b.y.astype(bool)].sum()) / (1 + w[ix].sum())
                )
            p = np.column_stack((1 - np.array(posterior), posterior))
            stored = saved.loc[(b.key[1], b.key[2], "table_" + name, world)]
            close(stored.brier, PRIOR["brier"](p, None, joint))
            totals += 1
    return totals


def alternatives(b, r):
    env, ctx = b.env, b.ctx
    full, report = source_weights(env, ctx)
    cap, crep = source_weights(env, ctx, 2)
    uniform = np.ones(len(env))
    whole = ("discovery", "validation", "refit")
    choices = {
        "native": (uniform, (), "native", "uniform"),
        "full": (full, whole, "full", report["reason"]),
        "cap2": (cap, whole, "cap2", crep["reason"]),
    }
    guard = (
        "full"
        if report["effective_n"] >= 0.65 * len(env)
        else "cap2"
        if crep["effective_n"] >= 0.65 * len(env)
        else "native"
    )
    choices["guarded"] = choices[guard]
    if r == 2:
        w, rep = source_weights(
            np.random.default_rng([b.key[-1], 1042]).permutation(env), ctx
        )
        choices["shuffled"] = (w, whole, "shuffled", rep["reason"])
    if r == 3:
        choices["discovery_only"] = full, ("discovery",), "full", report["reason"]
    if r == 4:
        for name, stage in (("selection_only", "validation"), ("refit_only", "refit")):
            choices[name] = full, (stage,), "full", report["reason"]
    if r == 5:
        w, rep = source_weights(env, ctx, 1.5)
        choices["cap15"] = (w, whole, "cap15", rep["reason"])
    if r == 8:
        choices["permuted_weights"] = (
            np.random.default_rng([b.key[-1], 1048]).permutation(full),
            whole,
            "permuted_weights",
            report["reason"],
        )
    return choices, report


def audit_sources(r, raw):
    with gzip.open(OUT / f"r{r:02}-source.jsonl.gz", "rt") as f:
        logs = [json.loads(line) for line in f]
    traces = {tuple(t[k] for k in KEY): t for t in logs}
    nbank = raw.groupby(KEY)
    assert set(traces) == set(nbank.groups) and len(nbank) == len(logs)
    for trace in traces.values():
        t = trace["full_weights"]
        counts = np.zeros((len(t["environment_ids"]), len(t["context_ids"])), dtype=int)
        for i, j, n in t["cell_counts"]:
            counts[i, j] = n
        ei, ci = np.indices(counts.shape)
        e = np.repeat(ei.ravel(), counts.ravel())
        c = np.repeat(ci.ravel(), counts.ravel())
        assert len(e) == trace["n_fit"]
        _, rfull = source_weights(e, c)
        _, rcap = source_weights(e, c, 2)
        assert rfull["reason"] == t["reason"]
        for col in (
            "effective_n",
            "raw_max",
            "final_max",
            "overlap",
            "missing_cells",
            "min_observed_cell",
            "clipped_rows",
        ):
            close(t[col], rfull[col])
        m = trace["methods"]
        guard = (
            "full"
            if rfull["effective_n"] >= 0.65 * len(e)
            else "cap2"
            if rcap["effective_n"] >= 0.65 * len(e)
            else "native"
        )
        assert m["guarded"]["policy"] == guard
        for col in ("fit_hash", "path_hash", "tree_hash", "weight_hash", "chosen"):
            assert m["guarded"][col] == m[guard][col]
        for name in ("native", "full", "cap2", "guarded"):
            if m[name]["stage"] == "uniform":
                assert m[name]["fit_hash"] == m["native"]["fit_hash"]
                assert m[name]["path_hash"] == m["native"]["path_hash"]
        if r == 4:
            for name in ("selection_only", "refit_only"):
                assert m[name]["path_hash"] == m["native"]["path_hash"]
        if r == 8:
            close(m["full"]["ess"], m["permuted_weights"]["ess"])
    return traces


def check_sample(r, raw, traces):
    sample = raw.set_index(KEY).sort_index()
    support = read(f"r{r:02}-support-scores.csv").set_index(
        ["family", "seed", "method", "world"]
    )
    totals = Counter()
    bayes = {}
    for b in group_banks(r):
        tr = traces[b.key]
        assert tr["env_hash"] == digest(b.env) and tr["ctx_hash"] == digest(b.ctx)
        assert tr["source_x_hash"] == digest(b.x) and tr["source_y_hash"] == digest(b.y)
        choices, _report = alternatives(b, r)
        rows = sample.loc[b.key].set_index(["world", "method"])
        st = sample.loc[b.key].drop_duplicates("method").set_index("method")
        for name, (w, stages, policy, reason) in choices.items():
            row = st.loc[name]
            assert (
                row.policy == policy
                and row.reason == reason
                and row.weight_hash == digest(w)
            )
            close(
                [row.ess, row.weight_min, row.weight_max],
                [w.sum() ** 2 / (w @ w), w.min(), w.max()],
            )
            if b.key[1] == "grouped_rare_event":
                groups = np.asarray(
                    generate_dataset(b.key[1], seed=b.key[-1], n_per_split=512).meta[
                        "evaluation_groups"
                    ]["fit"]["scene_id"]
                )
            else:
                groups = np.arange(len(w))
            _, ids = np.unique(groups, return_inverse=True)
            groupw = np.bincount(ids, weights=w)
            close(
                [row.groups, row.group_ess],
                [len(groupw), groupw.sum() ** 2 / (groupw @ groupw)],
            )
            assert row.stage == (
                "+".join(stages) if np.ptp(w) > 0 and stages else "uniform"
            )
            totals["recreated_weight_vectors"] += 1
        if b.key[0] == "focus":
            totals["jeffreys_table_risks"] += check_tables(
                b, support, {k: v[0] for k, v in choices.items()}
            )
            for world, _, _, joint in b.worlds:
                integrated = np.sum(2 * joint[:, 0] * joint[:, 1] / joint.sum(1))
                close(rows.loc[world, "oracle_brier"], integrated)
                close(integrated, bayes.setdefault(world, integrated))
                totals["conditional_joint_cells"] += len(joint)
        if b.key[1] not in ("gated_experts", "mixed"):
            continue
        cheaply = BASE["controls"](b.x, b.y, [world[1] for world in b.worlds])
        for (world, _, y, joint), preds in zip(b.worlds, cheaply, strict=True):
            for m, p in preds.items():
                close(rows.at[(world, m), "brier"], PRIOR["brier"](p, y, joint))
                totals["cheap_brier_replays"] += 1
        # Include the protected stage paths in their designated rounds, and the
        # R9-selected configuration on the final untouched seeds.
        methods = ["native", "full"]
        if r == 3:
            methods.append("discovery_only")
        if r == 4:
            methods.extend(["selection_only", "refit_only"])
        if r == 10:
            chosen = json.loads((HERE / "selection.json").read_text())["chosen"]
            if chosen not in methods:
                methods.append(chosen)
        for name in methods:
            w, stages, _, _ = choices[name]
            model = native_fit(b, w if name != "native" else None, stages)
            audit_stages(b, model, w, stages)
            assert model.chosen_ == st.at[name, "chosen"]
            assert digest(model.predict(b.x)) == st.at[name, "fit_hash"]
            for world, query, y, joint in b.worlds:
                pred = np.asarray(model.predict(query))
                assert digest(pred) == rows.at[(world, name), "test_hash"]
                close(rows.at[(world, name), "brier"], PRIOR["brier"](pred, y, joint))
                totals["native_prediction_hash_pairs"] += 1
            totals["native_source_refits"] += 1
    return dict(totals), bayes


def population_audit():
    """Independently sum the joint over environment, observed cue, report, label."""
    cells = np.indices((2, 2, 2, 2)).reshape(4, -1).T
    x = np.stack((1 - cells, cells), axis=-1)
    database = read("population.csv")
    cases = {**PROFILES, "no_shortcut": ((0.15, 0.85), (0, 0))}
    assert set(database.profile) == set(cases)
    targets = {
        k: PRIOR["independent_joint"](x, np.arange(4), *law)
        for k, law in WORLDS.items()
    }
    rows = 0
    for family, (prevalence, activity) in cases.items():
        # pi and activity index the true environment. The observed cue may flip,
        # but the four provider reports (including the clean cue) stay unchanged.
        eta = 0.25 if family == "noisy_context" else 0
        joint = np.zeros((2, 2, 16, 2))
        for e in range(2):
            clean = (
                PRIOR["independent_joint"](x, np.arange(4), prevalence[e], activity[e])
                / 2
            )
            for cue in (0, 1):
                likelihood = np.where(cells[:, 2] == cue, 1 - eta, eta)
                joint[e, cue] = clean * likelihood[:, None]
        counts = joint.sum((2, 3))
        raw = np.outer(counts.sum(1), counts.sum(0)) / counts
        weights = {
            "unweighted": np.ones((2, 2)),
            "full": raw,
            "cap2": np.minimum(raw, 2),
        }
        ess = {}
        for name, w in weights.items():
            weights[name] = w / np.sum(w * counts)
            ess[name] = 1 / np.sum(weights[name] ** 2 * counts)
        guarded = next(m for m in ("full", "cap2", "unweighted") if ess[m] >= 0.65)
        weights["guarded"] = weights[guarded]
        for m, w in weights.items():
            saved = database[
                (database.profile == family) & (database.policy == m)
            ].iloc[0]
            close(saved.ess_fraction, 1 / np.sum(w**2 * counts))
            j = (joint * w[:, :, None, None]).sum((0, 1))
            q = j / j.sum(1, keepdims=True)
            endpoint = []
            for world, law in targets.items():
                loss = PRIOR["brier"](q, None, law)
                optimal = np.sum(2 * law[:, 0] * law[:, 1] / law.sum(1))
                close(
                    [saved["risk_" + world], saved["regret_" + world]],
                    [loss, loss - optimal],
                )
                if world in ("s0", "s100"):
                    endpoint.append(loss - optimal)
            close(saved.max_regret, max(endpoint))
            rows += 1
    assert rows == len(database)
    return rows


def main(r):
    raw = read(f"r{r:02}.csv")
    targets = coverage(r, raw)
    traces = audit_sources(r, raw)
    counts, bayes = check_sample(r, raw, traces)
    nc, panels = panel_audit(r, raw, bayes)
    rec = dict(
        round=r,
        source_banks=len(traces),
        target_conditions=targets,
        score_rows=len(raw),
        standard_families=30,
        paired_intervals=nc,
        oracle_brier=bayes,
        **counts,
    )
    if r >= 9:
        rec.update(registered_outcome(r, panels, bayes))
    if r == 10:
        rec["analytic_source_policies"] = population_audit()
    (OUT / f"r{r:02}-verification.json").write_text(
        json.dumps(rec, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("round", type=int, choices=range(1, 11))
    with threadpool_limits(limits=1):
        main(cli.parse_args().round)

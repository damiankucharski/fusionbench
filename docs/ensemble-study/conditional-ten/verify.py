"""Reconstruct conditionally weighted native fits, source gates and suite estimands."""

import argparse
import gzip
import json
import subprocess
from collections import Counter
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from runpy import run_path

import numpy as np
import pandas as pd
from banks import sources
from threadpoolctl import threadpool_limits

from fusionbench import generate_dataset
from fusionbench.scenarios.source_environments import source_environment_bank
from fusionbench.transfer import load_preset

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
PAST = run_path(str(HERE.parent / "case-mix-ten" / "verify.py"))
KEY = ["cohort", "family", "seed"]
CANDIDATES = ["half", "replicated", "single"]
WORLD = PAST["WORLDS"]
close = PAST["close"]
digest = PAST["digest"]
row_weights = PAST["source_weights"]


def read(r, suffix=""):
    return pd.read_csv(OUT / f"r{r:02}{suffix}.csv", float_precision="round_trip")


def screening(loss, env, ctx, *, raw=False, half=False):
    """Estimate environment-specific mean provider losses and their sampling variances."""
    levels = list(dict.fromkeys(env.tolist()))
    contexts = np.zeros(len(ctx), dtype=int) if raw else ctx
    strata = list(dict.fromkeys(contexts.tolist()))
    minimum = 2 if raw else 4 if half else 8
    if len(levels) < 2 or loss.shape[1] < 2:
        return None, None
    pooled = np.array([(contexts == c).mean() for c in strata])
    means, variances = [], []
    for e in levels:
        cells = [loss[(env == e) & (contexts == c)] for c in strata]
        if min(map(len, cells)) < minimum:
            return None, None
        # Calculate the cells themselves first. Weights come from the rows
        # used by this screen, including when it is fit on one diagnostic half.
        ms = np.array([v.mean(0) for v in cells])
        vs = np.array(
            [((v - v.mean(0)) ** 2).sum(0) / (len(v) * (len(v) - 1)) for v in cells]
        )
        means.append(np.einsum("c,cp->p", pooled, ms))
        variances.append(np.einsum("c,cp->p", pooled**2, vs))
    z, threshold = (1.0, 0.05) if half else (1.96, 0.10)
    pairs = [(i, j) for i in range(len(levels)) for j in range(i + 1, len(levels))]
    diffs = np.array([means[j] - means[i] for i, j in pairs])
    scores = np.array(
        [
            abs(diffs[k]) - z * np.sqrt(variances[i] + variances[j])
            for k, (i, j) in enumerate(pairs)
        ]
    )
    per = scores.max(0)
    heads = np.flatnonzero(per == per.max())
    score_data = {
        "per": per,
        "diffs": diffs,
        "pairs": [(levels[i], levels[j]) for i, j in pairs],
        "scores": scores,
    }
    if len(heads) != 1 or per.max() <= threshold:
        return None, score_data
    return int(heads[0]), score_data


def halves(ids, env, ctx, seed):
    ids = np.asarray(sorted(ids))
    rng = np.random.default_rng(seed)
    out = [[], []]
    offset = 0
    for e, c in dict.fromkeys(zip(env[ids].tolist(), ctx[ids].tolist(), strict=True)):
        local = ids[(env[ids] == e) & (ctx[ids] == c)]
        perm = rng.permutation(local)
        out[offset].extend(perm[::2])
        out[1 - offset].extend(perm[1::2])
        offset = (offset + len(local)) % 2
    a, b = (np.array(sorted(v), dtype=int) for v in out)
    np.testing.assert_array_equal(np.sort(np.r_[a, b]), ids)
    assert not np.intersect1d(a, b).size and abs(len(a) - len(b)) <= 1
    return a, b


def direction(evidence, pair, provider):
    i, j = pair
    if pair in evidence["pairs"]:
        return int(np.sign(evidence["diffs"][evidence["pairs"].index(pair), provider]))
    return -int(np.sign(evidence["diffs"][evidence["pairs"].index((j, i)), provider]))


def reconstruct(fit, x, y, env, ctx, seed):
    labels = np.eye(x.shape[2])[y]
    err = np.square(x.astype(float) - labels[:, None, :]).sum(2)
    single, all_e = screening(err[fit], env[fit], ctx[fit])
    raw, _ = screening(err[fit], env[fit], ctx[fit], raw=True)
    pair = sign = None
    replicate = []
    if single is not None:
        which = int(np.argmax(all_e["scores"][:, single]))
        pair = all_e["pairs"][which]
        sign = direction(all_e, pair, single)
        for sub in halves(fit, env, ctx, seed):
            flag, h = screening(err[sub], env[sub], ctx[sub], half=True)
            replicate.append(
                flag == single and sign != 0 and direction(h, pair, single) == sign
            )
    return single, raw, single if len(replicate) == 2 and all(replicate) else None


def observed_idx(b):
    if b.cohort == "standard" and b.family in {
        "grouped_rare_event",
        "shared_error_bloc",
        "rare_scene_specialist",
    }:
        rng = np.random.default_rng([b.seed, 217])
        return np.sort(
            [rng.choice(np.flatnonzero(b.groups == g)) for g in np.unique(b.groups)]
        )
    return np.arange(len(b.y))


def independent_sample(b):
    if b.cohort == "standard":
        d = generate_dataset("gated_experts", seed=b.seed, n_per_split=512)
        x, y = d.fit_preds, d.fit_y
        avg = x.mean(1)
        var = np.square(x - avg[:, None]).sum(2).mean(1)
        conf = avg.max(1)
        env, ctx = (
            (var > np.median(var)).astype(int),
            (conf > np.median(conf)).astype(int),
        )
        worlds = [("test", d.test_preds, d.test_y, None)]
    else:
        a = source_environment_bank(
            region=(0.15, 0.85), shortcut=(0.0, 1.0), seed=b.seed, n_rows=512
        )
        x, y, env, ctx = a.x, a.y, a.env, a.context
        worlds = []
        for name, (p, s) in WORLD.items():
            target = a.target(p, s)
            joint = PAST["PRIOR"]["independent_joint"](target.x, a.provider_order, p, s)
            close(joint.sum(1), target.mass)
            close(joint[:, 1] / joint.sum(1), target.q)
            worlds.append((name, target.x, None, joint))
    for left, right in ((b.x, x), (b.y, y), (b.env, env), (b.ctx, ctx)):
        np.testing.assert_array_equal(left, right)
    return worlds


def guarded(env, ctx):
    full, fr = row_weights(env, ctx)
    cap, cr = row_weights(env, ctx, 2)
    if fr["effective_n"] >= 0.65 * len(env):
        return full, "full"
    if cr["effective_n"] >= 0.65 * len(env):
        return cap, "cap2"
    return np.ones(len(env)), "native"


def native_replay(b, rows, worlds, weights):
    models = {}
    cheap = PAST["BASE"]["controls"](b.x, b.y, [w[1] for w in worlds])
    count = 0
    for name, w in weights.items():
        code = digest(w)
        if code not in models:
            stages = ("discovery", "validation", "refit") if np.ptp(w) > 0 else ()
            model = PAST["native_fit"](b, w if stages else None, stages)
            PAST["audit_stages"](b, model, w, stages)
            models[code] = model
        model = models[code]
        source = digest(model.predict(b.x))
        for world, x, y, j in worlds:
            p = model.predict(x)
            saved = rows.loc[(world, name)]
            assert saved.fit_hash == source and saved.test_hash == digest(p)
            close(saved.brier, PAST["PRIOR"]["brier"](p, y, j))
            count += 1
    for pred, (world, _, y, j) in zip(cheap, worlds, strict=True):
        for name, p in pred.items():
            close(rows.at[(world, name), "brier"], PAST["PRIOR"]["brier"](p, y, j))
    return count, len(models), len(worlds) * 3


@lru_cache(maxsize=20)
def draw(r, f, n):
    i = np.random.default_rng(113000 + r).integers(n, size=(10000, f, n))
    i += (np.arange(f) * n)[None, :, None]
    return i


def interval(mat, r, m, ref, w):
    vals = (mat[m] - mat[ref]).sort_index()
    fs = vals.index.unique("family")
    weights = np.array([w[f] for f in fs])
    frac = weights / weights.sum()
    indices = draw(r, len(fs), len(vals) // len(fs))
    d = vals.to_numpy()
    boot = d[indices].mean(2) @ frac
    return [d.reshape(len(fs), -1).mean(1) @ frac, *np.quantile(boot, [0.025, 0.975])]


def standard_panels(raw):
    from fusionbench.taxonomy import MECHANISM_GROUPS, PURPOSE_TAGS

    std = raw[raw.cohort == "standard"].pivot(
        index=["family", "seed"], columns="method", values="brier"
    )
    names = set(load_preset("standard-v6")["scenarios"])
    members = dict(MECHANISM_GROUPS)
    count = Counter(s for vals in members.values() for s in vals)
    assert set(count) == set(std.index.unique("family")) == names and set(
        count.values()
    ) == {1}
    assert sorted(map(len, members.values())) == [3, 3, 3, 3, 4, 5, 9]
    weights = {"standard": dict.fromkeys(names, 1)}
    for g, fs in members.items():
        weights["cluster/" + g] = dict.fromkeys(fs, 1)
    for tag, fs in PURPOSE_TAGS.items():
        assert len(set(fs)) == len(fs) and set(fs) <= names
        weights["purpose/" + tag] = dict.fromkeys(fs, 1)
    applications = {
        "equal_cluster": dict.fromkeys(members, 1 / 7),
        "multi_site": {
            "transfer_shift": 0.5,
            "grouped_sampling": 0.3,
            "context_routing": 0.2,
        },
        "rule_design": {"structured_rules": 0.75, "context_routing": 0.25},
    }
    for key, shares in applications.items():
        weights[key] = {
            name: mass / len(members[g])
            for g, mass in shares.items()
            for name in members[g]
        }
        close(sum(weights[key].values()), 1)
    return {
        key: (std[std.index.get_level_values("family").isin(allocation)], allocation)
        for key, allocation in weights.items()
    }


def panel_check(r, raw, bayes):
    panels = standard_panels(raw)
    focal = PAST["all_panels"](raw, bayes)
    for name, frame in focal.items():
        if name.startswith(("mixed/", "case_mix/")):
            panels[name] = (frame, dict.fromkeys(frame.index.unique("family"), 1))
    c = read(r, "-contrasts").set_index(["group", "method", "reference"])
    levels = read(r, "-levels").set_index(["group", "method"])
    fam = read(r, "-families").set_index(["family", "method"]).sort_index()
    standard = raw[raw.cohort == "standard"].groupby(["family", "method"])
    assert set(fam.index) == set(standard.groups)
    cols = ["brier", "macro_f1", "providers"]
    close(fam[cols], standard[cols].mean().sort_index())
    n = 0
    for key, (m, w) in panels.items():
        by_family = m.groupby(level="family").mean()
        probabilities = np.array([w[f] for f in by_family.index])
        risk = np.average(by_family, axis=0, weights=probabilities)
        close(levels.loc[[(key, k) for k in m], "risk"], risk)
        # Verify all cluster and application estimates; focus the interval audit
        # on the three candidate rules and both relevant reference methods.
        for method in CANDIDATES:
            for reference in ("native", "guarded"):
                close(
                    c.loc[(key, method, reference), ["delta", "lo", "hi"]],
                    interval(m, r, method, reference, w),
                )
                n += 1
    return panels, n


def lock_check(r, panels, bayes):
    path = HERE / "selection.json"
    lock = json.loads(path.read_text())
    if r == 9:
        nine = {g: m for g, (m, _) in panels.items()}
    else:
        nine = PAST["all_panels"](read(9), bayes)
    deltas = {}
    eligible = []
    for g in ("standard", "mixed/s0", "case_mix/specialist80"):
        deltas[g] = nine[g].subtract(nine[g].native, axis=0).mean()
    worst = nine["mixed/max_regret"].mean()
    for m in CANDIDATES:
        observed = [
            deltas[g][m] for g in ("standard", "mixed/s0", "case_mix/specialist80")
        ]
        close(
            [
                lock[name][m]
                for name in (
                    "standard_delta_native",
                    "mixed_s0_delta_native",
                    "case_mix_specialist80_delta_native",
                    "mixed_max_regret",
                )
            ],
            [*observed, worst[m]],
        )
        if all(x <= t for x, t in zip(observed, [0.005, 0.010, 0.010], strict=True)):
            eligible.append(m)
    chosen = min(eligible, key=worst.get) if eligible else "native"
    assert chosen == lock["chosen"] and eligible == lock["eligible"]
    assert sha256((OUT / "r09.csv").read_bytes()).hexdigest() == lock["r09_sha256"]
    rec = {"chosen_in_r9": chosen}
    if r == 9:
        return rec
    confirm = json.loads((OUT / "confirmation.json").read_text())
    assert confirm["chosen"] == chosen
    passed = []
    for g, limit in (
        ("mixed/max_regret", 0),
        ("standard", 0.005),
        ("mixed/s0", 0.01),
        ("case_mix/specialist80", 0.01),
    ):
        mat, w = panels[g]
        d = interval(mat, r, chosen, "native", w)
        check = confirm["checks"][g]
        close([check[k] for k in ("delta", "lo", "hi", "upper_limit")], [*d, limit])
        ok = bool(d[2] < 0 if limit == 0 else d[2] <= limit)
        assert ok == check["passed"]
        passed.append(ok)
    assert (
        confirm["primary_pass"],
        confirm["all_guardrails_pass"],
        confirm["all_pass"],
    ) == (passed[0], all(passed[1:]), all(passed))
    repo = Path(
        subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], cwd=HERE, text=True
        ).strip()
    )
    p = str(path.relative_to(repo))
    commit = subprocess.check_output(
        ["git", "log", "-1", "--diff-filter=A", "--format=%H", "--", p],
        cwd=repo,
        text=True,
    ).strip()
    assert (
        commit
        and subprocess.check_output(["git", "show", f"{commit}:{p}"], cwd=repo)
        == path.read_bytes()
    )
    assert subprocess.run(
        ["git", "cat-file", "-e", f"{commit}:{(OUT / 'r10.csv').relative_to(repo)}"],
        cwd=repo,
        capture_output=True,
        check=False,
    ).returncode
    return {**rec, "selection_commit": commit, "all_pass": all(passed)}


def check_layout(r, raw):
    base = 1140000 + r * 1000
    std = set(load_preset("standard-v6")["scenarios"])
    extra = {
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
    blocks = [
        ("standard", std, 400, {9: 4, 10: 8}.get(r, 2), ("test",)),
        (
            "focus",
            ("mixed", "case_mix", *extra.get(r, ())),
            100,
            {9: 12, 10: 24}.get(r, 6),
            WORLD,
        ),
    ]
    expected = {
        (c, f, base + d + s, w)
        for c, fs, d, n, worlds in blocks
        for f in fs
        for s in range(n)
        for w in worlds
    }
    grouped = raw.groupby([*KEY, "world"]).method
    assert set(grouped.groups) == expected
    assert (
        grouped.nunique().eq(raw.method.nunique()).all()
        and not raw.duplicated([*KEY, "world", "method"]).any()
    )
    assert raw["round"].eq(r).all()
    return len(expected)


def reconstruct_plans(b, r):
    ids = observed_idx(b)
    np.testing.assert_array_equal(ids, b.screen_idx)
    single, raw, replicated = reconstruct(ids, b.x, b.y, b.env, b.ctx, b.seed)
    w, _action = guarded(b.env, b.ctx)
    ones = np.ones(len(w))
    plans = {
        "native": (False, None, ones),
        "guarded": (False, None, w),
        "single": (single is not None, single, w if single is not None else ones),
        "half": (
            single is not None,
            single,
            0.5 + 0.5 * w if single is not None else ones,
        ),
        "replicated": (
            replicated is not None,
            single,
            w if replicated is not None else ones,
        ),
    }
    if r == 2:
        plans["raw"] = raw is not None, raw, w if raw is not None else ones
    if r == 3:
        shuffled = np.array(b.env, copy=True)
        rng = np.random.default_rng([b.seed, 1143])
        for context in np.unique(b.ctx):
            take = np.flatnonzero(b.ctx == context)
            shuffled[take] = rng.permutation(b.env[take])
        flag, _, _ = reconstruct(ids, b.x, b.y, shuffled, b.ctx, b.seed)
        plans["gate_sham"] = flag is not None, flag, w if flag is not None else ones
    if r == 8:
        coin = np.random.default_rng([b.seed, 1148]).random() < 0.5
        plans["random_gate"] = bool(coin), None, w if coin else ones
    return plans


def audit_source_boundary(b, r, plans):
    """The deployed decision runs with a source-only facade and no target or role fields."""
    from types import SimpleNamespace

    from run import decisions

    source = SimpleNamespace(
        x=b.x, y=b.y, env=b.env, ctx=b.ctx, screen_idx=b.screen_idx, seed=b.seed
    )
    _, selected, _ = decisions(source, r)
    for name, (w, _info) in selected.items():
        np.testing.assert_array_equal(w, plans[name][2])
    return len(selected)


def check_sources(r, raw):
    counters = Counter()
    bayes = {}
    sample = raw.set_index(KEY).sort_index()
    with gzip.open(OUT / f"r{r:02}-source.jsonl.gz", "rt") as stream:
        traces = {tuple(t[k] for k in KEY): t for t in map(json.loads, stream)}
    assert set(sample.index.unique()) == set(traces)
    for b in sources(r):
        key = b.cohort, b.family, b.seed
        mrows = sample.loc[key].set_index(["world", "method"])
        trace = traces[key]
        for k, a in (
            ("source_x_hash", b.x),
            ("source_y_hash", b.y),
            ("env_hash", b.env),
            ("ctx_hash", b.ctx),
            ("screen_hash", b.screen_idx),
        ):
            assert trace[k] == digest(a), (key, k)
        plans = reconstruct_plans(b, r)
        for m, (active, provider, w) in plans.items():
            actual = trace["methods"][m]
            assert (actual["gate_active"], actual["gate_provider"]) == (
                active,
                provider,
            ), (r, key, m, active, provider, actual)
            assert actual["weight_hash"] == digest(w), (r, key, m)
            applied = np.ptp(w) > 0
            assert actual["applied"] == applied
            close(
                [actual["ess"], actual["weight_max"]], [w.sum() ** 2 / (w @ w), w.max()]
            )
            for world, *_ in b.worlds:
                row = mrows.loc[world, m]
                assert bool(row.gate_active) == active and bool(row.applied) == applied
                assert row.weight_hash == actual["weight_hash"]
                if not applied:
                    native = mrows.loc[world, "native"]
                    for k in ("fit_hash", "test_hash", "path_hash", "tree_hash"):
                        assert row[k] == native[k], (key, m, k)
                    counters["uniform_native_hash_pairs"] += 1
            counters["independent_source_decisions"] += 1
        counters["source_banks"] += 1
        if b.cohort == "standard" and b.family in {
            "grouped_rare_event",
            "shared_error_bloc",
            "rare_scene_specialist",
        }:
            assert (
                len(np.unique(b.groups[b.screen_idx]))
                == len(np.unique(b.groups))
                == len(b.screen_idx)
            )
            counters["grouped_diagnostic_samples"] += 1
        base = 1140000 + 1000 * r
        if key not in (
            ("standard", "gated_experts", base + 400),
            ("focus", "mixed", base + 100),
        ):
            continue
        worlds = independent_sample(b)
        for w, _, y, j in worlds:
            if j is not None:
                optimal = np.sum(2 * j[:, 0] * j[:, 1] / j.sum(1))
                close(mrows.loc[w, "oracle_brier"], optimal)
                bayes[w] = optimal
        checked, refits, cheap = native_replay(
            b, mrows, worlds, {m: v[2] for m, v in plans.items()}
        )
        counters["native_forecast_hash_pairs"] += checked
        counters["native_source_refits"] += refits
        counters["cheap_reference_risks"] += cheap
        counters["source_only_api_replays"] += audit_source_boundary(b, r, plans)
    assert counters["source_banks"] == len(traces)
    for w, optimal in bayes.items():
        f = raw[(raw.cohort == "focus") & (raw.world == w)]
        close(f.oracle_brier, np.full(len(f), optimal))
        assert (f.brier >= optimal - 1e-12).all()
    return dict(counters), bayes


def main(r):
    raw = read(r)
    count = check_layout(r, raw)
    totals, bayes = check_sources(r, raw)
    panels, n = panel_check(r, raw, bayes)
    rec = {
        "round": r,
        "score_rows": len(raw),
        "target_conditions": count,
        **totals,
        "paired_intervals": n,
        "standard_families": 30,
        "disjoint_mechanism_groups": 7,
        "application_profiles": 2,
    }
    if r >= 9:
        rec.update(lock_check(r, panels, bayes))
    (OUT / f"r{r:02}-verification.json").write_text(
        json.dumps(rec, allow_nan=False, indent=2) + "\n"
    )
    print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("round", type=int, choices=range(1, 11))
    with threadpool_limits(limits=1):
        main(cli.parse_args().round)

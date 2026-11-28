"""Independently reconstruct pool diagnostics, source routing and held-out comparisons."""

import argparse
import json
import subprocess
from collections import Counter
from hashlib import sha256
from itertools import product
from pathlib import Path
from runpy import run_path
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from banks_auto import banks
from fusionbench.transfer import load_preset
from okapi.pool_diagnostics import pool_fingerprint

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
PAST = run_path(str(HERE.parent / "mechanism-four" / "verify_four.py"))
BASELINES = PAST["OLD"]
KEY = ["cohort", "family", "seed"]
ACTIONS = ["mean", "top_n", "cal_mean", "flat", "cal_pull"]
HELDOUT = {
    "grouped_rare_event", "shared_error_bloc", "rare_scene_specialist", "cloned_shortcut",
    "parity", "similar_classes_38",
}
GROUPED = {"grouped_rare_event", "shared_error_bloc", "rare_scene_specialist"}
FEATURES = [
    "disagreement", "quality_spread", "gain_best", "residual_alignment",
    "effective_rank", "local_complementarity", "calibration_slope", "mean_brier",
]
ALL_FEATURES = [*FEATURES, "gain_best_se", "independent_units"]
MEASURES = ["brier", "expected_brier", "macro_f1", "ap"]


def table(r, name):
    return pd.read_csv(OUT / f"r{r}-{name}.csv", float_precision="round_trip")


def equal(actual, expected):
    np.testing.assert_allclose(
        np.asarray(actual, dtype=float), np.asarray(expected, dtype=float),
        rtol=0, atol=5e-12, equal_nan=True,
    )


def expected_keys(r):
    n = (8, 8, 16)[r - 1]
    families = set(load_preset("standard-v6")["scenarios"])
    assert len(families) == 30 and HELDOUT <= families
    cohorts = [
        ("standard", families, 400, (4, 3, 6)[r - 1]),
        ("factorial", [f"h{h}c{c}s{s}" for h, c, s in product(range(2), repeat=3)], 0, n),
        ("clustered", ["scene16", "scene128"], 100, n),
        ("controlled", ["weak", "strong"], 200, n),
    ]
    if r == 3:
        cohorts.append(("nfl", ["best_world", "mean_world"], 800, n))
    base = 610000 + 1000 * r
    return {
        (cohort, f, base + offset + i)
        for cohort, fs, offset, count in cohorts for f in fs for i in range(count)
    }


def explicit_features(b):
    x, y = b.x, b.y
    n, m, k = x.shape
    target = np.eye(k)[y]
    avg = x.sum(axis=1) / m
    err = (x - target[:, None, :]).transpose(1, 0, 2)
    losses = np.einsum("mnk,mnk->mn", err, err)
    quality = losses.mean(1)
    ensemble_loss = ((avg - target) ** 2).sum(1)
    flat = err.reshape(m, -1)
    norm = np.linalg.norm(flat, axis=1)
    denom = norm[:, None] * norm[None, :]
    cos = np.divide(flat @ flat.T, denom, out=np.zeros_like(denom), where=denom > 0)
    centered = (x.transpose(1, 0, 2) - 1 / k).reshape(m, -1)
    eigen = np.linalg.eigvalsh(centered @ centered.T)
    energy = np.sum((avg - 1 / k) ** 2)
    slope = ((avg - 1 / k) * (target - 1 / k)).sum() / energy if energy > n * np.finfo(float).eps else 1
    tied = np.abs(quality - quality.min()) <= 1e-12
    g = ensemble_loss - losses[tied].mean(0)
    groups = b.groups if b.groups is not None else np.arange(n)
    ids, inv = np.unique(groups, return_inverse=True)
    count = len(ids)
    residual_sums = np.bincount(inv, weights=g - g.mean())
    se = np.linalg.norm(residual_sums) / n * np.sqrt(count / (count - 1)) if count > 1 else 0
    return [
        np.mean(np.sum((x - avg[:, None, :]) ** 2, axis=2)), quality.std(),
        ensemble_loss.mean() - quality.min(), cos[np.triu_indices(m, 1)].mean() if m > 1 else 0,
        eigen.sum() ** 2 / (m * (eigen @ eigen)) if np.any(eigen) else 0,
        quality.min() - losses.min(0).mean(), np.clip(slope, -4, 4), ensemble_loss.mean(),
        se, count,
    ]


def verify_fingerprint(b, row):
    independent = explicit_features(b)
    equal(row[ALL_FEATURES], independent)
    n, m, k = b.x.shape
    rng = np.random.default_rng(b.context["seed"] + 97)
    rows, providers, classes = (rng.permutation(dim) for dim in (n, m, k))
    fp = pool_fingerprint(
        b.x[rows][:, providers][:, :, classes], np.argsort(classes)[b.y[rows]],
        groups=b.groups[rows] if b.groups is not None else None,
    )
    equal([fp[name] for name in ALL_FEATURES], independent)


def follow(row, node):
    while "feature" in node:
        node = node["left" if row[node["feature"]] <= node["threshold"] else "right"]
    return node["action"]


def check_node(frame, node, depth=0):
    # All eligible R1 families contributed exactly four banks. Preserve these original
    # weights in descendants instead of reweighting a family after every split.
    costs = frame[ACTIONS].mean()
    best_action = costs.idxmin()
    assert node["banks"] == len(frame) and node["families"] == frame.family.nunique()
    assert node["action"] == best_action
    equal([node["mean_losses"][c] for c in ACTIONS], costs)
    chosen = None
    best_gain = 0
    if depth < 2:
        for name in FEATURES:
            unique, counts = np.unique(frame[name], return_counts=True)
            for i in sorted(set(np.searchsorted(counts.cumsum(), np.arange(1, 5) * .2 * len(frame)))):
                if i + 1 == len(unique):
                    continue
                threshold = unique[i] / 2 + unique[i + 1] / 2
                parts = (frame[frame[name] <= threshold], frame[frame[name] > threshold])
                if any(len(p) < len(frame) / 8 or p.family.nunique() < 4 for p in parts):
                    continue
                gain = costs.min() - sum(p[ACTIONS].sum().min() for p in parts) / len(frame)
                if gain >= .001 and gain > best_gain + 1e-12:
                    chosen, best_gain = (name, threshold, parts), gain
    if chosen is None:
        assert "feature" not in node
        return
    feature, threshold, parts = chosen
    assert node["feature"] == feature
    equal([node["threshold"], node["gain"]], [threshold, best_gain])
    for part, child in zip(parts, ("left", "right"), strict=True):
        check_node(part, node[child], depth + 1)


def verify_policy():
    frozen = json.loads((OUT / "r1-policy.json").read_text())
    family_set = set(load_preset("standard-v6")["scenarios"]) - HELDOUT
    assert set(frozen["training_families"]) == family_set and len(family_set) == 24
    assert set(frozen["excluded_families"]) == HELDOUT and frozen["training_round"] == 1
    for name, expected in frozen["input_sha256"].items():
        assert sha256((OUT / name).read_bytes()).hexdigest() == expected
    fps, scores = table(1, "fingerprints"), table(1, "scores")
    src = fps[(fps.cohort == "standard") & fps.family.isin(family_set)]
    raw = scores[(scores.cohort == "standard") & scores.family.isin(family_set)]
    risk = raw.pivot(index=["family", "seed"], columns="method", values="brier")[ACTIONS]
    frame = src.set_index(["family", "seed"])[FEATURES].join(risk).sort_index().reset_index()
    assert len(frame) == frozen["row_count"] == 96 and frame.groupby("family").size().eq(4).all()
    assert sha256(frame.to_csv(index=False).encode()).hexdigest() == frozen["training_sha256"]
    policy = frozen["policy"]
    assert policy["features"] == FEATURES and policy["actions"] == ACTIONS
    check_node(frame, policy["tree"])
    assert frozen["constant"] == frame[ACTIONS].mean().idxmin()
    return frozen


def verify_routes(rows, fp, frozen):
    fixed = frozen["constant"]
    oracle = rows.loc[ACTIONS, "brier"].idxmin()
    for method in ("constant", "tree_rule", "cv_rule", "portfolio_oracle"):
        if method not in rows.index:
            continue
        action = rows.at[method, "action"]
        assert action in ACTIONS
        if method != "cv_rule":
            assert action == {"constant": fixed, "tree_rule": follow(fp, frozen["policy"]["tree"]),
                              "portfolio_oracle": oracle}[method]
        for col in ("fit_hash", "test_hash"):
            assert rows.at[method, col] == rows.at[action, col], (method, action, col)
        equal(rows.loc[method, [*MEASURES, "fit_brier", "nodes", "providers"]],
              rows.loc[action, [*MEASURES, "fit_brier", "nodes", "providers"]])
    return sum(m in rows.index for m in ("constant", "tree_rule", "cv_rule", "portfolio_oracle"))


def verify_witness(pairs):
    bounds, excesses = [], {}
    for worlds in pairs.values():
        (a, ar), (b, br) = worlds["best_world"], worlds["mean_world"]
        for prop in ("x", "y", "audit_x"):
            np.testing.assert_array_equal(getattr(a, prop), getattr(b, prop))
        assert np.any(a.audit_y != b.audit_y)
        pa, pb = a.audit_x[:, 0, 1], a.audit_x[:, :, 1].mean(1)
        equal(a.q, pa)
        equal(b.q, pb)
        bound = .5 * np.mean((pa - pb) ** 2)
        bounds.append(bound)
        for bank in (a, b):
            equal(bank.context["common_rule_excess_bound"], bound)
        for qhat in (pa, pb, (pa + pb) / 2):
            risk = np.mean((qhat - pa) ** 2 + (qhat - pb) ** 2)
            equal(risk, bound + 2 * np.mean((qhat - (pa + pb) / 2) ** 2))
        for m in ar.index.difference(["portfolio_oracle"]):
            assert ar.at[m, "action"] == br.at[m, "action"]
            assert ar.at[m, "test_hash"] == br.at[m, "test_hash"]
            assert ar.at[m, "fit_hash"] == br.at[m, "fit_hash"]
            excess = np.mean([ar.at[m, "expected_brier"] - np.mean(2 * pa * (1 - pa)),
                              br.at[m, "expected_brier"] - np.mean(2 * pb * (1 - pb))])
            assert excess + 5e-12 >= bound, (m, excess, bound)
            excesses.setdefault(m, []).append(excess)
    assert len(pairs) == 16
    return {"paired_sources": len(pairs), "mean_exact_lower_bound": np.mean(bounds),
            "min_exact_lower_bound": min(bounds), "max_exact_lower_bound": max(bounds),
            "mean_excess_over_bayes": {m: np.mean(v) for m, v in excesses.items()}}


def check_banks(r, frozen):
    scores, features = table(r, "scores"), table(r, "fingerprints").set_index(KEY)
    keys = expected_keys(r)
    assert len(keys) == (216, 186, 404)[r - 1]
    assert features.index.is_unique and set(features.index) == keys
    assert scores["round"].eq(r).all() and not scores.duplicated([*KEY, "method"]).any()
    grouped = scores.groupby(KEY, sort=False)
    assert set(grouped.groups) == keys
    counts, paired, witness = Counter(), {}, {}
    for b in banks(r):
        key = tuple(b.context[k] for k in KEY)
        rows = grouped.get_group(key).set_index("method")
        PAST["check_bank"](b, rows, paired)
        verify_fingerprint(b, features.loc[key])
        counts["banks"] += 1
        counts["routed_hash_pairs"] += verify_routes(rows, features.loc[key], frozen)
        fit, test = BASELINES["controls"](b.x, b.y, [b.x, b.audit_x])
        for name in ("mean", "top_n", "lr"):
            equal(rows.loc[name, MEASURES], PAST["checked_measures"](b, test[name]))
            equal(rows.at[name, "fit_brier"], BASELINES["empirical"](fit[name], b.y))
            counts["independently_refitted_cheap_controls"] += 1
        if key[0] == "nfl":
            witness.setdefault(key[2], {})[key[1]] = (b, rows)
    assert counts["banks"] == len(keys)
    result = {"counts": dict(counts)}
    if witness:
        result["no_free_lunch"] = verify_witness(witness)
    return result


def verify_trace(b, rows, trace):
    cv = trace["cv"]
    risks = cv["oof_brier"]
    best = min(ACTIONS, key=risks.get)
    assert cv["best"] == best and cv["oof_fits"] == 15
    eligible = [a for a in ACTIONS if risks[a] - risks[best] <= cv["paired_se"][a] + 1e-12]
    assert cv["eligible"] == eligible and rows.at["cv_rule", "action"] == eligible[0]
    n_units = len(np.unique(b.groups)) if b.groups is not None else len(b.y)
    assert n_units == cv["n_units"] and len(cv["folds"]) == 3
    assert sum(f["held_units"] for f in cv["folds"]) == n_units
    assert sum(f["held_rows"] for f in cv["folds"]) == len(b.y)
    assert all(f["held_rows"] + f["train_rows"] == len(b.y) for f in cv["folds"])


def outcome_probe(r, b, rows, frozen):
    """Exercise the normal runner with protected audit fields and reversed outcomes."""
    from types import SimpleNamespace
    from unittest.mock import patch
    import run_auto
    from portfolio import fit_actions

    phase = {"scoring": False}
    source_only = {"x", "y", "groups"}

    class Guard:
        def __getattr__(self, key):
            if not phase["scoring"] and key not in source_only:
                raise AssertionError(f"{key} accessed during a deployable fit/decision")
            return getattr(changed, key)

    changed = SimpleNamespace(**vars(b))
    changed.audit_y = (b.audit_y + 1) % b.x.shape[2]
    changed.q = None if b.q is None else 1 - b.q
    # The runner is allowed to read scores, hashes and simulator metadata only after fitting.
    def source_fit(*args, **kwargs):
        assert not phase["scoring"]
        return fit_actions(*args, **kwargs)

    def scoring(r, bank, models):
        phase["scoring"] = True
        return original(r, bank, models)

    original = run_auto.score_models
    with patch("portfolio.fit_actions", source_fit), patch.object(run_auto, "score_models", scoring):
        altered, _, _ = run_auto.measure((r, Guard(), frozen if r > 1 else None))
    altered = pd.DataFrame(altered).set_index("method")
    methods = rows.index.difference(["portfolio_oracle"])
    for name in methods:
        for k in ("action", "fit_hash", "test_hash"):
            assert altered.at[name, k] == rows.at[name, k], (name, k)
    assert abs(altered.at["mean", "brier"] - rows.at["mean", "brier"]) > 1e-7
    assert abs(altered.at["mean", "expected_brier"] - rows.at["mean", "expected_brier"]) > 1e-7
    return len(methods)


def panels(s):
    std = s[s.cohort == "standard"]
    yield "standard", std
    yield "trained24", std[~std.family.isin(HELDOUT)]
    yield "heldout6", std[std.family.isin(HELDOUT)]
    yield "legacy27", std[~std.family.isin(GROUPED)]
    yield "factorial", s[s.cohort == "factorial"]
    yield from s[s.cohort != "standard"].groupby("family")


def check_summaries(r):
    scores = table(r, "scores")
    contrasts = table(r, "contrasts").query("metric == 'brier'")
    levels = table(r, "levels").set_index(["group", "method"])
    fam = table(r, "families").set_index(["family", "method"]).sort_index()
    expected = scores[scores.cohort == "standard"].groupby(["family", "method"])[MEASURES].mean()
    equal(fam.loc[expected.index, MEASURES], expected)
    assert set(fam.index) == set(expected.index)
    mix = table(r, "action-mix").set_index(["group", "method"]) if r >= 2 else None
    checked = 0
    checks = {}
    for name, part in panels(scores):
        values = part.pivot(index=["family", "seed"], columns="method", values="brier").sort_index()
        family_n = values.index.get_level_values("family").nunique()
        reps = values.index.get_level_values("seed").nunique()
        costs = values[ACTIONS]
        if name == "factorial":
            values = values.groupby(level="seed").mean()
            family_n = 1
        methods = list(values)
        avg = values.mean()
        equal(levels.loc[[(name, m) for m in methods], "brier"], avg)
        # Independently index original measurements at resampled positions.
        rng = np.random.default_rng(73000 + r)
        indices = rng.integers(reps, size=(10_000, family_n, reps))
        indices += np.arange(family_n)[None, :, None] * reps
        indices = indices.reshape(len(indices), -1)
        simulated = {m: values[m].to_numpy()[indices].mean(1) for m in methods}
        for row in contrasts[contrasts.group == name].itertuples():
            diff = simulated[row.method] - simulated[row.reference]
            lo, hi = np.quantile(diff, [.025, .975])
            delta = avg[row.method] - avg[row.reference]
            equal([row.delta, row.lo, row.hi], [delta, lo, hi])
            assert row.resampling_units == len(values)
            checked += 1
            if row.reference == "constant":
                checks[name, row.method] = [float(delta), float(lo), float(hi)]
        if mix is None:
            continue
        for method in mix.loc[name].index:
            chosen = part[part.method == method].set_index(["family", "seed"]).loc[costs.index, "action"]
            frequencies = chosen.value_counts(normalize=True).reindex(ACTIONS, fill_value=0)
            random_loss = costs.mean() @ frequencies
            within_loss = np.mean([
                frame.mean() @ chosen.loc[f].value_counts(normalize=True).reindex(ACTIONS, fill_value=0)
                for f, frame in ((f, costs.xs(f)) for f in costs.index.get_level_values("family").unique())
            ])
            rec = mix.loc[name, method]
            equal([rec.actual, rec.expected_random, rec.delta_actual_minus_random,
                   rec.expected_within_family_random, rec.delta_actual_minus_within_family_random],
                  [avg[method], random_loss, avg[method] - random_loss, within_loss, avg[method] - within_loss])
    assert checked == len(contrasts)
    return checked, checks


def verify_lock(r, checks):
    lock = json.loads((HERE / "selection.json").read_text())
    rr = table(2, "scores")
    panel = rr[(rr.cohort == "standard") & ~rr.family.isin(HELDOUT)]
    risks = panel.pivot_table(index="family", columns="method", values="brier")
    assert len(risks) == 24 and len(panel.seed.unique()) == 3
    choices = ["tree_rule", "cv_rule"]
    avg = risks[choices].mean()
    equal([lock["trained24_risks"][a] for a in choices], avg)
    assert lock["chosen"] == avg.idxmin()
    for name, h in lock["sha256"].items():
        assert sha256((OUT / name).read_bytes()).hexdigest() == h
    files = [OUT / "r1-policy.json"] + ([HERE / "selection.json"] if r == 3 else [])
    git_root = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=HERE, text=True).strip())
    commits = {}
    for file in files:
        path = file.relative_to(git_root)
        commit = subprocess.check_output(["git", "log", "-1", "--format=%H", "--diff-filter=A", "--", str(path)],
                                         cwd=git_root, text=True).strip()
        assert commit
        blob = subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=git_root)
        assert blob == file.read_bytes()
        next_score = OUT / ("r2-scores.csv" if file == OUT / "r1-policy.json" else "r3-scores.csv")
        assert subprocess.run(["git", "cat-file", "-e", f"{commit}:{next_score.relative_to(git_root)}"],
                              cwd=git_root, capture_output=True).returncode != 0
        commits[file.name] = commit
    result = {"chosen_on_r2_trained24": lock["chosen"], "frozen_input_commits": commits}
    if r == 3:
        confirm = json.loads((OUT / "confirmation.json").read_text())
        chosen = lock["chosen"]
        passes = [checks[g, chosen][2] < 0 for g in ("standard", "heldout6")]
        passes.append(checks["scene16", chosen][2] <= .005)
        assert confirm["chosen"] == chosen and confirm["generalization_pass"] == all(passes)
        assert [confirm[k] for k in ("standard_pass", "heldout_pass", "scene16_within_limit")] == passes
        for g, rec in confirm["changes_vs_constant"].items():
            equal([rec["delta"], rec["lo"], rec["hi"]], checks[g, chosen])
        result["generalization_pass"] = all(passes)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", type=int, choices=(1, 2, 3), required=True)
    r = parser.parse_args().round
    start = perf_counter()
    with threadpool_limits(limits=1):
        policy = verify_policy()
        result = check_banks(r, policy)
        num_contrasts, checks = check_summaries(r)
        result.update(round=r, independent_paired_brier_intervals=num_contrasts,
                      verified_policy_training_banks=96, verified_policy_training_families=24)
        if r >= 2:
            result.update(verify_lock(r, checks))
    result["verification_seconds"] = perf_counter() - start
    path = OUT / f"verification-r{r}.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(path.name, json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()

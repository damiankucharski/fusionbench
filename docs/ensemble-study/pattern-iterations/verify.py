"""Audit paired study estimates and reproduce saved native and reference predictions."""

import argparse
import csv
import gzip
import json
import runpy
from collections import defaultdict
from itertools import combinations
from math import sqrt
from pathlib import Path
from statistics import fmean, stdev

import numpy as np
from scipy.stats import t
from threadpoolctl import threadpool_limits

from fusionbench.eval.full_suite_point import SHIFTS
from fusionbench.transfer import load_preset

HERE = Path(__file__).resolve().parent
transfer = runpy.run_path(str(HERE.parent / "transfer-grid" / "verify.py"))
reference = runpy.run_path(str(HERE.parent / "oof-gp-stack" / "verify.py"))
interpret = transfer["NATIVE"]


def records(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as stream:
        if path.name.endswith(".jsonl.gz") or path.suffix == ".jsonl":
            return [json.loads(line) for line in stream]
        return list(csv.DictReader(stream))


def near(expected, recorded, tol=1e-12):
    np.testing.assert_allclose(recorded, expected, atol=tol, rtol=0)


def ci(values):
    center = fmean(values)
    width = stdev(values) * float(t.ppf(0.975, len(values) - 1)) / sqrt(len(values))
    return center, center - width, center + width


def brier(p, y):
    assert (
        p.ndim == 2 and len(p) == len(y) and np.isfinite(p).all() and p.min() >= -1e-12
    )
    near(np.ones(len(y)), p.sum(1))
    return float(np.square(p - np.eye(p.shape[1])[y]).sum(1).mean())


ARMS = ("native", "challenger")
CANDIDATES = ("flat", "augmented", "gp")
CONTROLS = ("mean", "top_n", "prob_lr", "logit")
METHODS = (*ARMS, *CONTROLS)
BASE_SEED = {1: 2010000, 2: 2110000, 3: 2210000}
SLOTS = {
    name: spec["seed_slot"]
    for name, spec in load_preset("standard")["scenarios"].items()
}
assert len(SLOTS) == 19 and set(SLOTS.values()) == set(range(19))
assert len(SHIFTS) == 9 and set(SHIFTS) <= set(SLOTS)


def training_split(seed, n=640):
    indices = np.random.default_rng([seed, 51]).permutation(n)
    return indices[: 7 * n // 10], indices[7 * n // 10 :]


def repeats(round_no):
    return 10 if round_no == 3 else 6


def regular_conditions(name):
    return [("clean", 0.0), ("provider_swap", 0.5), ("log_jitter", 0.5)] + (
        [("canonical_shift", 1.0)] if name in SHIFTS else []
    )


def score_id(row):
    return (
        row["scenario"],
        int(row["replicate"]),
        int(row["gp_seed"]),
        row["mode"],
        float(row["severity"]),
        row["method"],
    )


def lookup_unique(rows, key):
    table = {key(row): row for row in rows}
    assert len(rows) == len(table)
    return table


def view_loss(loss, name, rep, gp_seed, panel, method):
    def score(condition, severity):
        return loss[name, rep, gp_seed, condition, severity, method]

    if panel in ("joint", "stress"):
        stress = (score("provider_swap", 0.5) + score("log_jitter", 0.5)) / 2
        return stress if panel == "stress" else (stress + score("clean", 0.0)) / 2
    severity = {"clean": 0.0, "canonical_shift": 1.0}.get(panel, 0.5)
    return score(panel, severity)


def bank_effects(round_no, loss):
    """Pair first, then average fixed families inside each independent data replicate."""
    panels = [
        "clean",
        "provider_swap",
        "log_jitter",
        "canonical_shift",
        "stress",
        "joint",
    ]
    if round_no == 3:
        panels.append("coherent_pair")
    contrasts = {}
    means = {}
    for panel in panels:
        families = list(SHIFTS if panel == "canonical_shift" else SLOTS)
        by_seed = {}
        for name in families:
            for rep in range(repeats(round_no)):
                gp_seed = BASE_SEED[round_no] + 1000 * SLOTS[name] + rep
                for method in METHODS:
                    by_seed[name, rep, method] = view_loss(
                        loss, name, rep, gp_seed, panel, method
                    )
        for method in METHODS:
            means[panel, method] = fmean(
                by_seed[n, r, method]
                for n in families
                for r in range(repeats(round_no))
            )
        for ref in ("native", *CONTROLS):
            diffs = {
                (n, r): by_seed[n, r, "challenger"] - by_seed[n, r, ref]
                for n in families
                for r in range(repeats(round_no))
            }
            by_rep = [
                fmean(diffs[n, r] for n in families) for r in range(repeats(round_no))
            ]
            wins = sum(
                fmean(diffs[n, r] for r in range(repeats(round_no))) < 0
                for n in families
            )
            contrasts[panel, ref] = (*ci(by_rep), wins, len(families))
    return contrasts, means


def expected_jobs(round_no):
    for name, index in SLOTS.items():
        for rep in range(repeats(round_no)):
            seed = BASE_SEED[round_no] + 1000 * index + rep
            search_seeds = [seed]
            if round_no == 3 and rep in (0, 1):
                search_seeds.append(seed + 10_000_000)
            for gp_seed in search_seeds:
                yield name, rep, seed, gp_seed


def topology(tree):
    """Only the final log-evidence head's fitted parameters may change during refit."""
    tree = dict(tree)
    children = list(tree["children"])
    if children and children[-1]["type"] in (
        "WeightedLogitMeanNode",
        "BinaryAffineNode",
    ):
        head = {
            k: v
            for k, v in children[-1].items()
            if k not in ("weights", "coefficients", "intercept")
        }
        children[-1] = head
    return {**tree, "children": children}


def verify_rows(round_no, score_rows, diagnostic_rows):
    scores = lookup_unique(score_rows, score_id)
    diag = lookup_unique(diagnostic_rows, score_id)
    expected_scores = set()
    expected_diag = set()
    for name, rep, seed, gp_seed in expected_jobs(round_no):
        views = regular_conditions(name) + (
            [("coherent_pair", 0.5)] if round_no == 3 else []
        )
        chosen = {}
        for mode, level in views:
            key = (name, rep, gp_seed, mode, level)
            for m in METHODS:
                tag = key + (m,)
                expected_scores.add(tag)
                rec = scores[tag]
                assert (
                    int(rec["round"]) == round_no
                    and int(rec["data_seed"]) == seed
                    and 0 <= float(rec["brier"]) <= 2
                )
                if round_no == 3 and gp_seed != seed and m in CONTROLS:
                    assert float(rec["brier"]) == float(
                        scores[name, rep, seed, mode, level, m]["brier"]
                    )
            for m in ARMS:
                tag = key + (m,)
                expected_diag.add(tag)
                r = diag[tag]
                val = {a: float(r["val_" + a]) for a in CANDIDATES}
                discovery = {a: float(r["discovery_" + a]) for a in CANDIDATES}
                refit = {a: float(r["refit_" + a]) for a in CANDIDATES}
                assert all(
                    0 <= x <= 2
                    for stage in (val, discovery, refit)
                    for x in stage.values()
                )
                assert (
                    int(r["round"]) == round_no
                    and int(r["data_seed"]) == seed
                    and r["chosen"] in CANDIDATES
                )
                signature = (r["chosen"], val)
                assert chosen.setdefault(m, signature) == signature
                selected = r["chosen"]
                near(min(discovery.values()), float(r["oracle_discovery_brier"]))
                near(min(refit.values()), float(r["oracle_refit_brier"]))
                near(
                    discovery[selected] - min(discovery.values()),
                    float(r["chosen_discovery_regret"]),
                )
                near(
                    refit[selected] - min(refit.values()),
                    float(r["chosen_refit_regret"]),
                )
                for stage, values in [("discovery", discovery), ("refit", refit)]:
                    assert (values[selected] - min(values.values()) <= 1e-12) == (
                        r["chosen_ties_" + stage + "_oracle"] == "True"
                    )
                if mode != "coherent_pair":
                    assert r["coherent_providers"] == ""
                near(refit[selected], float(scores[tag]["brier"]))
                near(discovery["gp"], refit["gp"])
    assert set(scores) == expected_scores and set(diag) == expected_diag
    return {k: float(v["brier"]) for k, v in scores.items()}, diag


def simple_baselines(train, y, views):
    """Refit public baselines on raw development labels, independently of the study runner."""
    ranking, losses = reference["top_prefix"](train, y)
    top = ranking[: 1 + np.argmin(losses)]
    names = list(views)
    tests = np.concatenate([views[name][0] for name in names])
    offsets = np.cumsum([len(views[name][0]) for name in names])[:-1]
    predictions = {
        "mean": tests.mean(axis=1),
        "top_n": reference["average_prefix"](tests, top),
        "prob_lr": reference["logistic"](train, y, tests, False),
        "logit": reference["logistic"](train, y, tests, True),
    }
    results = {}
    for m, p in predictions.items():
        for name, chunk in zip(names, np.split(p, offsets), strict=True):
            results[name, m] = brier(chunk, views[name][1])
    return results


def coherent_providers(dev):
    # Correlation of the concatenated provider channels, centering each channel
    # on source rows. Enumerate pairs in provider order, before drawing donors.
    vectors = [
        (dev[:, j, :] - dev[:, j, :].mean(0)).reshape(-1) for j in range(dev.shape[1])
    ]
    unit = [
        x / np.linalg.norm(x) if np.linalg.norm(x) else np.zeros_like(x)
        for x in vectors
    ]
    pairs = [(i, j) for i in range(len(unit)) for j in range(i + 1, len(unit))]
    return max(pairs, key=lambda p: unit[p[0]] @ unit[p[1]])


def heldout_coherent(dev, clean, seed):
    i, j = coherent_providers(dev)
    rng = np.random.default_rng([seed, 903])
    perm = rng.permutation(len(clean))
    donor = np.empty_like(perm)
    donor[perm] = np.r_[perm[-1], perm[:-1]]
    take = np.flatnonzero(rng.random(len(clean)) < 0.5)
    altered = clean.copy()
    for col in (i, j):
        altered[take, col] = clean[donor[take], col]
    return altered, (i, j)


def provenance_key(row):
    return (row["scenario"], int(row["replicate"]), int(row["gp_seed"]), row["method"])


def check_head_fitness(tree, x, y, stage, strength):
    """Rebuild the raw objective and the native weight penalty from saved roots."""
    head = tree["children"][-1]
    roots = [{**tree, "children": tree["children"][:-1]}, *head["children"]]
    coefficients = head.get("coefficients", head.get("weights"))
    variances = []
    for root in roots:
        p = interpret(root, x).clip(1.0e-6, 1 - 1.0e-6)
        if p.shape[1] == 2:
            z = np.log(p[:, 1] / (1.0 - p[:, 1]))
        else:
            z = np.log(p)
            z -= z.mean(1, keepdims=True)
        variances.append(np.var(z, axis=0).mean())
    penalty = (
        sum(w * w * v for w, v in zip(coefficients, variances, strict=True))
        * strength
        / len(y)
    )
    raw = 1.0 - 0.5 * brier(interpret(tree, x), y)
    near(
        [raw, penalty, raw - penalty],
        [stage[k] for k in ("raw_score", "penalty", "penalized_score")],
    )


def sample_audit(round_no, manifests, loss, diagnostics, costs, settings):
    entries = lookup_unique(manifests, provenance_key)
    expected = set()
    checks = 0
    nref = 0
    cache = {}
    for name, rep, seed, gp_seed in expected_jobs(round_no):
        if rep:
            continue
        if name not in cache:
            x, y = transfer["data"](name, seed, 0, 640)
            conditions = regular_conditions(name)
            views = {
                (m, s): (z, labels)
                for m, s, z, labels in transfer["replay_test"](name, seed, 2048)
                if (m, s) in conditions
            }
            if round_no == 3:
                clean, ly = views["clean", 0.0]
                unknown, pair = heldout_coherent(x, clean, seed)
                views["coherent_pair", 0.5] = unknown, ly
                assert (
                    tuple(
                        map(
                            int,
                            diagnostics[
                                name, rep, gp_seed, "coherent_pair", 0.5, "native"
                            ]["coherent_providers"].split(","),
                        )
                    )
                    == pair
                )
            baselines = simple_baselines(x, y, views)
            cache[name] = x, y, views, baselines
        x, y, views, baselines = cache[name]
        for (view, m), b in baselines.items():
            near(b, loss[name, rep, gp_seed, *view, m])
            nref += 1
        d, v = training_split(seed)
        for m in ARMS:
            tag = name, rep, gp_seed, m
            expected.add(tag)
            row = entries[tag]
            assert (
                int(row["data_seed"]) == seed
                and row["kwargs"] == settings[m + "_kwargs"]
            )
            fit_stats = json.loads(costs[tag]["counts_json"])
            np.testing.assert_array_equal(row["discovery_indices"], d)
            np.testing.assert_array_equal(row["validation_indices"], v)
            assert not set(d) & set(v) and len(d) == 448 and len(v) == 192
            disc = row["discovery_trees"]
            final = row["trees"]
            assert (
                set(disc) == set(final) == set(CANDIDATES) and disc["gp"] == final["gp"]
            )
            for kind in CANDIDATES:
                assert topology(disc[kind]) == topology(final[kind])
                if kind != "gp" and "head_fit_scores" in fit_stats:
                    ledger = fit_stats["head_fit_scores"]
                    strength = fit_stats["l2_strength"]
                    check_head_fitness(
                        disc[kind], x[d], y[d], ledger[kind + "_discovery"], strength
                    )
                    check_head_fitness(
                        final[kind], x, y, ledger[kind + "_refit"], strength
                    )
                probs = interpret(disc[kind], x[v])
                loss_v = brier(probs, y[v])
                near(loss_v, row["val_brier"][kind])
                assert reference["digest"](probs, "<f8") == row["val_sha256"][kind]
                for view, (test, target) in views.items():
                    rec = diagnostics[name, rep, gp_seed, *view, m]
                    assert row["chosen"] == rec["chosen"]
                    near(loss_v, float(rec["val_" + kind]))
                    near(
                        brier(interpret(disc[kind], test), target),
                        float(rec["discovery_" + kind]),
                    )
                    near(
                        brier(interpret(final[kind], test), target),
                        float(rec["refit_" + kind]),
                    )
                    checks += 2
    assert set(entries) == expected
    return len(entries), checks, nref


def check_pairs(stats, row, x, y, discovery):
    info = stats["pair_basis"]
    ids = list(range(x.shape[1]))
    if len(ids) > 12:
        ids = sorted(
            sorted(ids, key=lambda i: brier(x[discovery, i], y[discovery]))[:12]
        )
    assert info["provider_indices"] == ids
    proposals = [list(pair) for pair in combinations(ids, 2)]
    assert info["pairs"] == proposals and info["programs"] == int(
        row["pair_programs"]
    ) == len(proposals)
    assert (
        info["evaluations"]
        == int(row["pair_ranking_calls"])
        == (x.shape[1] if x.shape[1] > 12 else 0)
    )
    assert int(row["pair_screen_calls"]) == info["screen_calls"]
    assert (
        info["additional_discovery_calls"] == info["screen_calls"] + info["evaluations"]
    )
    chosen = [
        index - info["gp_archive_size"] for index in info["selected_archive_indices"]
    ]
    assert all(0 <= j < len(proposals) for j in chosen)
    assert (
        [proposals[i] for i in chosen]
        == info["selected_pairs"]
        == json.loads(row["selected_pairs"])
    )
    assert (
        len(chosen)
        == int(row["selected_pair_count"])
        == info["selected_pair_count"]
        <= 3
    )


def verify_costs(round_no, settings, rows, diagnostics):
    index = lookup_unique(rows, provenance_key)
    expected = set()
    for name, rep, seed, gp_seed in expected_jobs(round_no):
        x, y = transfer["data"](name, seed, 0, 640)
        discovery, validation = training_split(seed)
        shared_digests = set()
        for method in METHODS:
            key = name, rep, gp_seed, method
            expected.add(key)
            row = index[key]
            assert int(row["round"]) == round_no and int(row["data_seed"]) == seed
            assert int(row["development_rows"]) == 640
            if method in CONTROLS:
                assert bool(row["reference_reused_from_seed"]) == (gp_seed != seed)
                if gp_seed != seed:
                    assert int(row["reference_reused_from_seed"]) == seed
                    assert (
                        float(row["fit_seconds"]) == float(row["test_seconds"]) == 0.0
                    )
                if method == "top_n":
                    assert int(row["top_n_prefixes"]) == x.shape[1]
                    assert 1 <= int(row["top_n_selected"]) <= x.shape[1]
                continue
            stats = json.loads(row["counts_json"])
            assert stats["fitness"] == "one_minus_half_full_brier"
            for k, value in [
                ("discovery_rows", 448),
                ("validation_rows", 192),
                ("refit_rows", 640),
            ]:
                assert stats[k] == int(row[k]) == value
            assert stats["discovery_sha256"] == reference["digest"](discovery, "<f8")
            assert stats["validation_sha256"] == reference["digest"](validation, "<f8")
            calls = stats["score_calls"]
            assert (
                sum(calls.values())
                == stats["total_score_calls"]
                == int(row["total_score_calls"])
            )
            assert calls["gp_discovery"] == int(row["gp_discovery_calls"]) == 600
            assert (
                calls["gp_validation"]
                == int(row["gp_unique_programs"])
                == stats["gp_unique_programs"]
            )
            assert calls["family_validation"] == int(row["validation_candidates"]) == 3
            assert (
                len(stats["screen"]["selected"]) == int(row["selected_features"]) <= 3
            )
            assert row["gp_path_digest"] == stats["gp_path_digest"]
            shared_digests.add(row["gp_path_digest"])
            clean = diagnostics[name, rep, gp_seed, "clean", 0.0, method]
            val_scores = stats["validation_scores"]
            for candidate in CANDIDATES:
                near(2 * (1 - val_scores[candidate]), float(clean["val_" + candidate]))
            assert (
                row["chosen"]
                == clean["chosen"]
                == max(CANDIDATES, key=val_scores.__getitem__)
            )
            config = settings[method + "_kwargs"]
            assert stats.get("l2_strength", 0) == config.get("l2_strength", 0)
            assert ("head_fit_scores" in stats) == bool(config.get("l2_strength"))
            for stage in stats.get("head_fit_scores", {}).values():
                assert 0 <= stage["raw_score"] <= 1 and stage["penalty"] >= 0
                near(stage["raw_score"] - stage["penalty"], stage["penalized_score"])
            assert ("pair_basis" in stats) == bool(config.get("pair_basis"))
            if "pair_basis" in stats:
                check_pairs(stats, row, x, y, discovery)
            if round_no == 3:
                coherent = diagnostics[name, rep, gp_seed, "coherent_pair", 0.5, method]
                original = diagnostics[name, rep, seed, "coherent_pair", 0.5, "native"]
                assert coherent["coherent_providers"] == original["coherent_providers"]
                if gp_seed == seed and method == "native":
                    assert coherent_providers(x) == tuple(
                        int(i) for i in coherent["coherent_providers"].split(",")
                    )
                if gp_seed != seed:
                    assert (
                        row["gp_path_digest"]
                        != index[name, rep, seed, method]["gp_path_digest"]
                    )
        assert len(shared_digests) == 1
    assert set(index) == expected
    return index


def check_summaries(round_no, folder, contrasts, means, diagnostics, loss):
    report = lookup_unique(
        records(folder / "contrasts.csv"), lambda r: (r["panel"], r["reference"])
    )
    assert set(report) == set(contrasts)
    for key, numbers in contrasts.items():
        near(
            numbers,
            [
                float(report[key][k])
                for k in ("delta", "low", "high", "family_wins", "families")
            ],
        )
    bank_means = lookup_unique(records(folder / "means.csv"), lambda r: r["panel"])
    for (panel, method), expected in means.items():
        near(expected, float(bank_means[panel][method]))
    families = lookup_unique(records(folder / "families.csv"), lambda r: r["scenario"])
    assert set(families) == set(SLOTS)
    for name in families:
        for method in METHODS:
            by_rep = [
                view_loss(
                    loss,
                    name,
                    rep,
                    BASE_SEED[round_no] + 1000 * SLOTS[name] + rep,
                    "joint",
                    method,
                )
                for rep in range(repeats(round_no))
            ]
            near(fmean(by_rep), float(families[name][method]))
    summarized = defaultdict(lambda: defaultdict(list))
    for row in diagnostics.values():
        if row["data_seed"] != row["gp_seed"]:
            continue
        selected = row["chosen"]
        values = summarized[row["method"], row["mode"]]
        before = float(row["discovery_" + selected])
        after = float(row["refit_" + selected])
        values["chosen_discovery"].append(before)
        values["chosen_refit"].append(after)
        values["refit_change"].append(after - before)
        for stage in ("discovery", "refit"):
            regret = float(row["chosen_" + stage + "_regret"])
            values["chosen_" + stage + "_regret"].append(regret)
            values["chosen_ties_" + stage + "_oracle"].append(regret <= 1e-12)
        for choice in CANDIDATES:
            values["refit_" + choice].append(float(row["refit_" + choice]))
    table = lookup_unique(
        records(folder / "candidate-summary.csv"), lambda r: (r["method"], r["mode"])
    )
    assert set(table) == set(summarized)
    for key, values in summarized.items():
        for col, data in values.items():
            near(fmean(data), float(table[key][col]))
    if round_no == 3:
        ss = lookup_unique(
            records(folder / "search-seed-sensitivity.csv"),
            lambda r: (r["scenario"], int(r["replicate"])),
        )
        assert set(ss) == {(name, rep) for name in SLOTS for rep in (0, 1)}
        for (name, rep), row in ss.items():
            seed = BASE_SEED[3] + 1000 * SLOTS[name] + rep
            for arm in ARMS:
                near(
                    abs(
                        view_loss(loss, name, rep, seed + 10_000_000, "joint", arm)
                        - view_loss(loss, name, rep, seed, "joint", arm)
                    ),
                    float(row[arm]),
                )


def verify_round(folder, round_no):
    settings = json.loads((folder / "settings.json").read_text())
    assert settings["round"] == round_no and settings["replicates"] == repeats(round_no)
    assert (
        settings["seed_root"] == BASE_SEED[round_no] and settings["scenarios"] == SLOTS
    )
    assert settings["source_label_noise"] == 0 and settings["split_seed"] == "data_seed"
    assert settings["development_rows"] == 640 and settings["test_rows"] == 2048
    assert settings["primary_weights"] == {
        "clean": 0.5,
        "provider_swap": 0.25,
        "log_jitter": 0.25,
    }
    assert settings["coherent_pair"] == (round_no == 3)
    assert settings["second_gp_seed_replicates"] == ([0, 1] if round_no == 3 else [])
    assert settings["second_gp_seed_offset"] == (10_000_000 if round_no == 3 else None)
    if round_no == 1:
        assert settings["native_kwargs"] == {} and settings["challenger_kwargs"] == {
            "pair_basis": True
        }
    if round_no in (2, 3):
        assert settings["native_kwargs"] == {} and settings["challenger_kwargs"] == {
            "l2_strength": 0.25
        }
    loss, diag = verify_rows(
        round_no,
        records(folder / "results.csv.gz"),
        records(folder / "diagnostics.csv.gz"),
    )
    costs = verify_costs(round_no, settings, records(folder / "costs.csv"), diag)
    effects, means = bank_effects(round_no, loss)
    sample_fits, sample_checks, ref_checks = sample_audit(
        round_no, records(folder / "sample_trees.jsonl.gz"), loss, diag, costs, settings
    )
    check_summaries(round_no, folder, effects, means, diag, loss)
    print(
        f"R{round_no}: {len(loss)} scores, {len(diag)} candidate diagnostics, {len(costs)} cost rows; replayed {sample_fits} saved fits, {sample_checks} candidate test scores, {ref_checks} control test scores."
    )
    for panel in ("joint", "clean", "stress", "canonical_shift", "coherent_pair"):
        if (panel, "native") in effects:
            delta, lo, hi, wins, n = effects[panel, "native"]
            print(
                f"{panel}: challenger minus native {delta:+.8f} [{lo:+.8f}, {hi:+.8f}]; wins {wins}/{n} families"
            )
    for method in ARMS:
        clean = [
            row
            for row in diag.values()
            if row["mode"] == "clean"
            and row["gp_seed"] == row["data_seed"]
            and row["method"] == method
        ]
        gap = fmean(float(row["chosen_discovery_regret"]) for row in clean)
        change = fmean(
            float(row["refit_" + row["chosen"]])
            - float(row["discovery_" + row["chosen"]])
            for row in clean
        )
        print(
            f"{method} clean: validation choice regret before refit {gap:.8f}; refit loss change {change:+.8f}"
        )


def verify_digits(folder):
    """Reconstruct the test-row identities and read every archived per-case loss."""
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    manifest = json.loads((folder / "manifest.json").read_text())
    assert manifest["challenger_kwargs"] == {"l2_strength": 0.25}
    seeds = list(range(2_810_000, 2_810_008))
    assert manifest["seeds"] == seeds
    controls = ("previous", "mean", "top_n", "stack_p", "stack_logit")
    methods = ("challenger", *controls)

    def key(r):
        return int(r["seed"]), r["method"]

    scores = lookup_unique(records(folder / "scores.csv"), key)
    assert set(scores) == {(s, m) for s in seeds for m in methods}
    cases = defaultdict(list)
    for row in records(folder / "cases.csv.gz"):
        cases[key(row)].append(row)
    assert set(cases) == set(scores)
    provenance = lookup_unique(manifest["split_banks"], lambda r: r["seed"])
    assert set(provenance) == set(seeds)
    labels = load_digits().target
    test_sets = []
    for seed, bank in provenance.items():
        base, other = train_test_split(
            np.arange(len(labels)), train_size=0.5, random_state=seed, stratify=labels
        )
        fit, test = train_test_split(
            other, test_size=0.5, random_state=seed + 1, stratify=labels[other]
        )
        assert (len(base), len(fit), len(test)) == (898, 449, 450)
        np.testing.assert_array_equal(
            np.sort(np.r_[base, fit, test]), np.arange(len(labels))
        )
        for part, idx in [("base", base), ("fit", fit), ("test", test)]:
            assert bank[part + "_idx_sha256"] == reference["digest"](idx, "<i8")
        test_sets.append(set(test))
        for method in methods:
            group = cases[seed, method]
            np.testing.assert_array_equal([int(r["row"]) for r in group], test)
            np.testing.assert_array_equal(
                [int(r["label"]) for r in group], labels[test]
            )
            near(
                fmean(float(r["squared_loss"]) for r in group),
                float(scores[seed, method]["test_brier"]),
            )
            near(
                fmean(r["prediction"] == r["label"] for r in group),
                float(scores[seed, method]["accuracy"]),
            )
    assert len(set.union(*test_sets)) == manifest["distinct_test_images"]
    near(
        fmean(len(a & b) for a, b in combinations(test_sets, 2)),
        manifest["mean_pairwise_test_overlap"],
    )
    contrasts = lookup_unique(
        records(folder / "contrasts.csv"), lambda r: r["comparator"]
    )
    assert set(contrasts) == set(controls)
    for method, row in contrasts.items():
        changes = [
            float(scores[s, method]["test_brier"])
            - float(scores[s, "challenger"]["test_brier"])
            for s in seeds
        ]
        near(fmean(changes), float(row["challenger_brier_gain"]))
        assert int(row["wins"]) == sum(x > 1e-12 for x in changes)
        assert int(row["ties"]) == sum(abs(x) <= 1e-12 for x in changes)
        assert int(row["losses"]) == sum(x < -1e-12 for x in changes)
    print(
        f"Digits: {len(scores)} fitted-method scores, {sum(map(len, cases.values()))} per-case records, {len(set.union(*test_sets))} distinct test images."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--round", dest="round_no", required=True, type=int, choices=range(1, 4)
    )
    parser.add_argument(
        "--out", type=Path, help="Directory with raw and summarized artifacts"
    )
    parser.add_argument(
        "--digits",
        action="store_true",
        help="Also audit the separate digits case records in the sibling digits directory.",
    )
    args = parser.parse_args()
    if args.digits and args.round_no != 3:
        parser.error("--digits belongs to round 3")
    with threadpool_limits(1):
        folder = args.out or HERE / f"round-{args.round_no}"
        verify_round(folder, args.round_no)
        if args.digits:
            verify_digits(folder.parent / "digits")

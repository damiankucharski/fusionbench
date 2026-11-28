"""Independently replay controls, audit paired contrasts and verify the registered choice."""

from __future__ import annotations

import argparse
import json
import subprocess
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from fusionbench.transfer import load_preset

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
KEYS = ["cohort", "family", "seed"]
TOL = 2e-12


def empirical(p, labels):
    targets = np.eye(p.shape[1])[labels]
    return np.square(p - targets).sum(1).mean()


def population(bank, p):
    if bank.audit_y is not None:
        return empirical(p, bank.audit_y)
    # Directly integrate both label outcomes, independent of any Bayes-risk decomposition.
    return 2 * bank.mass @ (bank.q * (1 - p[:, 1]) ** 2 + (1 - bank.q) * p[:, 1] ** 2)


def norm(p):
    p = np.maximum(np.asarray(p, dtype=float), 1e-12)
    return p / p.sum(1, keepdims=True)


def digest(a):
    return sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def controls(x, y, queries):
    """Fit the three reference rules directly, including ties in ranked prefixes."""
    m = x.shape[1]
    losses = [empirical(norm(x[:, j]), y) for j in range(m)]
    order = np.argsort(losses, kind="stable")
    prefix = np.zeros((len(y), x.shape[2]))
    best_loss = np.inf
    for k, j in enumerate(order, 1):
        prefix += x[:, j]
        loss = empirical(norm(prefix / k), y)
        if loss < best_loss:
            best_loss, selected = loss, k

    scaler = StandardScaler().fit(x.reshape(len(x), -1))
    lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000)
    lr.fit(scaler.transform(x.reshape(len(x), -1)), y)
    outputs = []
    for t in queries:
        prefix = np.zeros((len(t), t.shape[2]))
        for j in order[:selected]:
            prefix += t[:, j]
        padded = np.zeros((len(t), t.shape[-1]))
        padded[:, lr.classes_] = lr.predict_proba(
            scaler.transform(t.reshape(len(t), -1))
        )
        outputs.append(
            {
                "mean": norm(t.mean(1)),
                "top_n": norm(prefix / selected),
                "lr": norm(padded),
            }
        )
    return outputs


def read_table(r, name):
    return pd.read_csv(OUT / f"round{r}-{name}.csv", float_precision="round_trip")


def bank_integrity(b, block):
    assert not np.shares_memory(b.x, b.audit_x)
    n, m, c = b.x.shape
    assert len(b.y) == n and b.audit_x.shape[1:] == (m, c)
    for name in ("n_fit", "n_audit", "n_classes", "n_providers"):
        assert (block[name] == b.context[name]).all()
    for key, val in b.context.items():
        if key.endswith("_sha256") and key in block:
            assert (block[key] == val).all()
    if b.audit_y is None:
        assert np.min(b.mass) >= 0 and abs(1 - b.mass.sum()) < TOL
        assert np.all((b.q >= 0) & (b.q <= 1))
        assert len(b.mass) == len(b.q) == len(b.audit_x) == 512
    if b.context["cohort"] == "digits":
        fit, audit, base = map(set, (b.fit_ids, b.audit_ids, b.base_ids))
        assert not fit & audit and not base & audit and not fit & base
        assert (len(fit), len(audit), len(base)) == (n, 450, 898)


def summary_differences(r, scores):
    """Recompute paired bounds by bootstrap weights on seed cells within each fixed family."""
    contrasts = read_table(r, "contrasts")
    ids = ["metric", "group", "method", "reference"]
    saved = contrasts.set_index(ids, verify_integrity=True)
    levels = read_table(r, "levels").set_index(
        ["group", "method"], verify_integrity=True
    )
    families = read_table(r, "families").set_index(
        ["family", "method"], verify_integrity=True
    )
    n_checked, largest_error = 0, 0.0
    scores = scores.assign(gap=scores.brier - scores.fit_brier)
    macro_groups = scores.family.where(scores.cohort != "standard", "standard")
    for group, part in scores.groupby(macro_groups, sort=True):
        risk = part.pivot(
            index=["family", "seed"], columns="method", values="brier"
        ).sort_index()
        nf = risk.index.get_level_values("family").nunique()
        ns = risk.index.get_level_values("seed").nunique()
        assert nf * ns == len(risk) and np.isfinite(risk.values).all()
        if not group.startswith("digits"):
            draws = np.random.default_rng(91700 + r).integers(ns, size=(10000, nf, ns))
            # Each record is an independent seed draw of a bank, not one row/pixel.
            weights = np.stack([(draws == i).sum(2) for i in range(ns)], axis=-1)
        for metric in ("brier", "fit_brier", "gap", "macro_f1", "average_precision"):
            data = part.pivot(
                index=["family", "seed"], columns="method", values=metric
            ).reindex_like(risk)
            available = not data.isna().any().any()
            for method in data:
                level = levels.loc[(group, method)]
                assert (level.families, level.seeds_per_family, level.banks) == (
                    nf,
                    ns,
                    nf * ns,
                )
                if group == "standard":
                    for family, vals in data[method].groupby(level="family"):
                        reported = families.loc[(family, method)]
                        assert reported.banks == ns
                        np.testing.assert_allclose(
                            vals.mean(),
                            reported[metric],
                            rtol=0,
                            atol=TOL,
                            equal_nan=True,
                        )
                        if metric == "brier":
                            diff = vals.mean() - data["full"].loc[family].mean()
                            assert abs(reported.delta_full - diff) < TOL
                if not available:
                    assert pd.isna(level[metric])
                    continue
                assert abs(level[metric] - data[method].mean()) < TOL
                for ref in data:
                    if method == ref:
                        continue
                    delta = (data[method] - data[ref]).to_numpy().reshape(nf, ns)
                    row = saved.loc[metric, group, method, ref]
                    benefit = (
                        delta if metric in {"macro_f1", "average_precision"} else -delta
                    )
                    assert (
                        delta.size,
                        (benefit > 1e-12).sum(),
                        (benefit < -1e-12).sum(),
                    ) == (row.total, row.better_banks, row.worse_banks)
                    error = abs(delta.mean() - row.delta)
                    if group.startswith("digits"):
                        assert np.isnan(row.lo) and np.isnan(row.hi)
                    else:
                        boot = np.einsum("bfs,fs->b", weights, delta) / (nf * ns)
                        lo, hi = np.quantile(boot, [0.025, 0.975])
                        error = max(error, abs(lo - row.lo), abs(hi - row.hi))
                    largest_error = max(largest_error, error)
                    n_checked += 1
    assert n_checked == len(contrasts)
    assert largest_error < TOL
    return n_checked, largest_error


def classification_scores(b, p):
    """Integrate exact class masses and use each tied score threshold once for AP."""
    c = p.shape[1]
    predicted = np.argmax(p, axis=1)
    if b.q is None:
        masses = np.eye(c)[b.audit_y] / len(b.audit_y)
    else:
        masses = b.mass[:, None] * np.column_stack((1 - b.q, b.q))
    conf = np.column_stack([masses[predicted == j].sum(0) for j in range(c)])
    denom = conf.sum(0) + conf.sum(1)
    macro = np.divide(2 * np.diag(conf), denom, out=np.zeros(c), where=denom > 0).mean()
    if c != 2:
        return macro, np.nan
    sort = np.argsort(-p[:, 1], kind="stable")
    ordered = masses[sort]
    change = np.r_[np.diff(p[sort, 1]) != 0, True]
    tp = np.cumsum(ordered[:, 1])[change]
    mass = np.cumsum(ordered.sum(1))[change]
    pos = np.diff(np.r_[0, tp])
    ap = np.sum(pos * tp / mass) / tp[-1] if tp[-1] else 0.0
    return macro, ap


def ordinary_refit(b):
    from fusionbench.methods.okapi import OKAPIMethod

    method = OKAPIMethod(
        fitness="brier",
        seed=0,
        n_restarts=1,
        n_seeds=0,
        generations=8,
        population_size=8,
        population_multiplier=2,
        tournament_size=3,
        max_nodes=31,
        max_depth=7,
    )
    return method.fit(
        b.x,
        b.y,
        task="binary" if b.x.shape[-1] == 2 else "multiclass",
        n_classes=b.x.shape[-1],
    )


def instrumented_refit(b, name, row):
    """Witness the full label vector at every native population fitness evaluation."""
    from unittest.mock import patch

    from interventions import fit_candidate
    from okapi.okapi import Okapi

    original = Okapi._calculate_fitnesses
    calls = []

    def checked(instance, *args, **kwargs):
        np.testing.assert_array_equal(np.asarray(instance.gt_tensor).reshape(-1), b.y)
        calls.append(len(b.y))
        return original(instance, *args, **kwargs)

    with patch.object(Okapi, "_calculate_fitnesses", checked):
        model = (
            ordinary_refit(b)
            if name == "full"
            else fit_candidate(name, b.x, b.y, groups=b.groups)
        )
    assert calls and len(set(calls)) == 1
    p = model.predict(b.audit_x)
    assert digest(p) == row.test_prediction_hash
    assert digest(model.predict(b.x)) == row.fit_prediction_hash
    assert abs(population(b, p) - row.brier) < TOL
    return {
        "method": name,
        "family": b.context["family"],
        "seed": b.context["seed"],
        "rows_every_evaluation": calls[0],
        "fitness_batches": len(calls),
        "test_prediction_hash": digest(p),
    }


def expected_methods(r, chosen):
    base = {"full", "lr", "mean", "top_n", "previous8"}
    extra = {
        1: {"reduced", "split"},
        2: {"sham", "seeded_mean"},
        3: {"seeded_mean", "pull_mean", "pull_zero"},
        4: {"seeded_mean", "paired_se", "paired_zero"},
        5: {chosen},
    }
    return base | extra[r]


def check_round(r, chosen=None):
    from banks_full import banks

    scores = read_table(r, "scores")
    assert not scores.duplicated(KEYS + ["method"]).any()
    assert set(scores["round"]) == {r}
    blocks = scores.groupby(KEYS)
    assert len(blocks) == (284 if r == 5 else 118)
    s = scores.query("cohort == 'standard'")
    assert set(s.family) == set(load_preset("standard-v5")["scenarios"])
    assert scores[["brier", "fit_brier"]].min().min() >= 0
    assert scores[["brier", "fit_brier"]].max().max() <= 2 + TOL
    details = [
        json.loads(line)
        for line in (OUT / f"round{r}-fits.jsonl").read_text().splitlines()
    ]
    traces = {tuple(d[k] for k in KEYS): d["fits"] for d in details}
    assert len(traces) == len(details) == len(blocks)
    base = 380000 + r * 1000
    seen, fit_inputs, replays, err = set(), set(), [], []
    probes = []
    scene_audit = {}
    nested_digits = {}
    for b in banks(r):
        cohort, family, seed = key = tuple(b.context[k] for k in KEYS)
        assert key not in seen
        seen.add(key)
        fit_inputs.add(digest(b.x))
        block = blocks.get_group(key).set_index("method", verify_integrity=True)
        assert set(block.index) == expected_methods(r, chosen)
        bank_integrity(b, block)
        fitted_rows = block.n_evolution_rows.dropna()
        assert set(fitted_rows.index) == set(block.index) - {
            "mean",
            "top_n",
            "lr",
            "previous8",
        }
        group_count = len(np.unique(b.groups)) if b.groups is not None else len(b.y)
        disc = 7 * group_count // 10 * (len(b.y) // group_count)
        for method, count in fitted_rows.items():
            correct = disc if method in {"reduced", "split"} else len(b.y)
            assert count == correct == traces[key][method]["n_evolution_rows"]
        if r == 1:
            hold = traces[key]["split"]["holdout"]
            assert hold["n_selection_rows"] == len(b.y) - disc
            assert hold["winner_heldout_brier"] <= hold["champion_heldout_brier"] + TOL
            if hold["winner_index"] == 0:
                assert (
                    block.loc["split", "test_prediction_hash"]
                    == block.loc["reduced", "test_prediction_hash"]
                )
        start, reps = {
            "controlled": (0, 32 if r == 5 else 12),
            "clustered": (300, 32 if r == 5 else 12),
            "standard": (100, 4 if r == 5 else 2),
            "digits": (200, 8 if r == 5 else 4),
        }[cohort]
        assert base + start <= seed < base + start + reps
        if cohort == "controlled":
            np.testing.assert_array_equal(np.bincount(b.y), [64, 64])
            assert b.audit_y is None
        if cohort == "clustered":
            train_ids, sizes = np.unique(b.groups, return_counts=True)
            audit_ids, asizes = np.unique(b.audit_groups, return_counts=True)
            assert (len(train_ids), len(audit_ids)) == (int(family[5:]), 256)
            assert not np.intersect1d(train_ids, audit_ids).size
            assert len(b.y) == 128 and np.all(sizes == 128 // len(train_ids))
            assert np.all(asizes == 8)
            assert scene_audit.setdefault(seed, digest(b.audit_x)) == digest(b.audit_x)
            assert set(np.unique(b.y)) == {0, 1}
        if cohort == "digits":
            if family == "digits_449":
                nested_digits[seed] = (
                    set(b.fit_ids),
                    digest(b.audit_x),
                    digest(b.audit_y),
                )
            else:
                ids, xh, yh = nested_digits[seed]
                assert len(b.fit_ids) == 100 and set(b.fit_ids) < ids
                assert (xh, yh) == (digest(b.audit_x), digest(b.audit_y))
        fit, test = controls(b.x, b.y, (b.x, b.audit_x))
        for m in fit:
            row = block.loc[m]
            assert digest(test[m]) == row.test_prediction_hash, (key, m, "test")
            assert digest(fit[m]) == row.fit_prediction_hash, (key, m, "fit")
            a, f = population(b, test[m]), empirical(fit[m], b.y)
            f1, ap = classification_scores(b, test[m])
            err.extend(
                [abs(a - row.brier), abs(f - row.fit_brier), abs(f1 - row.macro_f1)]
            )
            if np.isfinite(ap):
                err.append(abs(ap - row.average_precision))
            else:
                assert np.isnan(row.average_precision)
        if (family, seed - start) in {
            (k, base) for k in ("weak", "scene16", "multiclass_committee", "digits_449")
        }:
            replays.append(instrumented_refit(b, "full", block.loc["full"]))
            if chosen:
                replays.append(instrumented_refit(b, chosen, block.loc[chosen]))
        if family == "scene16" and seed == base + 300:
            probes.append(audit_outcome_probe(r, b, block, chosen))
    assert len(probes) == 1 and seen == set(blocks.groups)
    assert len(replays) == (8 if chosen else 4)
    assert max(err) < TOL, max(err)
    contrasts, worst = summary_differences(r, scores)
    return {
        "banks": len(seen),
        "score_rows": len(scores),
        "paired_contrasts_checked": int(contrasts),
        "reference_fits_reconstructed": 3 * len(seen),
        "fit_and_audit_hashes_matched": 6 * len(seen),
        "max_reference_metric_error": max(err),
        "max_paired_summary_error": worst,
        "native_replays": replays,
        "audit_outcome_probes": probes,
    }, fit_inputs


def audit_outcome_probe(r, bank, saved, chosen):
    from unittest.mock import patch

    import run_full

    original = run_full.fit_methods
    trained = False

    def fit(*args, **kwargs):
        nonlocal trained
        assert args[1] is bank.x and args[2] is bank.y
        result = original(*args, **kwargs)
        trained = True
        return result

    class Guard:
        x, y, groups = bank.x, bank.y, bank.groups

        def __getattr__(self, key):
            assert trained, f"Access before fitting: {key}"
            if key == "audit_y":
                return 1 - bank.audit_y
            return getattr(bank, key)

    with patch.object(run_full, "fit_methods", fit):
        rows, _ = run_full.measure(r, Guard(), chosen)
    matched = 0
    for row in rows:
        method = row["method"]
        assert row["test_prediction_hash"] == saved.at[method, "test_prediction_hash"]
        matched += 1
    assert any(
        abs(row["brier"] - saved.at[row["method"], "brier"]) > 1e-4 for row in rows
    )
    return {
        "family": bank.context["family"],
        "seed": bank.context["seed"],
        "forecasts_unchanged_after_reversing_audit_labels": matched,
        "premature_audit_reads": 0,
    }


def choose():
    options = []
    caps = {"strong": 0.010, "standard": 0.005, "digits_449": 0.004}
    for r, m in [(2, "seeded_mean"), (3, "pull_mean"), (4, "paired_se")]:
        s = read_table(r, "scores")
        s["group"] = s.family.where(s.cohort != "standard", "standard")
        avg = (
            s.groupby(["group", "family", "method"])
            .brier.mean()
            .groupby(["group", "method"])
            .mean()
            .unstack()
        )
        delta = avg[m] - avg["full"]
        violation = max([0.0] + [(delta[g] - c) / c for g, c in caps.items()])
        options.append((violation, (delta["weak"] + delta["scene16"]) / 2, m))
    return min(options, key=lambda t: t[:2])[2]


def git_lock():
    root = HERE.parents[2]
    path = (HERE / "selection.json").relative_to(root)
    commit = subprocess.check_output(
        ["git", "log", "-1", "--diff-filter=A", "--format=%H", "--", str(path)],
        cwd=root,
        text=True,
    ).strip()
    assert commit, "Screen choice must be committed"
    raw = subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=root)
    assert raw == (root / path).read_bytes()
    final = (OUT / "round5-scores.csv").relative_to(root)
    assert not subprocess.check_output(
        ["git", "ls-tree", commit, "--", str(final)], cwd=root
    )
    selection = json.loads(raw)
    expected_hashes = {
        f"round{r}-{kind}.csv" for r in range(1, 5) for kind in ("scores", "contrasts")
    }
    assert set(selection["sha256"]) == expected_hashes
    for name, stored_hash in selection["sha256"].items():
        fp = OUT / name
        contents = fp.read_bytes()
        assert sha256(contents).hexdigest() == stored_hash
        assert (
            subprocess.check_output(
                ["git", "show", f"{commit}:{fp.relative_to(root)}"], cwd=root
            )
            == contents
        )
    return commit, selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--round", type=int, choices=range(1, 6), help="Audit a single finished round"
    )
    opts = parser.parse_args()
    chosen = None
    confirm = opts.round in (None, 5)
    if confirm:
        lock, selection = git_lock()
        chosen = choose()
        assert chosen == selection["chosen"]
    report = {}
    start = perf_counter()
    used = set()
    with threadpool_limits(limits=1):
        for r in [opts.round] if opts.round else range(1, 6):
            results, inputs = check_round(r, chosen if r == 5 else None)
            assert not used.intersection(inputs)
            used.update(inputs)
            report[f"round{r}"] = results
    if confirm:
        s = read_table(5, "contrasts")
        test = s[
            (s.method == chosen) & (s.reference == "full") & (s.metric == "brier")
        ].set_index("group")
        success = test.loc["weak", "hi"] < 0 and test.loc["scene16", "hi"] < 0
        broad = (
            success
            and test.loc["standard", "hi"] <= 0.005
            and test.loc["strong", "hi"] <= 0.010
            and test.loc["digits_449", "delta"] <= 0.004
        )
        claims = json.loads((OUT / "confirmation.json").read_text())
        expected_checks = {k: bool(test.loc[k, "hi"] < 0) for k in ("weak", "scene16")}
        expected_checks.update(
            {
                "standard": bool(test.loc["standard", "hi"] <= 0.005),
                "strong": bool(test.loc["strong", "hi"] <= 0.010),
                "digits_449": bool(test.loc["digits_449", "delta"] <= 0.004),
            }
        )
        assert claims["chosen"] == chosen and claims["criteria"] == expected_checks
        assert claims["both_targets"] == bool(success) and claims[
            "all_criteria"
        ] == bool(broad)
        report.update(
            selection_commit=lock,
            selected=chosen,
            joint_target_pass=bool(success),
            broad_pass=bool(broad),
        )
    report["audit_seconds"] = perf_counter() - start
    if opts.round:
        print(json.dumps(report, indent=2))
    else:
        (OUT / "verification-full.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )


if __name__ == "__main__":
    main()

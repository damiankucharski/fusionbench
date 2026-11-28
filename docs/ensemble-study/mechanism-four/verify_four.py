"""Reconstruct the four synthetic trials independently of their fit/score/summary functions."""

from __future__ import annotations

import argparse
import json
import subprocess
from hashlib import sha256
from itertools import product
from pathlib import Path
from runpy import run_path
from time import perf_counter
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from fusionbench.transfer import load_preset

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
OLD = run_path(str(HERE.parent / "full-data-five" / "verify_full.py"))
KEY = ["cohort", "family", "seed"]
LIMITS = {"standard": 0.005, "scene16": 0.005, "strong": 0.010}
TOL = 4e-12


def table(r, name):
    return pd.read_csv(OUT / f"r{r}-{name}.csv", float_precision="round_trip")


def hash_array(a):
    return sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def grouped_parts(raw):
    for name, part in raw[raw.cohort != "standard"].groupby("family"):
        yield name, part, "single"
    standard = raw[raw.cohort == "standard"]
    yield "standard", standard, "standard"
    legacy = set(load_preset("standard-v5")["scenarios"])
    yield "legacy27", standard[standard.family.isin(legacy)], "standard"
    yield "factorial", raw[raw.cohort == "factorial"], "factorial"


def cube(group, grouping, metric):
    """Build a family by replication by method cube; collapse matched interventions per seed."""
    cells = group.pivot(
        index=["family", "seed"], columns="method", values=metric
    ).sort_index()
    m = list(cells)
    ns = group.seed.nunique()
    array = cells.to_numpy().reshape(-1, ns, len(m))
    if grouping == "factorial":
        assert array.shape[0] == 8
        array = array.mean(0, keepdims=True)
    return m, array


def summaries(r, raw):
    saved = table(r, "contrasts")
    levels = table(r, "levels").set_index(["group", "method"], verify_integrity=True)
    families = table(r, "families").set_index(
        ["family", "method"], verify_integrity=True
    )
    delta_records = saved[saved.metric.isin(["brier", "expected_brier"])]
    assert not delta_records.duplicated(
        ["group", "metric", "method", "reference"]
    ).any()
    largest, checked, family_checks = 0.0, 0, 0
    for name, group, grouping in grouped_parts(raw):
        for metric in ["brier", "expected_brier"]:
            methods, arr = cube(group, grouping, metric)
            nf, ns, _ = arr.shape
            if not np.isfinite(arr).all():
                assert metric == "expected_brier" and grouping == "standard"
                assert delta_records.query("group==@name and metric==@metric").empty
                continue
            # Verify the quantiles independently, weighting a cell by how often it was sampled.
            draws = np.random.default_rng(84100 + r).integers(ns, size=(10000, nf, ns))
            counts = np.stack([(draws == j).sum(2) for j in range(ns)], axis=-1)
            means = arr.mean((0, 1))
            for j, method in enumerate(methods):
                err = abs(levels.loc[(name, method), metric] - means[j])
                assert err < TOL
                largest = max(largest, err)
            records = delta_records.query("group == @name and metric == @metric")
            assert not records.empty
            for row in records.itertuples(index=False):
                j, k = methods.index(row.method), methods.index(row.reference)
                delta = arr[:, :, j] - arr[:, :, k]
                boot = np.einsum("bfs,fs->b", counts, delta) / (nf * ns)
                endpoints = np.quantile(boot, [0.025, 0.975])
                err = max(
                    abs(row.delta - delta.mean()),
                    np.abs(endpoints - [row.lo, row.hi]).max(),
                )
                largest = max(largest, err)
                assert err < TOL, (r, name, metric, row.method, row.reference, err)
                checked += 1
        if grouping == "standard":
            f = group.groupby(["family", "method"])[
                ["brier", "fit_brier", "macro_f1", "ap"]
            ].mean()
            for metric in f.columns.intersection(families.columns):
                np.testing.assert_allclose(
                    f[metric],
                    families.loc[f.index, metric],
                    rtol=0,
                    atol=TOL,
                    equal_nan=True,
                )
                family_checks += len(f)
    assert checked == len(delta_records)
    return {
        "brier_intervals": checked,
        "family_scores": family_checks,
        "max_error": float(largest),
    }


def expected_methods(r, chosen):
    shared = {"full", "pull_mean", "mean", "top_n", "lr", "previous8"}
    arms = {
        1: {"sparse_pool"},
        2: {"sparse_pool", "random_pool", "sparse_gp", "random_gp"},
        3: {"sparse_gp", "cal_pull", "cal_sparse", "raw_pull", "cal_mean"},
        4: {chosen, "sparse_gp"} if chosen == "cal_sparse" else {chosen},
    }
    return shared | arms[r]


def checked_measures(b, p):
    expected = np.nan
    if b.q is not None:
        assert p.shape[1] == 2 and len(b.q) == len(p)
        w = np.ones(len(p)) / len(p) if b.mass is None else b.mass
        bayes = np.column_stack([1 - b.q, b.q])
        expected = w @ (2 * b.q * (1 - b.q) + np.square(p - bayes).sum(1))
    brier = expected if b.audit_y is None else OLD["empirical"](p, b.audit_y)
    target = SimpleNamespace(
        audit_y=b.audit_y, q=b.q if b.audit_y is None else None, mass=b.mass
    )
    f1, ap = OLD["classification_scores"](target, p)
    return [brier, expected, f1, ap]


def check_bank(b, rows, paired):
    cohort, family, seed = (b.context[k] for k in KEY)
    assert not np.shares_memory(b.x, b.audit_x)
    assert len(b.x) == len(b.y) and b.x.shape[1:] == b.audit_x.shape[1:]
    for field in ("fit_x", "fit_y", "audit_x", "audit_y", "q"):
        arr = getattr(b, {"fit_x": "x", "fit_y": "y"}.get(field, field))
        if arr is not None:
            key = field + "_sha256"
            if key in rows:
                assert (rows[key] == hash_array(arr)).all(), key
    if b.groups is not None:
        assert len(b.groups) == len(b.x) and len(b.audit_groups) == len(b.audit_x)
        assert not set(b.groups) & set(b.audit_groups)
    if cohort == "factorial":
        assert (len(np.unique(b.groups)), len(np.unique(b.audit_groups))) == (16, 256)
        sig = tuple(
            hash_array(a)
            for a in (b.y, b.audit_y, b.q, b.groups, b.audit_groups, b.audit_regime)
        )
        assert sig == paired.setdefault(("factorial", seed), sig)
    if cohort == "clustered":
        sig = tuple(hash_array(a) for a in (b.audit_x, b.audit_y, b.q, b.audit_groups))
        assert sig == paired.setdefault(("clustered", seed), sig)
    if cohort == "controlled":
        assert (
            family in ("strong", "weak")
            and b.mass.shape == (512,)
            and b.audit_y is None
        )


def layout(r, scores):
    n = 24 if r == 4 else 8
    rep = 4 if r == 4 else 2
    base = 491000 + 1000 * r
    cells = {f"h{h}c{c}s{s}" for h, c, s in product(range(2), repeat=3)}
    groups = {
        "standard": (set(load_preset("standard-v6")["scenarios"]), 400, rep),
        "factorial": (cells, 0, n),
        "clustered": ({"scene16", "scene128"}, 100, n),
        "controlled": ({"weak", "strong"}, 200, n),
    }
    assert set(scores.cohort) == set(groups) and len(groups["standard"][0]) == 30
    expected = set()
    for cohort, (families, shift, count) in groups.items():
        for f in families:
            expected.update(
                (cohort, f, seed) for seed in range(base + shift, base + shift + count)
            )
    assert len(expected) == (408 if r == 4 else 156)
    assert expected == set(scores.groupby(KEY).groups)
    return expected


def audit_probe(r, b, saved, chosen):
    """Rerun normal fit/predict after reversing held-out labels and blocking early access."""
    import run_four

    fitted = False
    fit = run_four.fit_models

    def wrapped(round_no, x, y, groups, chosen=None):
        nonlocal fitted
        assert x is b.x and y is b.y and groups is b.groups
        result = fit(round_no, x, y, groups, chosen)
        fitted = True
        return result

    class Guard:
        x, y, groups = b.x, b.y, b.groups

        def __getattr__(self, attr):
            if attr in {
                "audit_x",
                "audit_y",
                "audit_groups",
                "audit_regime",
                "q",
                "mass",
            }:
                assert fitted, f"{attr} accessed before fit returned"
            if attr == "audit_y":
                return 1 - b.audit_y
            return getattr(b, attr)

    with patch.object(run_four, "fit_models", wrapped):
        item = run_four.measure(r, Guard(), chosen)
    results = item[0] if isinstance(item, tuple) else item
    assert len(results) == len(saved)
    for row in results:
        method = row["method"]
        assert row["test_hash"] == saved.at[method, "test_hash"]
        assert row["fit_hash"] == saved.at[method, "fit_hash"]
    assert any(
        abs(row["brier"] - saved.at[row["method"], "brier"]) > 0.01 for row in results
    )
    return {
        "family": b.context["family"],
        "seed": b.context["seed"],
        "unchanged_forecasts": len(results),
    }


def temperature_replay(b, chosen, saved):
    """Independently choose the scalar exponent on all source labels and check its held-out effect."""
    import run_four
    from okapi.okapi import Okapi
    from sparse_gp import fit_sparse_gp

    raw_name = "pull_mean" if chosen == "cal_pull" else "sparse_gp"
    original = Okapi._calculate_fitnesses
    batches = []

    def check_labels(model, *args, **kwargs):
        np.testing.assert_array_equal(np.asarray(model.gt_tensor).ravel(), b.y)
        batches.append(len(b.y))
        return original(model, *args, **kwargs)

    with patch.object(Okapi, "_calculate_fitnesses", check_labels):
        model = (
            run_four.fit_candidate(raw_name, b.x, b.y, groups=b.groups)
            if raw_name == "pull_mean"
            else fit_sparse_gp(raw_name, b.x, b.y)
        )
    source, test = model.predict(b.x), model.predict(b.audit_x)
    assert hash_array(source) == saved.at[raw_name, "fit_hash"]
    assert hash_array(test) == saved.at[raw_name, "test_hash"]
    n_eff = len(b.y) if b.groups is None else len(np.unique(b.groups))

    def power(x, j):
        if not j:
            return x
        raised = np.clip(x, 1e-6, 1 - 1e-6) ** (2.0 ** (j / 4))
        return raised / raised.sum(1, keepdims=True)

    def criterion(j):
        penalty = 0.25 * (j / 4) ** 2 / np.sqrt(n_eff)
        return OLD["empirical"](power(source, j), b.y) + penalty, abs(j), j

    j = min(range(-4, 5), key=criterion)
    beta = 2.0 ** (j / 4)
    assert beta == saved.at[chosen, "beta"] and batches and set(batches) == {len(b.y)}
    if not j:
        for label in ("test_hash", "fit_hash"):
            assert saved.at[chosen, label] == saved.at[raw_name, label]
    predicted = power(test, j)
    values = checked_measures(b, predicted)
    np.testing.assert_allclose(
        values,
        saved.loc[chosen, ["brier", "expected_brier", "macro_f1", "ap"]].astype(float),
        atol=TOL,
        rtol=0,
    )
    return {
        "beta": float(beta),
        "label_rows": len(b.y),
        "n_eff": n_eff,
        "batches": len(batches),
        "ap_change": float(values[-1] - saved.at[raw_name, "ap"]),
        "reordered_binary_pairs": not np.array_equal(
            np.argsort(test[:, 1], kind="stable"),
            np.argsort(predicted[:, 1], kind="stable"),
        ),
    }


def check_round(r, chosen):
    from banks_study import banks

    raw = table(r, "scores")
    assert not raw.duplicated([*KEY, "method"]).any()
    expected = layout(r, raw)
    expected_arms = expected_methods(r, chosen)
    blocks = raw.groupby(KEY)
    refs, probes, fitting, pairing, replayed = 0, [], {}, {}, set()
    metric_err = 0.0
    temperatures = []
    for b in banks(r):
        key = tuple(b.context[name] for name in KEY)
        assert key in expected and key not in replayed
        replayed.add(key)
        rows = blocks.get_group(key).set_index("method", verify_integrity=True)
        assert set(rows.index) == expected_arms, set(rows.index) ^ expected_arms
        check_bank(b, rows, pairing)
        fitting[key] = hash_array(b.x), hash_array(b.y)
        source, query = OLD["controls"](b.x, b.y, (b.x, b.audit_x))
        for m in source:
            fit_pred, pred = source[m], query[m]
            assert rows.at[m, "test_hash"] == hash_array(pred) and rows.at[
                m, "fit_hash"
            ] == hash_array(fit_pred)
            obs = np.array(
                [OLD["empirical"](fit_pred, b.y), *checked_measures(b, pred)]
            )
            saved = rows.loc[
                m, ["fit_brier", "brier", "expected_brier", "macro_f1", "ap"]
            ].to_numpy(dtype=float)
            np.testing.assert_allclose(obs, saved, atol=TOL, rtol=0, equal_nan=True)
            metric_err = max(metric_err, np.nanmax(np.abs(obs - saved)))
            refs += 1
        if r == 4 and chosen in {"cal_pull", "cal_sparse"} and key[0] == "factorial":
            temperatures.append(temperature_replay(b, chosen, rows))
        if key[1] == "h1c1s1" and key[2] == 491000 + 1000 * r:
            probes.append(audit_probe(r, b, rows, chosen))
    assert replayed == expected and len(probes) == 1
    return {
        "banks": len(replayed),
        "score_rows": len(raw),
        "cheap_fits": refs,
        "temperature_replay": {
            "focal_fits": len(temperatures),
            "beta_counts": {
                str(v): sum(t["beta"] == v for t in temperatures)
                for v in sorted({t["beta"] for t in temperatures})
            },
            "unique_source_row_counts": sorted({t["label_rows"] for t in temperatures}),
            "unique_independent_unit_counts": sorted(
                {t["n_eff"] for t in temperatures}
            ),
            "verified_label_batches": sum(t["batches"] for t in temperatures),
            "max_abs_binary_ap_change": max(
                [abs(t["ap_change"]) for t in temperatures], default=None
            ),
            "changed_sorted_orders": sum(
                t["reordered_binary_pairs"] for t in temperatures
            ),
        },
        "cheap_forecast_hashes": 2 * refs,
        "max_cheap_metric_error": float(metric_err),
        "audit_reversal": probes[0],
        "checks": summaries(r, raw),
    }, fitting


def factorial_checks(r, raw):
    """Independent intervention differences via a binary finite difference along each named axis."""
    data = raw[raw.cohort == "factorial"].copy()
    for index, key in enumerate("HCS"):
        data[key] = data.family.str[2 * index + 1].astype(int)
    results = table(r, "factorial-effects")
    k = 0
    for metric in ["brier", "expected_brier"]:
        arr = data.pivot(
            index=["H", "C", "S", "seed"], columns="method", values=metric
        ).sort_index()
        methods = list(arr)
        n = data.seed.nunique()
        cells = arr.to_numpy().reshape(2, 2, 2, n, len(methods))
        draws = np.random.default_rng(84100 + r).integers(n, size=(10000, 1, n))[:, 0]
        for factor, part in results[results.metric == metric].groupby("factor"):
            effect = cells
            for dim, name in enumerate("HCS"):
                if name in factor:
                    effect = np.diff(effect, axis=dim)
            per_seed = effect.mean(axis=(0, 1, 2))
            for row in part.itertuples(index=False):
                obs = per_seed[:, methods.index(row.method)]
                if row.reference != "absolute":
                    obs = obs - per_seed[:, methods.index(row.reference)]
                est = obs.mean(), *np.quantile(obs[draws].mean(1), [0.025, 0.975])
                np.testing.assert_allclose(
                    est, [row.delta, row.lo, row.hi], atol=TOL, rtol=0
                )
                k += 1
    return k


def raw_delta(round_no, family, metric, method):
    frame = table(round_no, "scores")
    if family in {"standard", "factorial"}:
        frame = frame[frame.cohort == family]
    else:
        frame = frame[frame.family == family]
    p = frame.pivot(index=["family", "seed"], columns="method", values=metric)
    return float((p[method] - p["pull_mean"]).groupby("family").mean().mean())


def choose():
    choices = []
    for r, method in ((2, "sparse_gp"), (3, "cal_pull"), (3, "cal_sparse")):
        violations = [
            max(0.0, raw_delta(r, group, "brier", method) / cap - 1)
            for group, cap in LIMITS.items()
        ]
        target = raw_delta(r, "factorial", "expected_brier", method)
        choices.append((max(violations), target, len(choices), method))
    return min(choices)[-1]


def check_lock():
    repo = HERE.parents[2]
    file = (HERE / "selection.json").relative_to(repo)
    commit = subprocess.check_output(
        ["git", "log", "--diff-filter=A", "-1", "--format=%H", "--", str(file)],
        cwd=repo,
        text=True,
    ).strip()
    assert commit
    blob = subprocess.check_output(["git", "show", f"{commit}:{file}"], cwd=repo)
    assert blob == (HERE / "selection.json").read_bytes()
    assert not subprocess.check_output(
        [
            "git",
            "ls-tree",
            commit,
            "--",
            str((OUT / "r4-scores.csv").relative_to(repo)),
        ],
        cwd=repo,
    )
    record = json.loads(blob)
    for r in (1, 2, 3):
        for table_name in ["scores", "contrasts"]:
            p = OUT / f"r{r}-{table_name}.csv"
            assert sha256(p.read_bytes()).hexdigest() == record["sha256"][p.name]
            assert (
                sha256(
                    subprocess.check_output(
                        ["git", "show", f"{commit}:{p.relative_to(repo)}"], cwd=repo
                    )
                ).digest()
                == sha256(p.read_bytes()).digest()
            )
    winner = choose()
    assert winner == record["chosen"]
    return commit, winner


def refit_finalist(b, chosen, saved):
    from engine import fit_model
    from interventions import fit_candidate
    from okapi.okapi import Okapi
    from sparse_gp import fit_sparse_gp

    from fusionbench.methods.okapi import OKAPIMethod

    opts = {
        "fitness": "brier",
        "seed": 0,
        "n_restarts": 1,
        "n_seeds": 0,
        "generations": 8,
        "population_size": 8,
        "population_multiplier": 2,
        "tournament_size": 3,
        "max_depth": 7,
        "max_nodes": max(31, b.x.shape[1] + 1),
    }
    original = Okapi._calculate_fitnesses
    counts = []

    def checked(instance, *args, **kwargs):
        np.testing.assert_array_equal(np.asarray(instance.gt_tensor).reshape(-1), b.y)
        counts.append(len(b.y))
        return original(instance, *args, **kwargs)

    with patch.object(Okapi, "_calculate_fitnesses", checked):
        full = fit_model(OKAPIMethod(**opts), b.x, b.y)
        if chosen == "cal_pull":
            base = fit_candidate("pull_mean", b.x, b.y, groups=b.groups)
        else:
            base = fit_sparse_gp("sparse_gp", b.x, b.y)
        if chosen.startswith("cal_"):
            from calibrate_gp import fit_calibrated

            winner = fit_calibrated(base, b.x, b.y, groups=b.groups)
            # Recompute the choice independently in probability space, with identity
            # retaining exact zeros and the same deterministic tie preference.
            p = base.predict(b.x)
            unit_count = len(b.x) if b.groups is None else len(np.unique(b.groups))

            def source_loss(j):
                beta = 2.0 ** (j / 4)
                adj = np.clip(p, 1e-6, 1 - 1e-6) ** beta
                prob = adj / adj.sum(1, keepdims=True) if j else p
                loss = (
                    OLD["empirical"](prob, b.y)
                    + 0.25 / np.sqrt(unit_count) * (j / 4) ** 2
                )
                return loss, abs(j), j

            step = min(range(-4, 5), key=source_loss)
            assert 2.0 ** (step / 4) == saved.at[chosen, "beta"]
            assert (
                winner.best_tree.unique_value_node_ids
                == base.best_tree.unique_value_node_ids
            )
        else:
            winner = base
    for name, m in (("full", full), (chosen, winner)):
        assert hash_array(m.predict(b.x)) == saved.at[name, "fit_hash"]
        assert hash_array(m.predict(b.audit_x)) == saved.at[name, "test_hash"]
    assert len(counts) >= 2 and set(counts) == {len(b.y)}
    return {
        "family": b.context["family"],
        "seed": b.context["seed"],
        "models": ["full", chosen],
        "all_source_rows_per_native_fitness_batch": len(b.y),
        "native_batches": len(counts),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", type=int, choices=range(1, 5))
    args = parser.parse_args()
    t0 = perf_counter()
    lock, chosen = (None, None) if args.round in (1, 2, 3) else check_lock()
    rounds = {}
    round_inputs = set()
    with threadpool_limits(limits=1):
        for r in [args.round] if args.round else [1, 2, 3, 4]:
            report, source = check_round(r, chosen)
            current_inputs = {(k[0], k[1], *v) for k, v in source.items()}
            assert not round_inputs & current_inputs
            round_inputs |= current_inputs
            report["factorial_intervals"] = factorial_checks(r, table(r, "scores"))
            rounds[r] = report
        data = {"rounds": rounds}
        if 4 in rounds:
            import run_four  # Adds the original full-data study source to the research import path.
            from banks_study import banks

            assert run_four.fit_models
            pool = table(4, "scores").groupby(KEY)
            reps = []
            specs = {
                ("factorial", "h1c1s1"): 495000,
                ("standard", "committee"): 495400,
                ("standard", "multiclass_committee"): 495400,
                ("clustered", "scene16"): 495100,
            }
            for b in banks(4):
                k = tuple(b.context[name] for name in KEY)
                if k[2] == specs.get(k[:2]):
                    reps.append(
                        refit_finalist(b, chosen, pool.get_group(k).set_index("method"))
                    )
            assert len(reps) == len(specs)
            out = table(4, "contrasts")

            def hi(group, metric):
                row = out[
                    (out.group == group)
                    & (out.metric == metric)
                    & (out.method == chosen)
                    & (out.reference == "pull_mean")
                ]
                assert len(row) == 1
                return float(row.iloc[0].hi)

            primary = hi("factorial", "expected_brier") < 0
            limits = {g: hi(g, "brier") <= cap for g, cap in LIMITS.items()}
            reported = json.loads((OUT / "confirmation.json").read_text())
            assert reported == {
                "chosen": chosen,
                "primary_pass": primary,
                "breadth_checks": limits,
                "broad_pass": primary and all(limits.values()),
            }
            data.update(
                lock=lock,
                chosen=chosen,
                confirmations=limits,
                primary_pass=primary,
                broad_pass=primary and all(limits.values()),
                native_confirmation_replays=reps,
            )
    data["seconds"] = perf_counter() - t0
    text = json.dumps(data, indent=2, allow_nan=False) + "\n"
    if args.round:
        print(text)
    else:
        (OUT / "verification.json").write_text(text)


if __name__ == "__main__":
    main()

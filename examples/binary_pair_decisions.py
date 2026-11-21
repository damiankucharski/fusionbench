"""Measure exact binary pair decisions and certified routing among six source experts."""

import argparse
import gzip
import json
import time
from datetime import UTC, datetime
from itertools import combinations, combinations_with_replacement
from pathlib import Path

import numpy as np
import pandas as pd
from okapi.binary_pairs import compile_binary_pairs, midpoint_certificate
from real_sigmoid_pool import (
    fit_member,
    load_har,
    load_spam,
    paired_interval,
    row_weights,
)
from scipy.optimize import brentq
from scipy.special import expit
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


def matchings(indices):
    if not indices:
        yield ()
        return
    first, *rest = indices
    for i, j in enumerate(rest):
        for tail in matchings(rest[:i] + rest[i + 1 :]):
            yield ((first, j), *tail)


PARTITIONS = list(matchings(list(range(6))))
EDGES = list(combinations(range(6), 2))


def centers(logits, partition):
    i, j = np.asarray(partition).T
    return logits[:, i] / 2 + logits[:, j] / 2


def optimal_certificate(logits):
    order = np.sort(logits, axis=1)
    mid = order[:, :3] / 2 + order[:, ::-1][:, :3] / 2
    return midpoint_certificate(mid)


def controls():
    grid = np.array(list(combinations_with_replacement([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], 6)))
    # All 15 possible perfect matchings, including weak inequalities and true ties.
    union = np.logical_or.reduce([midpoint_certificate(centers(grid, k))[1] for k in PARTITIONS])
    np.testing.assert_array_equal(optimal_certificate(grid)[1], union)
    functions = {
        "unequal_pair": lambda x: 0.55 * expit(x) + 0.45 * expit(-10 * x) - 0.5,
        "equal_triple": lambda x: (expit(10 * (x + 1)) + expit(-10 * x) + expit(10 * (x - 1))) / 3 - 0.5,
    }
    tests = {}
    for name, f in functions.items():
        right = brentq(f, 0.5, 10.0)
        probes = np.array([-10.0, -0.5, 0.5, 10.0])
        p = f(probes) + 0.5
        np.testing.assert_array_equal(p >= 0.5, [False, True, False, True])
        np.testing.assert_allclose(f(np.array([-right, 0.0, right])), 0.0, atol=1e-12)
        tests[name] = {
            "crossings": [-right, 0.0, right],
            "probe_inputs": probes.tolist(),
            "probabilities": p.tolist(),
        }
    return {"multisets_tested": len(grid), "partitions": len(PARTITIONS), "counterexamples": tests}


def strong_baselines(data, x):
    dimensions = np.arange(x.shape[1])
    raw = fit_member(x, data, dimensions)
    key = "raw"
    best = raw
    pred = expit(x[data.test] @ raw.w + raw.b)
    if data.name == "spambase":
        log = np.log1p(data.x)
        for name, features in [("log1p", log), ("raw_and_log1p", np.column_stack([data.x, log]))]:
            alt = StandardScaler().fit(features[data.train]).transform(features)
            model = fit_member(alt, data, np.arange(alt.shape[1]))
            if model.val_loss < best.val_loss:
                best = model
                key = name
                pred = expit(alt[data.test] @ best.w + best.b)
    return expit(x[data.test] @ raw.w + raw.b), pred, key


def bank(data, x, scaler, case, raw_full, strong, strong_kind):
    dims = np.arange(x.shape[1])
    _train, val, test = data.train, data.val, data.test
    vwt = row_weights(data, val)
    twt = row_weights(data, test)
    models = [
        fit_member(x, data, dims[(dims % 6 == j) if case == "disjoint" else (dims % 6 != j)])
        for j in range(6)
    ]
    w = np.stack([m.w for m in models]) / scaler.scale_
    b = np.array([m.b for m in models]) - w @ scaler.mean_
    v = data.x[val] @ w.T + b
    z = data.x[test] @ w.T + b
    val_p = expit(v)
    p = expit(z)
    y = data.y[test]
    key = {"dataset": data.name, "case": case}
    # Pair selection requires source labels. Selection of a matching uses only source logits.
    losses = [vwt @ ((val_p[:, i] / 2 + val_p[:, j] / 2 - data.y[val]) ** 2) for i, j in EDGES]
    idx = int(np.argmin(losses))
    i, j = EDGES[idx]
    best = int(np.argmin([m.val_loss for m in models]))
    pair_p = (p[:, i] + p[:, j]) / 2
    pair_compiler = compile_binary_pairs(w[[i, j]], b[[i, j]], [(0, 1)])
    pair_rule = pair_compiler.predict(data.x[test])
    pair_score = data.x[test] @ pair_compiler.center_weights[0] + pair_compiler.center_bias[0]
    pool = p.mean(1)
    reference = pool >= 0.5
    predictions = {
        "chosen_pair": pair_p,
        "pair_compiled": expit(pair_score),
        "best_member": p[:, best],
        "full_raw": raw_full,
        "strong_logistic": strong,
        "six_pool": pool,
    }
    methods = []
    pairs = []
    costs = []
    contrasts = []
    groups = []
    for name, prob in predictions.items():
        label = pair_rule if name == "pair_compiled" else prob >= 0.5
        methods.append(
            dict(
                **key,
                method=name,
                brier=twt @ ((prob - y) ** 2),
                accuracy=twt @ (label == y),
                pair_rmse=np.sqrt(twt @ ((prob - pair_p) ** 2))
                if name in ("chosen_pair", "pair_compiled")
                else np.nan,
            )
        )
    for name in ["best_member", "full_raw", "strong_logistic"]:
        a = np.asarray((pair_p >= 0.5) == y, dtype=float)
        used = (predictions[name] >= 0.5) == y
        ci = paired_interval(data.groups[test], a - used.astype(float), data.name)
        contrasts.append(dict(**key, contrast="pair_accuracy_minus_" + name, **ci))
    for k, (u, vv) in enumerate(EDGES):
        half_w = w[u] / 2 + w[vv] / 2
        half_b = b[u] / 2 + b[vv] / 2
        scores = data.x[test] @ half_w + half_b
        full = (p[:, u] + p[:, vv]) / 2
        top = full >= 0.5
        compiled = scores >= 0
        mismatches = np.count_nonzero(compiled != top)
        nonround = np.count_nonzero((top != compiled) & (np.abs(full - 0.5) > 1e-12))
        if nonround:
            raise AssertionError("Difference outside the declared floating-point tolerance")
        pairs.append(
            dict(
                **key,
                first=u,
                second=vv,
                selected=k == idx,
                validation_brier=losses[k],
                test_rows=len(test),
                legacy_float_ties=np.count_nonzero(full == 0.5),
                compiled_ties=np.count_nonzero(scores == 0),
                float_decision_mismatches=mismatches,
                largest_margin_of_mismatch=np.max(abs(full[compiled != top] - 0.5), initial=0),
                probability_mean_abs_error=twt @ np.abs(expit(scores) - full),
            )
        )
    val_masks = [midpoint_certificate(centers(v, part))[1] for part in PARTITIONS]
    val_cover = [float(vwt @ c) for c in val_masks]
    selected = int(np.argmax(val_cover))
    individual, individual_cert = midpoint_certificate(z)
    optimal, optimal_cert = optimal_certificate(z)
    coverage = {}
    split_rows = []
    for name, cert, labels in [
        ("individual", individual_cert, individual),
        ("dynamic_extremes", optimal_cert, optimal),
    ]:
        routed = np.where(cert, labels, reference)
        coverage[name] = cert
        costs.append(
            dict(
                **key,
                method=name,
                coverage=twt @ cert,
                mean_dense_dots=6,
                mean_dense_macs=6 * len(dims),
                mean_nonzero_macs=np.count_nonzero(w),
                mean_sigmoids=6 * (twt @ (~cert)),
                fallback_agreement=twt @ (routed == reference),
                mismatches=np.count_nonzero(routed != reference),
            )
        )
    for k, part in enumerate(PARTITIONS):
        router = compile_binary_pairs(w, b, part)
        routed, cert = router.predict(data.x[test], return_certificate=True)
        # Closeness is separately recorded if the naive score and real-valued sign differ at rounding.
        if np.any((routed != reference) & (np.abs(pool - 0.5) > 1e-12)):
            raise AssertionError("Routed class differs even away from numerical ties")
        if np.any(individual_cert & ~cert) or np.any(cert & ~optimal_cert):
            raise AssertionError("Coverage bounds violated")
        split_rows.append(
            dict(
                **key,
                partition=k,
                edges=json.dumps(part),
                selected=k == selected,
                val_cover=val_cover[k],
                test_cover=twt @ cert,
                mismatches=np.count_nonzero(routed != reference),
            )
        )
        for flag, enabled in [("ordered", k == 0), ("source_matching", k == selected)]:
            if enabled:
                coverage[flag] = cert
                failure = float(twt @ (~cert))
                ncenter = np.count_nonzero(router.center_weights)
                ndiff = np.count_nonzero(router.difference_weights)
                costs.append(
                    dict(
                        **key,
                        method=flag,
                        coverage=twt @ cert,
                        mean_dense_dots=3 + 3 * failure,
                        mean_dense_macs=(3 + 3 * failure) * len(dims),
                        mean_nonzero_macs=ncenter + failure * ndiff,
                        mean_sigmoids=6 * failure,
                        fallback_agreement=twt @ (routed == reference),
                        mismatches=np.count_nonzero(routed != reference),
                    )
                )
    costs.append(
        dict(
            **key,
            method="full_six",
            coverage=0.0,
            mean_dense_dots=6,
            mean_dense_macs=6 * len(dims),
            mean_nonzero_macs=np.count_nonzero(w),
            mean_sigmoids=6.0,
            fallback_agreement=1.0,
            mismatches=0,
        )
    )
    for left, right in [
        ("source_matching", "individual"),
        ("source_matching", "ordered"),
        ("dynamic_extremes", "source_matching"),
    ]:
        delta = coverage[left].astype(float) - coverage[right]
        contrasts.append(
            dict(
                **key,
                contrast=left + "_coverage_minus_" + right,
                **paired_interval(data.groups[test], delta, data.name),
            )
        )
    for g in np.unique(data.groups[test]):
        mask = data.groups[test] == g
        groups.append(
            dict(
                **key,
                group=int(g),
                n=mask.sum(),
                sum_pair_correct=np.sum(((pair_p >= 0.5) == y)[mask]),
                sum_compiled_correct=np.sum((pair_rule == y)[mask]),
                sum_six_correct=np.sum((reference == y)[mask]),
                **{k + "_certified": np.sum(v[mask]) for k, v in coverage.items()},
            )
        )
    export = dict(
        **key,
        selected_pair=[i, j],
        pair_weights=pair_compiler.center_weights[0].tolist(),
        pair_bias=float(pair_compiler.center_bias[0]),
        expert_weights=w.tolist(),
        expert_bias=b.tolist(),
        selected_member=best,
        regularization=[m.c for m in models],
        selected_matching=PARTITIONS[selected],
        strong_feature_set=strong_kind,
    )
    return methods, pairs, costs, contrasts, split_rows, groups, export


def run(spam, har, out):
    tic = time.perf_counter()
    start = datetime.now(UTC).isoformat()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tables = [[] for _ in range(6)]
    exports = []
    analytic = controls()
    for data in (load_spam(spam), load_har(har)):
        for a, b in combinations((data.train, data.val, data.test), 2):
            if np.intersect1d(data.groups[a], data.groups[b]).size:
                raise AssertionError("Group leakage")
        scaler = StandardScaler().fit(data.x[data.train])
        x = scaler.transform(data.x)
        raw, strong, kind = strong_baselines(data, x)
        for case in ("disjoint", "overlap"):
            result = bank(data, x, scaler, case, raw, strong, kind)
            for table, rows in zip(tables, result[:-1]):
                table.extend(rows)
            exports.append(result[-1])
    for name, rows in zip(
        ("methods", "pairs", "routing", "contrasts", "matchings", "test_groups"), tables
    ):
        pd.DataFrame(rows).to_csv(out / (name + ".csv"), index=False)
    (out / "models.json.gz").write_bytes(
        gzip.compress(json.dumps(exports, allow_nan=False).encode(), mtime=0)
    )
    (out / "config.json").write_text(
        json.dumps(
            {
                "analytic": analytic,
                "compute_seconds": time.perf_counter() - tic,
                "compute_started_utc": start,
                "compute_ended_utc": datetime.now(UTC).isoformat(),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spambase", required=True)
    parser.add_argument("--har", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.spambase, args.har, args.out)

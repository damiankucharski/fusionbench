"""Audit multi-class compression and top-1 certificates on digits and UCI HAR."""

import argparse
import json
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.datasets import load_digits
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from okapi.softmax_pool import compile_softmax_pool
from real_sigmoid_pool import Data
from real_transfer import populations


def datasets(har):
    x, y = load_digits(return_X_y=True)
    _, g = np.unique(x, axis=0, return_inverse=True)
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=3031)
    folds = [idx for _, idx in cv.split(x, y, g)]
    yield Data("digits", x, y, g, np.concatenate(folds[2:]), folds[1], folds[0])
    _, x, y, g, (base, a, b, test) = next(d for d in populations(har) if d[0] == "har")
    yield Data("har", x.astype(float), y, g, np.r_[base, a], b, test)


def row_weights(dataset, indices):
    if dataset.name == "digits":
        return np.full(len(indices), 1 / len(indices))
    _, inverse, count = np.unique(dataset.groups[indices], return_inverse=True, return_counts=True)
    return 1 / (len(count) * count[inverse])


def brier(probabilities, y):
    return np.sum((probabilities - np.eye(probabilities.shape[1])[y]) ** 2, axis=1)


@dataclass
class Expert:
    weights: np.ndarray
    intercepts: np.ndarray
    validation_brier: float
    c: float


def fit_member(dataset, scaler, standardized, columns):
    train, val = dataset.train, dataset.val
    weights = row_weights(dataset, val)
    best = None
    model = LogisticRegression(max_iter=4000, tol=1e-7, warm_start=True)
    train_columns = standardized[train][:, columns]
    validation_columns = standardized[val][:, columns]
    for c in (0.01, 0.1, 1.0):
        model.set_params(C=c)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit(train_columns, dataset.y[train])
        np.testing.assert_array_equal(model.classes_, np.arange(len(np.unique(dataset.y))))
        loss = weights @ brier(model.predict_proba(validation_columns), dataset.y[val])
        if best is None or loss < best.validation_brier:
            w = np.zeros((len(model.classes_), dataset.x.shape[1]))
            w[:, columns] = model.coef_ / scaler.scale_[columns]
            b = model.intercept_ - w @ scaler.mean_
            best = Expert(w, b, loss, c)
    return best


def intervals(groups, losses, subject_balanced):
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    sums = np.column_stack([np.bincount(inverse, weights=v) for v in losses.T])
    if subject_balanced:
        sums = sums / counts[:, None]
        counts = np.ones_like(counts)
    rng = np.random.default_rng(3033)
    draws = []
    for _ in range(16):
        ids = rng.integers(len(counts), size=(250, len(counts)))
        draws.append(sums[ids].sum(1) / counts[ids].sum(1)[:, None])
    q = np.quantile(np.concatenate(draws), [0.025, 0.975], axis=0)
    return sums.sum(0) / counts.sum(), q[0], q[1]


def average_pool(x, w, b, alpha):
    components = np.stack([softmax(x @ wi.T + bi, axis=1) for wi, bi in zip(w, b)])
    return np.tensordot(alpha, components, axes=1), components


def gauge_check(w, b, alpha, x, old_pool, old_model):
    shifted_w, shifted_b = w.copy(), b.copy()
    rng = np.random.default_rng(3032)
    shifted_w[0] += rng.normal(size=w.shape[-1])
    shifted_b[0] += rng.normal()
    changed = compile_softmax_pool(shifted_w, shifted_b, alpha)
    new_pool, _ = average_pool(x, shifted_w, shifted_b, alpha)
    discrepancies = {}
    for name, left, right in [
        ("pool", old_pool, new_pool),
        ("compiled", old_model.predict(x), changed.predict(x)),
        ("quadratic", old_model.error_bound(x), changed.error_bound(x)),
        ("oscillation", old_model.oscillation_bound(x), changed.oscillation_bound(x)),
    ]:
        np.testing.assert_allclose(left, right, atol=2e-9, rtol=2e-11)
        discrepancies[name] = float(np.max(np.abs(left - right)))
    return discrepancies


def evaluate(dataset, export_dir=None):
    train, val, test = dataset.train, dataset.val, dataset.test
    for a, b in ((train, val), (val, test), (train, test)):
        if np.intersect1d(dataset.groups[a], dataset.groups[b]).size:
            raise ValueError("Splits must have disjoint groups")
    x, y = dataset.x[test], dataset.y[test]
    weight = row_weights(dataset, test)
    scaler = StandardScaler().fit(dataset.x[train])
    std = scaler.transform(dataset.x)
    cols = np.arange(std.shape[1])
    nonconstant = cols[scaler.var_ > 0]
    omit = nonconstant[np.linspace(0, len(nonconstant) - 1, num=5, dtype=int)]
    full = fit_member(dataset, scaler, std, cols)
    fullp = softmax(x @ full.weights.T + full.intercepts, axis=1)
    alpha = np.full(5, 0.2)
    points, summaries, contrasts, models, gauges = [], [], [], [], []
    for condition in ("disjoint", "overlap", "near_duplicate", "duplicate"):
        experts = []
        for j in range(5):
            if condition == "duplicate":
                experts.append(full)
                continue
            if condition == "near_duplicate":
                mask = cols != omit[j]
            elif condition == "overlap":
                mask = cols % 5 != j
            else:
                mask = cols % 5 == j
            experts.append(fit_member(dataset, scaler, std, cols[mask]))
        chosen = int(np.argmin([e.validation_brier for e in experts]))
        w = np.stack([e.weights for e in experts])
        b = np.stack([e.intercepts for e in experts])
        pool, members = average_pool(x, w, b, alpha)
        compiled = compile_softmax_pool(w, b, alpha)
        if export_dir is not None:
            np.savez_compressed(
                export_dir / f"{dataset.name}_{condition}.npz",
                weights=w,
                intercepts=b,
                mixing_weights=alpha,
                projected_weights=compiled.weights,
                projected_intercepts=compiled.intercepts,
                training_mean=scaler.mean_,
                training_scale=scaler.scale_,
                test_indices=test,
            )
        compact = compiled.predict(x)
        np.testing.assert_allclose(
            compact, softmax(x @ compiled.weights.T + compiled.intercepts, axis=1), atol=1e-11
        )
        gauges.append(
            dict(
                dataset=dataset.name, condition=condition, **gauge_check(w, b, alpha, x, pool, compiled)
            )
        )
        tv = np.abs(pool - compact).sum(1) / 2
        bounds = {
            "quadratic": np.asarray(compiled.error_bound(x)),
            "oscillation": np.asarray(compiled.oscillation_bound(x)),
        }
        for bound in bounds.values():
            if bound.shape != tv.shape or not np.isfinite(bound).all() or np.any(tv > bound + 1e-9):
                raise AssertionError("Invalid TV certificate")
            if np.any(bound < -1e-12):
                raise AssertionError("Negative TV certificate")
        if np.any(bounds["oscillation"] > bounds["quadratic"] + 1e-9):
            raise AssertionError("Oscillation must be at least as sharp as the quadratic relaxation")
        if condition == "duplicate":
            for a in [tv, *bounds.values()]:
                np.testing.assert_allclose(a, 0, atol=1e-10)
        compact_order = np.sort(compact, axis=1)
        margin = compact_order[:, -1] - compact_order[:, -2]
        agree = pool.argmax(1) == compact.argmax(1)
        values = {
            "pool": pool,
            "compiled": compact,
            "best_member": members[chosen],
            "full_logistic": fullp,
        }
        losses = {name: brier(p, y) for name, p in values.items()}
        key = dict(dataset=dataset.name, condition=condition)
        row = dict(**key, mean_tv=weight @ tv, max_tv=tv.max(), top1_agreement=weight @ agree)
        for name, p in values.items():
            row[name + "_brier"] = weight @ losses[name]
            row[name + "_accuracy"] = weight @ (p.argmax(1) == y)
        for name, bound in bounds.items():
            guaranteed = margin > 2 * bound
            if np.any(guaranteed & ~agree):
                raise AssertionError("A certified example changed its winning class")
            row[name + "_top1_coverage"] = weight @ guaranteed
            row[name + "_maximum_excess"] = np.max(tv - bound)
            for lim in (0.05, 0.10):
                row[name + f"_coverage_{lim:.2f}"] = weight @ (bound <= lim)
                row[f"actual_within_{lim:.2f}"] = weight @ (tv <= lim)
        summaries.append(row)
        labels = ["best_member", "full_logistic", "compiled"]
        diffs = np.column_stack([losses[name] - losses["pool"] for name in labels])
        estimates, lo, hi = intervals(dataset.groups[test], diffs, dataset.name == "har")
        for tag, v, left, right in zip(labels, estimates, lo, hi):
            contrasts.append(dict(**key, contrast=tag + "_minus_pool", difference=v, lo=left, hi=right))
        points.append(
            pd.DataFrame(
                dict(
                    **key,
                    row=test,
                    group=dataset.groups[test],
                    label=y,
                    tv=tv,
                    quadratic=bounds["quadratic"],
                    oscillation=bounds["oscillation"],
                    margin=margin,
                    top1_agree=agree,
                    **{name + "_class": p.argmax(1) for name, p in values.items()},
                    **{name + "_brier": v for name, v in losses.items()},
                )
            )
        )
        models.append(
            dict(
                **key,
                full_C=full.c,
                selected_member=chosen,
                member_C=[m.c for m in experts],
                train_rows=len(train),
                validation_rows=len(val),
                test_rows=len(test),
                test_groups=len(np.unique(dataset.groups[test])),
                omitted_near_duplicate_indices=omit.tolist(),
            )
        )
    return points, summaries, contrasts, models, gauges


def run(har, out, *, export_experts=False):
    start = time.perf_counter()
    began = datetime.now(timezone.utc).isoformat()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    export_dir = out / "frozen_experts" if export_experts else None
    if export_dir is not None:
        export_dir.mkdir(exist_ok=True)
    tables = [[] for _ in range(5)]
    for dataset in datasets(har):
        for aggregate, part in zip(tables, evaluate(dataset, export_dir)):
            aggregate.extend(part)
    pd.concat(tables[0], ignore_index=True).to_csv(
        out / "test_examples.csv.gz", index=False, compression={"method": "gzip", "mtime": 0}
    )
    for name, data in zip(("results", "contrasts", "fits", "gauge_checks"), tables[1:]):
        pd.DataFrame(data).to_csv(out / (name + ".csv"), index=False)
    (out / "timing.json").write_text(
        json.dumps(
            dict(
                compute_seconds=time.perf_counter() - start,
                compute_started_utc=began,
                compute_ended_utc=datetime.now(timezone.utc).isoformat(),
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--har", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--export-experts",
        action="store_true",
        help="Save each fitted pool and compiled coefficients in raw feature coordinates as NPZ.",
    )
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.har, args.out, export_experts=args.export_experts)

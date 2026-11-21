"""Compare utility and parameter-only compression of logistic opinion pools on real data."""

import argparse
import gzip
import hashlib
import json
import time
import warnings
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from okapi.sigmoid_pool import collapse_sigmoids
from real_transfer import populations

SPAM_SHA = "813ac1df8effac70463c09c9c4b11e8803eefcab54771af66150852bcdcd1636"
C_GRID = (0.01, 0.1, 1.0, 10.0)


@dataclass
class Data:
    name: str
    x: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def load_spam(path):
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != SPAM_SHA:
        raise ValueError("Expected the preregistered UCI Spambase archive")
    with zipfile.ZipFile(path) as archive, archive.open("spambase.data") as f:
        table = np.loadtxt(f, delimiter=",")
    x, y = table[:, :-1], table[:, -1].astype(int)
    _, groups = np.unique(x, axis=0, return_inverse=True)
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=2901)
    folds = [idx for _, idx in cv.split(x, y, groups)]
    return Data("spambase", x, y, groups, np.concatenate(folds[2:]), folds[1], folds[0])


def load_har(path):
    _, x, labels, groups, (base, a, b, test) = next(v for v in populations(path) if v[0] == "har")

    def keep(rows):
        return rows[np.isin(labels[rows], (1, 2))]

    return Data(
        "har_stairs",
        x.astype(np.float64),
        (labels == 2).astype(int),
        groups,
        keep(np.r_[base, a]),
        keep(b),
        keep(test),
    )


def row_weights(data, idx):
    if data.name == "spambase":
        return np.full(len(idx), 1 / len(idx))
    _, inverse, counts = np.unique(data.groups[idx], return_inverse=True, return_counts=True)
    return 1 / (len(counts) * counts[inverse])


@dataclass
class Rule:
    w: np.ndarray
    b: float
    c: float
    val_loss: float


def fit_member(x, data, columns):
    best = None
    xtrain = x[data.train][:, columns]
    xval = x[data.val][:, columns]
    weight = row_weights(data, data.val)
    for c in C_GRID:
        model = LogisticRegression(C=c, max_iter=4000, tol=1e-7)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit(xtrain, data.y[data.train])
        p = model.predict_proba(xval)[:, 1]
        loss = weight @ ((p - data.y[data.val]) ** 2)
        if best is None or loss < best.val_loss:
            w = np.zeros(x.shape[1])
            w[columns] = model.coef_[0]
            best = Rule(w, float(model.intercept_[0]), c, float(loss))
    return best


def distill(train, soft_targets, w0, b0):
    def objective(theta):
        logits = train @ theta[:-1] + theta[-1]
        loss = np.logaddexp(0, logits).sum() - soft_targets @ logits + 0.005 * (theta[:-1] @ theta[:-1])
        delta = expit(logits) - soft_targets
        grad = np.r_[train.T @ delta + 0.01 * theta[:-1], delta.sum()]
        return loss, grad

    fit = minimize(
        objective,
        np.r_[w0, b0],
        jac=True,
        method="L-BFGS-B",
        options=dict(maxiter=5000, gtol=1e-7, ftol=1e-13),
    )
    if not fit.success:
        raise RuntimeError(f"Distillation failed: {fit.message}")
    return fit.x[:-1], float(fit.x[-1])


def group_frame(group, values):
    ids, ix, count = np.unique(group, return_inverse=True, return_counts=True)
    return ids, count, np.stack([np.bincount(ix, weights=v, minlength=len(ids)) for v in values], axis=1)


def paired_interval(group, delta, name):
    _, counts, total = group_frame(group, [delta])
    values = total[:, 0]
    if name == "har_stairs":
        values = values / counts
        counts = np.ones_like(counts)
    rng = np.random.default_rng(2902)
    draws = []
    for _ in range(16):
        ids = rng.integers(len(counts), size=(250, len(counts)))
        draws.extend(values[ids].sum(1) / counts[ids].sum(1))
    return dict(
        difference=values.sum() / counts.sum(),
        lo=np.quantile(draws, 0.025),
        hi=np.quantile(draws, 0.975),
    )


def summarize_dataset(data):
    train, val, test = data.train, data.val, data.test
    if any(
        np.intersect1d(data.groups[a], data.groups[b]).size
        for a, b in ((train, val), (val, test), (train, test))
    ):
        raise AssertionError("Training, selection and test groups must be disjoint")
    if not all(np.unique(data.y[i]).size == 2 for i in (train, val, test)):
        raise ValueError("Both classes must occur in every partition")
    scale = StandardScaler().fit(data.x[train])
    x = scale.transform(data.x)
    dims = np.arange(x.shape[1])
    full = fit_member(x, data, dims)
    strong, alt_name = x[test] @ full.w + full.b, "raw"
    alt_loss = full.val_loss
    if data.name == "spambase":
        expanded = np.log1p(data.x)
        for tag, raw in [("log1p", expanded), ("raw_and_log1p", np.column_stack([data.x, expanded]))]:
            alt = StandardScaler().fit(raw[train]).transform(raw)
            m = fit_member(alt, data, np.arange(alt.shape[1]))
            if m.val_loss < alt_loss:
                strong, alt_name, alt_loss = alt[test] @ m.w + m.b, tag, m.val_loss
    metric_weight = row_weights(data, test)
    y = data.y[test]
    rows, contrasts, group_rows, certificates, exports = [], [], [], [], []
    split_meta = {
        key: dict(
            rows=len(ids),
            groups=len(np.unique(data.groups[ids])),
            positive=int(data.y[ids].sum()),
            sha256=hashlib.sha256(np.sort(ids).astype("<i8").tobytes()).hexdigest(),
        )
        for key, ids in [("fit", train), ("validation", val), ("test", test)]
    }
    split_meta.update(full_C=full.c, strong_feature_set=alt_name)
    for scenario in ("disjoint", "overlap", "duplicate"):
        if scenario == "duplicate":
            experts = [full] * 6
        else:
            experts = [
                fit_member(x, data, dims[(dims % 6 == j) if scenario == "disjoint" else (dims % 6 != j)])
                for j in range(6)
            ]
        w = np.stack([m.w for m in experts])
        b = np.array([m.b for m in experts])
        wt = np.full(6, 1 / 6)
        logits = x[test] @ w.T + b
        pool = expit(logits).mean(axis=1)
        ranked = int(np.argmin([m.val_loss for m in experts]))
        center = x[train].mean(0)
        cov = np.cov(x[train], rowvar=False, bias=True)
        wavg, bavg = w.mean(0), b.mean()
        rules = {
            "average_logits": (wavg, bavg),
            "tangent": collapse_sigmoids(w, b, wt, center),
            "secant": collapse_sigmoids(w, b, wt, center, covariance=cov),
            "distilled": distill(x[train], expit(x[train] @ w.T + b).mean(1), wavg, bavg),
            "best_member": (experts[ranked].w, experts[ranked].b),
            "full_logistic": (full.w, full.b),
        }
        pred = {"pool": pool, "strong_logistic": expit(strong)}
        for tag, (weights, bias) in rules.items():
            pred[tag] = expit(x[test] @ weights + bias)
            if tag in ("average_logits", "tangent", "secant", "distilled"):
                raw_w = weights / scale.scale_
                raw_b = float(bias - scale.mean_ @ raw_w)
                np.testing.assert_allclose(
                    expit(data.x[test] @ raw_w + raw_b), pred[tag], atol=1e-11, rtol=1e-11
                )
                exports.append(
                    dict(
                        dataset=data.name,
                        scenario=scenario,
                        method=tag,
                        weights=raw_w.tolist(),
                        intercept=raw_b,
                    )
                )
        if scenario == "duplicate":
            for tag in ("tangent", "secant", "average_logits", "full_logistic"):
                np.testing.assert_allclose(pred[tag], pool, atol=1e-11)
        per_loss = {k: (v - y) ** 2 for k, v in pred.items()}
        for tag, p in pred.items():
            loss = per_loss[tag]
            rows.append(
                dict(
                    dataset=data.name,
                    scenario=scenario,
                    method=tag,
                    brier=metric_weight @ loss,
                    accuracy=metric_weight @ ((p >= 0.5) == y),
                    pool_rmse=np.sqrt(metric_weight @ ((p - pool) ** 2)),
                )
            )
            ids, counts, totals = group_frame(data.groups[test], [loss, (p - pool) ** 2])
            group_rows.extend(
                dict(
                    dataset=data.name,
                    scenario=scenario,
                    method=tag,
                    group_id=int(g),
                    count=int(n),
                    sum_brier=float(tot[0]),
                    sum_pool_sq_diff=float(tot[1]),
                )
                for g, n, tot in zip(ids, counts, totals)
            )
        for tag in (
            "best_member",
            "full_logistic",
            "strong_logistic",
            "tangent",
            "secant",
            "average_logits",
            "distilled",
        ):
            result = paired_interval(data.groups[test], per_loss[tag] - per_loss["pool"], data.name)
            contrasts.append(
                dict(dataset=data.name, scenario=scenario, contrast=f"{tag}_minus_pool", **result)
            )
        actual = np.abs(pred["average_logits"] - pool)
        bound = np.var(logits, axis=1) / (12 * np.sqrt(3))
        certified = bound <= 0.05
        violations = (actual - bound) > 1e-12
        if violations.any():
            raise AssertionError("Logit-variance certificate violated")
        certificates.append(
            dict(
                dataset=data.name,
                scenario=scenario,
                certified_fraction=metric_weight @ certified,
                actual_within_005_fraction=metric_weight @ (actual <= 0.05),
                max_error=float(actual.max()),
                max_certified_error=float(actual[certified].max()) if certified.any() else np.nan,
                violations=int(violations.sum()),
                best_member=ranked,
                member_C=json.dumps([m.c for m in experts]),
            )
        )
    return rows, contrasts, group_rows, certificates, exports, split_meta


def run(spam, har, out):
    tic = time.perf_counter()
    began = datetime.now(timezone.utc).isoformat()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tables = [[] for _ in range(5)]
    metadata = {}
    for data in (load_spam(spam), load_har(har)):
        result = summarize_dataset(data)
        for table, subset in zip(tables, result[:5]):
            table.extend(subset)
        metadata[data.name] = result[5]
    for name, table in zip(
        ("methods.csv", "contrasts.csv", "test_groups.csv.gz", "certificates.csv"), tables[:4]
    ):
        pd.DataFrame(table).to_csv(
            out / name,
            index=False,
            compression={"method": "gzip", "mtime": 0} if name.endswith("gz") else None,
        )
    (out / "raw_models.json.gz").write_bytes(gzip.compress(json.dumps(tables[4]).encode(), mtime=0))
    (out / "config.json").write_text(
        json.dumps(
            dict(
                datasets=metadata,
                bootstrap_repeats=4000,
                compute_seconds=time.perf_counter() - tic,
                compute_started_utc=began,
                compute_ended_utc=datetime.now(timezone.utc).isoformat(),
            ),
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

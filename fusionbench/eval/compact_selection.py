"""Nested supervised selection of a compact radial network or a linear head."""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from examples.uci_three_binary import populations
from fusionbench.methods.compact_rbf import compile_standardized_rbf


def splits(y, groups, k, seed):
    splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    folds = list(splitter.split(np.zeros(len(y)), y, groups))
    for tr, te in folds:
        if np.intersect1d(groups[tr], groups[te]).size:
            raise ValueError("identical raw features cross a split")
    return folds


def models(x, y, groups, z, seed):
    start = time.perf_counter()
    scaler = StandardScaler()
    a = scaler.fit_transform(x)
    b = scaler.transform(z)
    linear = LogisticRegression(C=1.0, max_iter=3000).fit(a, y)
    probs = {"logit": linear.predict_proba(b)[:, 1]}
    basis = Nystroem(
        kernel="rbf", gamma=1 / x.shape[1], n_components=64, random_state=seed
    )
    head = LogisticRegression(C=100.0, max_iter=3000).fit(basis.fit_transform(a), y)
    compact = compile_standardized_rbf(scaler, basis, head)
    probs["radial64"] = compact.predict_positive(z)
    native = head.predict_proba(basis.transform(b))[:, 1]
    max_abs = float(np.abs(native - probs["radial64"]).max())
    if not np.isfinite(max_abs):
        raise ValueError(
            f"compact radial network and source predictor disagree: {max_abs}"
        )
    # Use explicit inner splits for sigmoid calibration to keep duplicates together.
    svc = make_pipeline(StandardScaler(), SVC(C=1.0, gamma="scale"))
    calibrated = CalibratedClassifierCV(
        svc, method="sigmoid", cv=splits(y, groups, 3, 13601 + seed), ensemble=False
    )
    calibrated.fit(x, y)
    probs["rbf_svc"] = calibrated.predict_proba(z)[:, 1]
    forest = RandomForestClassifier(n_estimators=150, n_jobs=1, random_state=seed).fit(
        x, y
    )
    probs["forest"] = forest.predict_proba(z)[:, 1]
    probs["ensemble"] = (probs["logit"] + probs["rbf_svc"] + probs["forest"]) / 3
    fitted_svc = calibrated.calibrated_classifiers_[0].estimator.named_steps["svc"]
    costs = {
        "landmarks": len(compact.centers),
        "svc_support": len(fitted_svc.support_),
        "forest_nodes": sum(tree.tree_.node_count for tree in forest.estimators_),
        "compile_max_abs": max_abs,
        "fit_predict_seconds": time.perf_counter() - start,
    }
    return probs, costs


def run(out, data):
    start = datetime.now(UTC)
    timer = time.perf_counter()
    out.mkdir(parents=True, exist_ok=True)
    predictions, selections, fold_scores = [], [], []
    for name, x, y in populations(data):
        groups = np.unique(x, axis=0, return_inverse=True)[1].reshape(-1)
        for fold, (tr, te) in enumerate(splits(y, groups, 5, 11801)):
            base_seed = 20500 + 100 * fold
            inner = {m: np.empty(len(tr)) for m in ["logit", "radial64", "ensemble"]}
            inner_error = 0.0
            for j, (sub, validation) in enumerate(
                splits(y[tr], groups[tr], 3, 11802 + fold)
            ):
                pr, inner_cost = models(
                    x[tr[sub]],
                    y[tr[sub]],
                    groups[tr[sub]],
                    x[tr[validation]],
                    base_seed + j,
                )
                inner_error = max(inner_error, inner_cost["compile_max_abs"])
                for m, values in inner.items():
                    values[validation] = pr[m]
            inner_loss = {m: np.mean((p - y[tr]) ** 2) for m, p in inner.items()}
            winner = min(["logit", "radial64"], key=inner_loss.get)
            chosen = (
                winner
                if inner_loss[winner] <= inner_loss["ensemble"] + 0.005
                else "ensemble"
            )
            p, cost = models(x[tr], y[tr], groups[tr], x[te], base_seed + 10)
            p["selected_compact"] = p[winner]
            p["inner_policy"] = p[chosen]
            outer_loss = {m: np.mean((v - y[te]) ** 2) for m, v in p.items()}
            selections.append(
                dict(
                    dataset=name,
                    fold=fold,
                    train_rows=len(tr),
                    test_rows=len(te),
                    train_groups=np.unique(groups[tr]).size,
                    test_groups=np.unique(groups[te]).size,
                    inner_compile_max_abs=inner_error,
                    inner_winner=winner,
                    policy=chosen,
                    inner_compact=inner_loss[winner],
                    inner_ensemble=inner_loss["ensemble"],
                    outer_compact=outer_loss[winner],
                    outer_ensemble=outer_loss["ensemble"],
                    inner_accept=chosen != "ensemble",
                    outer_within=outer_loss[winner] <= outer_loss["ensemble"] + 0.005,
                    **cost,
                )
            )
            rows = dict(dataset=name, fold=fold, row=te, group=groups[te], y=y[te], **p)
            predictions.append(pd.DataFrame(rows))
            for m, v in p.items():
                fold_scores.append(
                    {
                        "dataset": name,
                        "fold": fold,
                        "method": m,
                        "rows": len(te),
                        "brier": outer_loss[m],
                        "accuracy": np.mean((v >= 0.5) == y[te]),
                    }
                )
    pred = pd.concat(predictions, ignore_index=True)
    assert not pred.duplicated(["dataset", "row"]).any()
    assert pred.groupby(["dataset", "group"]).fold.nunique().max() == 1
    pred.to_csv(
        out / "predictions.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    pd.DataFrame(selections).to_csv(out / "selection.csv", index=False)
    scores = pd.DataFrame(fold_scores)
    scores.to_csv(out / "folds.csv", index=False)
    means = []
    for (name, m), g in scores.groupby(["dataset", "method"]):
        means.append(
            {
                "dataset": name,
                "method": m,
                "brier": np.average(g.brier, weights=g.rows),
                "accuracy": np.average(g.accuracy, weights=g.rows),
            }
        )
    pd.DataFrame(means).to_csv(out / "means.csv", index=False)
    config = {
        "outer_folds": 5,
        "inner_folds": 3,
        "svc_calibration_folds": 3,
        "outer_seed": 11801,
        "inner_seed": 11802,
        "deduplicate_split": True,
        "acceptance_margin": 0.005,
        "landmarks": 64,
        "readout_C": 100.0,
        "gamma": "1/d",
        "compute_start_utc": start.isoformat(),
        "compute_seconds": time.perf_counter() - timer,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.output, args.data_dir)

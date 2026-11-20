"""External validity of supervised majority networks on genuine binary labels."""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch import nn

from examples.real_transfer import HAR_SHA, populations
from fusionbench.eval.lattice_compile import train
from fusionbench.eval.majority_moments import orthogonal
from fusionbench.eval.majority_routing import MajorityNetwork
from fusionbench.methods.affine_preprocessing import fold_standardized_pca_affines
from fusionbench.methods.majority import independent_majority_logit
from fusionbench.methods.majority_moments import fit_majority_moments


def datasets(archive):
    for name, loader in [
        ("wisconsin", load_breast_cancer),
        ("digits_parity", load_digits),
    ]:
        x, y = loader(return_X_y=True)
        if name == "digits_parity":
            y = y % 2
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=9921)
        for fold, (tr, te) in enumerate(folds.split(x, y)):
            yield name, fold, x[tr], y[tr], x[te], y[te], te, np.full(len(te), fold)
    for name, x, y, groups, splits in populations(archive):
        if name != "har":
            continue
        tr = np.concatenate(splits[:3])
        te = splits[3]
        tr = tr[np.isin(y[tr], [1, 2])]
        te = te[np.isin(y[te], [1, 2])]
        assert not np.intersect1d(groups[tr], groups[te]).size
        yield (
            "har_stairs",
            0,
            x[tr].astype(float),
            y[tr] - 1,
            x[te].astype(float),
            y[te] - 1,
            te,
            groups[te],
        )


def estimators(seed):
    return {
        "logit": LogisticRegression(C=1.0, max_iter=3000),
        "rbf": SVC(C=1.0, gamma="scale", probability=True, random_state=seed),
    }


def main_split(seed, raw, y, test):
    rng = np.random.default_rng(seed)
    scaler = StandardScaler()
    x = scaler.fit_transform(raw)
    z = scaler.transform(test)
    pca = PCA(n_components=9, whiten=True, svd_solver="full")
    reduced = pca.fit_transform(x)
    query = pca.transform(z)
    preds = {}
    for suffix, x_tr, x_te in [("pca", reduced, query), ("full", x, z)]:
        for name, model in estimators(seed).items():
            preds[f"{name}_{suffix}"] = model.fit(x_tr, y).predict_proba(x_te)[:, 1]
    forest = RandomForestClassifier(n_estimators=150, random_state=seed, n_jobs=1)
    preds["forest_full"] = forest.fit(x, y).predict_proba(z)[:, 1]
    preds["ensemble_full"] = np.mean(
        [preds[name] for name in ("logit_full", "rbf_full", "forest_full")], axis=0
    )
    axes = fit_majority_moments(reduced, y).axes
    preds["direct_moment"] = expit(5 * np.median(query @ axes.T, axis=1))
    tensor = torch.as_tensor(reduced.copy(), dtype=torch.float64)
    labels = torch.as_tensor(y, dtype=torch.float64)
    schedule = rng.integers(0, len(y), size=(600, 64))
    error = 0.0
    for name in ["moment_net", "random_net", "relu_net"]:
        torch.manual_seed(seed)
        if name == "relu_net":
            net = nn.Sequential(nn.Linear(9, 3), nn.ReLU(), nn.Linear(3, 1)).double()
        else:
            net = MajorityNetwork(independent_majority_logit)
            init = axes if name == "moment_net" else orthogonal(rng)
            with torch.no_grad():
                net.leaves.weight.copy_(torch.from_numpy(init))
                net.leaves.bias.zero_()
        train(net, tensor, labels, schedule)
        with torch.no_grad():
            if name == "relu_net":
                preds[name] = net(torch.from_numpy(query)).reshape(-1).sigmoid().numpy()
            else:
                w, b = net.leaves.weight.numpy(), net.leaves.bias.numpy()
                original = expit(np.median(query @ w.T + b, axis=1))
                w_raw, b_raw = fold_standardized_pca_affines(scaler, pca, w, b)
                preds[name] = expit(np.median(test @ w_raw.T + b_raw, axis=1))
                error = max(error, np.abs(original - preds[name]).max())
    return preds, error, pca.explained_variance_ratio_.sum()


def run(out, archive):
    torch.set_num_threads(1)
    start = datetime.now(UTC)
    timer = time.perf_counter()
    out.mkdir(exist_ok=True, parents=True)
    predictions, details = [], []
    for idx, (name, fold, x, y, z, target, ids, groups) in enumerate(datasets(archive)):
        preds, error, var = main_split(7100 + idx, x, y, z)
        np.testing.assert_array_less(error, 1e-9)
        predictions.append(
            pd.DataFrame(
                dict(dataset=name, fold=fold, group=groups, row=ids, y=target, **preds)
            )
        )
        details.append(
            {
                "dataset": name,
                "fold": fold,
                "train_rows": len(y),
                "test_rows": len(target),
                "raw_features": x.shape[1],
                "test_groups": len(np.unique(groups)),
                "pca_variance_fraction": var,
                "raw_fold_max_error": error,
                "raw_majority_parameters": 3 * (x.shape[1] + 1),
            }
        )
    frame = pd.concat(predictions, ignore_index=True)
    assert not frame.duplicated(["dataset", "row"]).any()
    frame.to_csv(
        out / "predictions.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    pd.DataFrame(details).to_csv(out / "splits.csv", index=False)
    rows = []
    methods = frame.columns.difference(["dataset", "fold", "group", "row", "y"])
    for (name, group), c in frame.groupby(["dataset", "group"]):
        for method in methods:
            prob = c[method].values
            rows.append(
                {
                    "dataset": name,
                    "group": group,
                    "method": method,
                    "rows": len(c),
                    "accuracy": np.mean((prob >= 0.5) == c.y),
                    "brier": np.mean((prob - c.y) ** 2),
                }
            )
    grouped = pd.DataFrame(rows)
    grouped.to_csv(out / "units.csv", index=False)
    summary, paired = [], []
    for (name, method), g in grouped.groupby(["dataset", "method"]):
        weights = None if name == "har_stairs" else g.rows
        summary.append(
            {
                "dataset": name,
                "method": method,
                "accuracy": np.average(g.accuracy, weights=weights),
                "brier": np.average(g.brier, weights=weights),
            }
        )
    for name, g in grouped.groupby("dataset"):
        width = g.pivot(index="group", columns="method", values="brier")
        for baseline in [
            "random_net",
            "ensemble_full",
            "rbf_pca",
            "logit_pca",
            "relu_net",
        ]:
            diff = width.moment_net - width[baseline]
            weight = (
                None
                if name == "har_stairs"
                else g[g.method == "moment_net"]
                .set_index("group")
                .loc[diff.index, "rows"]
            )
            paired.append(
                {
                    "dataset": name,
                    "reference": baseline,
                    "brier_difference": np.average(diff, weights=weight),
                    "units": len(diff),
                    "units_favor_moments": (diff < 0).sum(),
                    "min_unit": diff.min(),
                    "max_unit": diff.max(),
                }
            )
    pd.DataFrame(summary).to_csv(out / "means.csv", index=False)
    pd.DataFrame(paired).to_csv(out / "paired.csv", index=False)
    config = {
        "cv_seed": 9921,
        "estimator_seeds": "7100 + split order",
        "components": 9,
        "trees": 150,
        "adam_steps": 600,
        "batch_size": 64,
        "learning_rate": 0.015,
        "har_sha256": HAR_SHA,
        "compute_start_utc": start.isoformat(),
        "compute_seconds": time.perf_counter() - timer,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--har-archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.output, args.har_archive)

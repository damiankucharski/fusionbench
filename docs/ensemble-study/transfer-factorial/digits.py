"""Nested fusion-training samples from eight stratified splits of sklearn digits."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from gate import gate_choices
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from examples.digits_predictions import write_digits_predictions
from fusionbench.methods.averaging import SimpleAverage, TopN
from fusionbench.methods.native_pattern import RegularizedNativePatternOKAPI
from fusionbench.methods.stacking import StackingLR
from fusionbench.store import load_dataset

HERE = Path(__file__).resolve().parent
SEEDS = range(256100, 256108)


def source_partitions(seed, labels, ds):
    # Recover global source IDs to check the public builder and archive held-out row losses.
    ids = np.arange(len(labels))
    base, remainder = train_test_split(
        ids, stratify=labels, train_size=0.5, random_state=seed
    )
    fit, test = train_test_split(
        remainder, stratify=labels[remainder], test_size=0.5, random_state=seed + 1
    )
    assert (len(base), len(fit), len(test)) == (898, 449, 450)
    np.testing.assert_array_equal(np.sort(np.concatenate((base, fit, test))), ids)
    np.testing.assert_array_equal(labels[fit], ds.fit_y.ravel())
    np.testing.assert_array_equal(labels[test], ds.test_y.ravel())
    small, _ = train_test_split(
        np.arange(len(fit)), train_size=100, stratify=labels[fit], random_state=seed + 2
    )
    return base, fit, test, small


def scores_for_bank(x, y, test_x, *, source_ids, test_ids, seed):
    models = {
        "regularized": RegularizedNativePatternOKAPI(seed=0, split_seed=0),
        "mean": SimpleAverage(),
        "top_n": TopN(metric="brier"),
        "prob_lr": StackingLR(c_value=1.0),
    }
    for model in models.values():
        model.fit(x, y, task="multiclass", n_classes=10)
    gate = gate_choices(x, y)
    forecasts = {
        m: (model.predict(x), model.predict(test_x)) for m, model in models.items()
    }
    decisions = {m: m for m in models}
    for rule in ("plain", "guard"):
        decisions[rule] = "regularized" if gate[rule] else "mean"
        forecasts[rule] = forecasts[decisions[rule]]
    audit = {
        "seed": seed,
        "n": len(x),
        "family": models["regularized"].chosen_,
        "n_top": models["top_n"].n_selected_,
        "native_calls": models["regularized"].counts_["total_score_calls"],
        "gain": gate["gain"],
        "se": gate["se"],
        "plain": decisions["plain"],
        "guard": decisions["guard"],
        "fit_ids_sha256": fingerprint(source_ids),
        "test_ids_sha256": fingerprint(test_ids),
        "provider_fit_sha256": fingerprint(x),
        "provider_test_sha256": fingerprint(test_x),
    }
    return forecasts, decisions, audit


def fingerprint(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def run(directory):
    directory.mkdir(parents=True, exist_ok=True)
    _, labels = load_digits(return_X_y=True)
    risks, audit, row_tables = [], [], []
    with threadpool_limits(limits=1):
        for seed in SEEDS:
            with TemporaryDirectory() as tmp:
                ds = load_dataset(write_digits_predictions(tmp, seed=seed))
            base_ids, fit_ids, test_ids, small = source_partitions(seed, labels, ds)
            for subset in (small, np.arange(len(fit_ids))):
                active_ids = fit_ids[subset]
                x, y = ds.fit_preds[subset], ds.fit_y[subset].ravel()
                forecasts, decisions, details = scores_for_bank(
                    x,
                    y,
                    ds.test_preds,
                    source_ids=active_ids,
                    test_ids=test_ids,
                    seed=seed,
                )
                details["base_ids_sha256"] = fingerprint(base_ids)
                audit.append(details)
                for method, (p_fit, p_test) in forecasts.items():
                    row = {
                        "seed": seed,
                        "n": len(y),
                        "method": method,
                        "deployed": decisions[method],
                    }
                    for split, p, ids in (
                        ("fit", p_fit, active_ids),
                        ("test", p_test, test_ids),
                    ):
                        np.testing.assert_allclose(
                            p.sum(axis=1), 1.0, rtol=0, atol=1e-12
                        )
                        assert p.shape == (len(ids), 10) and np.all((p >= 0) & (p <= 1))
                        losses = ((p - np.eye(10)[labels[ids]]) ** 2).sum(axis=1)
                        row[f"{split}_brier"] = losses.mean()
                        if method not in ("plain", "guard"):
                            row_tables.append(
                                pd.DataFrame(
                                    {
                                        "seed": seed,
                                        "n": len(y),
                                        "method": method,
                                        "split": split,
                                        "digit_id": ids,
                                        "label": labels[ids],
                                        "brier": losses,
                                    }
                                )
                            )
                    row["gap"] = row["test_brier"] - row["fit_brier"]
                    risks.append(row)
    frame = pd.DataFrame(risks)
    frame.to_csv(directory / "digits-scores.csv", index=False)
    pd.DataFrame(audit).to_csv(directory / "digits-choices.csv", index=False)
    pd.concat(row_tables).to_csv(
        directory / "digits-row-losses.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    grouped = frame.groupby(["n", "method"])
    summary = grouped[["fit_brier", "test_brier", "gap"]].mean().reset_index()
    summary.to_csv(directory / "digits-summary.csv", index=False)
    changes = []
    for method, block in frame.groupby("method"):
        pivot = block.pivot(index="seed", columns="n", values="test_brier")
        for seed, value in (pivot[449] - pivot[100]).items():
            changes.append(
                {
                    "seed": seed,
                    "method": method,
                    "contrast": "449_minus_100",
                    "difference": value,
                }
            )
    for (n, seed), block in frame.groupby(["n", "seed"]):
        losses = block.set_index("method").test_brier
        for other in ("mean", "top_n", "prob_lr", "guard"):
            changes.append(
                {
                    "seed": seed,
                    "n": n,
                    "method": "regularized",
                    "contrast": f"minus_{other}",
                    "difference": losses.regularized - losses[other],
                }
            )
    pd.DataFrame(changes).to_csv(directory / "digits-paired.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "results")
    run(parser.parse_args().out)

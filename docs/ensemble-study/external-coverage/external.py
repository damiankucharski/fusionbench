"""Fit-only profiles and held-out external coverage on sklearn digits and wine."""

import argparse
import json
from dataclasses import dataclass
from hashlib import sha256
from itertools import combinations
from pathlib import Path
from runpy import run_path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import pandas as pd
import sklearn
from common import evaluate_methods, profile_fit
from sklearn.datasets import load_digits, load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from fusionbench import Dataset, load_dataset
from fusionbench.eval.transfer_grid import brier

BUILDER = Path(__file__).resolve().parents[3] / "examples" / "digits_predictions.py"
SUITES = {"digits": 3_140_000, "digits_lr_bag": 3_140_000, "wine": 3_150_000}
HETEROGENEOUS = ("lr", "neighbours", "forest")
BAG = ("lr_bootstrap_0", "lr_bootstrap_1", "lr_bootstrap_2")


@dataclass
class Bank:
    suite: str
    seed: int
    ds: Dataset
    base: np.ndarray
    fit: np.ndarray
    test: np.ndarray
    bootstraps: list
    build_seconds: float

    @property
    def key(self):
        return f"{self.suite}_{self.seed}"


def digest(array):
    return sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def partition(labels, seed):
    indices = np.arange(len(labels), dtype="<i8")
    base, rest = train_test_split(
        indices, stratify=labels, train_size=0.5, random_state=seed
    )
    fit, test = train_test_split(
        rest, stratify=labels[rest], test_size=0.5, random_state=seed + 1
    )
    np.testing.assert_array_equal(np.sort(np.concatenate([base, fit, test])), indices)
    return base, fit, test


def bootstrap(base, labels, seed, provider):
    rng = np.random.default_rng([seed, 6001, provider])
    samples = []
    for cls in np.unique(labels):
        rows = base[labels[base] == cls]
        samples.append(rng.choice(rows, size=len(rows), replace=True))
    chosen = rng.permutation(np.concatenate(samples))
    np.testing.assert_array_equal(
        np.bincount(labels[chosen]), np.bincount(labels[base])
    )
    assert np.isin(chosen, base).all()
    return chosen


def fitted_dataset(suite, raw, labels, base, fit, test, seed):
    if suite == "digits_lr_bag":
        features = raw / 16.0
        names = BAG
        experts = [LogisticRegression(max_iter=2000) for _ in names]
        training_sets = [bootstrap(base, labels, seed, i) for i in range(len(names))]
        boots = [digest(rows) for rows in training_sets]
    else:
        features = StandardScaler().fit(raw[base]).transform(raw)
        names = HETEROGENEOUS
        experts = [
            LogisticRegression(max_iter=2000),
            KNeighborsClassifier(n_neighbors=5),
            RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1),
        ]
        training_sets = [base] * len(names)
        boots = []
    classes = np.unique(labels)
    for model, train in zip(experts, training_sets, strict=True):
        model.fit(features[train], labels[train])
        np.testing.assert_array_equal(model.classes_, classes)

    def predict(rows):
        return np.stack(
            [m.predict_proba(features[rows]) for m in experts], axis=1
        ).astype(np.float32)

    return Dataset(
        name=suite,
        task="multiclass",
        n_classes=len(classes),
        model_ids=list(names),
        fit_preds=predict(fit),
        fit_y=labels[fit],
        test_preds=predict(test),
        test_y=labels[test],
        fit_oracle=None,
        test_oracle=None,
        meta={"primary_metric": "brier"},
    ), boots


def build():
    builder = run_path(str(BUILDER))["write_digits_predictions"]
    digits, wine = load_digits(return_X_y=True), load_wine(return_X_y=True)
    banks = []
    for suite, root in SUITES.items():
        raw, labels = wine if suite == "wine" else digits
        for rep in range(8):
            seed = root + rep
            base, fit, test = partition(labels, seed)
            tick = perf_counter()
            if suite == "digits":
                with TemporaryDirectory() as temp:
                    ds = load_dataset(builder(temp, seed))
                assert tuple(ds.model_ids) == HETEROGENEOUS and ds.meta[
                    "base_train_n"
                ] == len(base)
                boots = []
            else:
                ds, boots = fitted_dataset(suite, raw, labels, base, fit, test, seed)
            np.testing.assert_array_equal(ds.fit_y, labels[fit])
            # The pairing is by original row ID. No outcome from ds.test_y
            # is scored or summarized by the fit-only phase.
            banks.append(
                Bank(suite, seed, ds, base, fit, test, boots, perf_counter() - tick)
            )
    return banks


def profile_tables(banks):
    profiles, providers, records, tensors = [], [], [], {}
    for bank in banks:
        ds = bank.ds
        key = {"suite": bank.suite, "seed": bank.seed}
        x, y = ds.fit_preds, ds.fit_y
        profiles.append(
            {
                **key,
                "n_base": len(bank.base),
                "n_fit": len(y),
                "n_test": len(bank.test),
                "n_classes": ds.n_classes,
                "n_providers": len(ds.model_ids),
                **profile_fit(x, y),
            }
        )
        for i, name in enumerate(ds.model_ids):
            p = x[:, i, :]
            providers.append(
                {
                    **key,
                    "provider": name,
                    "fit_brier": brier(p, y),
                    "zero_fraction": float(np.mean(p == 0)),
                    "mean_max_probability": float(p.max(1).mean(dtype=np.float64)),
                }
            )
        tensors[bank.key + "_inputs"] = x
        tensors[bank.key + "_labels"] = y
        tensors[bank.key + "_rows"] = bank.fit
        records.append(
            {
                **key,
                "providers": ds.model_ids,
                "n_classes": ds.n_classes,
                "base_rows": len(bank.base),
                "fit_rows": len(y),
                "test_rows": len(bank.test),
                "base_rows_sha256": digest(bank.base),
                "fit_rows_sha256": digest(bank.fit),
                "test_rows_sha256": digest(bank.test),
                "fit_inputs_sha256": digest(x),
                "fit_labels_sha256": digest(y),
                "bootstrap_rows_sha256": bank.bootstraps,
            }
        )
    return pd.DataFrame(profiles), pd.DataFrame(providers), records, tensors


def write_profiles(out, banks):
    profiles, providers, records, tensors = profile_tables(banks)
    manifest = {
        "schema_version": 1,
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "digits_builder_sha256": sha256(BUILDER.read_bytes()).hexdigest(),
        "banks": records,
    }
    out.mkdir(parents=True, exist_ok=True)
    file = out / "fit_manifest.json"
    if file.exists() and json.loads(file.read_text()) != manifest:
        raise ValueError(
            "Generated fit tensors or partitions differ from the frozen profile stage"
        )
    for name, frame in (("fit_profiles", profiles), ("fit_providers", providers)):
        path = out / (name + ".csv")
        value = frame.to_csv(index=False)
        if path.exists() and path.read_text() != value:
            raise ValueError(f"{name} changed from the recorded profile stage")
        path.write_text(value)
    if not (out / "fit_banks.npz").exists():
        np.savez_compressed(out / "fit_banks.npz", **tensors)
    file.write_text(json.dumps(manifest, indent=2) + "\n")


def evaluate(out, banks):
    scores, tensors = [], {}
    tick = perf_counter()
    for bank in banks:
        ds = bank.ds
        methods = evaluate_methods(ds, seed=bank.seed)
        for row in methods:
            scores.append({"suite": bank.suite, "seed": bank.seed, **row})
        tensors[bank.key + "_inputs"] = ds.test_preds
        tensors[bank.key + "_labels"] = ds.test_y
        tensors[bank.key + "_rows"] = bank.test
    frame = pd.DataFrame(scores)
    ref = frame[frame.method == "mean"].set_index(["suite", "seed"]).brier
    frame = frame.join(ref.rename("mean_brier"), on=["suite", "seed"])
    frame["gain_over_mean"] = frame.mean_brier - frame.brier
    summary = frame.groupby(["suite", "method"])[
        ["brier", "accuracy", "gain_over_mean", "fit_seconds"]
    ].mean()
    frame.to_csv(out / "scores.csv", index=False)
    summary.to_csv(out / "means.csv")
    np.savez_compressed(out / "test_banks.npz", **tensors)
    overlap = {}
    for suite in SUITES:
        rows = [bank.test for bank in banks if bank.suite == suite]
        amounts = [len(np.intersect1d(a, b)) for a, b in combinations(rows, 2)]
        overlap[suite] = {
            "n_unique": len(np.unique(rows)),
            "mean_pair_overlap": np.mean(amounts),
            "min_pair_overlap": min(amounts),
            "max_pair_overlap": max(amounts),
        }
    meta = {
        "test_overlap": overlap,
        "base_build_seconds": sum(b.build_seconds for b in banks),
        "fusion_and_scoring_seconds": perf_counter() - tick,
        "fit_counts": len(scores),
    }
    (out / "run.json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--profiles-only",
        action="store_true",
        help="Only write development profiles, fit tensors and split provenance",
    )
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        data = build()
        write_profiles(args.out, data)
        if not args.profiles_only:
            evaluate(args.out, data)

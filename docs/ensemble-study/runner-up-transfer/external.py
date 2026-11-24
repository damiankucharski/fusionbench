"""Build or evaluate three source-disjoint real multiclass provider banks."""

import argparse
import json
import sys
from hashlib import sha256
from io import BytesIO
from itertools import combinations
from pathlib import Path
from tempfile import gettempdir
from time import perf_counter
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import load_digits, load_iris
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from fusionbench import Dataset
from fusionbench.eval.transfer_grid import brier

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "external-coverage"))
from common import evaluate_methods, profile_fit

SEEDS_ZIP = "https://archive.ics.uci.edu/static/public/236/seeds.zip"
ZIP_SHA256 = "5033bd71b24592fa70d9b3cc5be02332dc6d03d0b27e6655e6d553f39afab323"
DATA_SHA256 = "1f3f83c0d8485ae9148061389d19628607e3f5660e3d6f40ec9102fb398bb12f"
SUITES = {
    "digits": (3300000, 898, 449),
    "iris": (3310000, 74, 44),
    "seeds": (3320000, 104, 44),
}


def digest(array):
    return sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def load_seeds(path):
    """Read the UCI archive from a verified local cache or its official URL."""
    if path.exists():
        archive = path.read_bytes()
    else:
        with urlopen(SEEDS_ZIP, timeout=45) as response:
            archive = response.read()
    if sha256(archive).hexdigest() != ZIP_SHA256:
        raise ValueError("UCI seeds archive checksum mismatch")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(archive)
    with ZipFile(BytesIO(archive)) as handle:
        data = handle.read("seeds_dataset.txt")
    if sha256(data).hexdigest() != DATA_SHA256:
        raise ValueError("UCI seeds data checksum mismatch")
    rows = np.loadtxt(BytesIO(data))
    labels = rows[:, -1].astype(np.int64) - 1
    if rows.shape != (210, 8) or not np.array_equal(np.bincount(labels), [70, 70, 70]):
        raise ValueError("Unexpected seeds dataset dimensions or class encoding")
    return rows[:, :-1], labels


def partition(labels, n_base, n_fit, seed):
    rows = np.arange(len(labels), dtype=np.int64)
    base, rest = train_test_split(
        rows, train_size=n_base, stratify=labels, random_state=seed
    )
    fit, test = train_test_split(
        rest, train_size=n_fit, stratify=labels[rest], random_state=seed + 1
    )
    np.testing.assert_array_equal(np.sort(np.concatenate([base, fit, test])), rows)
    return base, fit, test


def providers(suite, seed):
    if suite == "digits":
        return {
            "gaussian_nb": GaussianNB(var_smoothing=0.01),
            "distance_knn11": KNeighborsClassifier(n_neighbors=11, weights="distance"),
            "extra_trees": ExtraTreesClassifier(
                n_estimators=100, random_state=seed, n_jobs=1
            ),
        }
    return {
        "lr": LogisticRegression(max_iter=2000),
        "neighbours": KNeighborsClassifier(n_neighbors=5),
        "forest": RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1),
    }


def make_bank(suite, seed, raw, labels, n_base, n_fit, *, score):
    base, fit, test = partition(labels, n_base, n_fit, seed)
    x = (
        raw / 16.0
        if suite == "digits"
        else StandardScaler().fit(raw[base]).transform(raw)
    )
    models = providers(suite, seed)
    for model in models.values():
        model.fit(x[base], labels[base])
        np.testing.assert_array_equal(model.classes_, np.unique(labels))

    def predictions(rows):
        return np.stack(
            [model.predict_proba(x[rows]) for model in models.values()], axis=1
        ).astype(np.float32)

    p, y = predictions(fit), labels[fit]
    info = {"suite": suite, "seed": seed}
    key = f"{suite}_{seed}"
    cells = {
        "n_base": len(base),
        "n_fit": len(fit),
        "n_test": len(test),
        "n_classes": p.shape[-1],
        "n_providers": len(models),
    }
    metadata = {
        **info,
        **cells,
        "providers": list(models),
        "base_rows_sha256": digest(base),
        "fit_rows_sha256": digest(fit),
        "test_rows_sha256": digest(test),
        "fit_inputs_sha256": digest(p),
        "fit_labels_sha256": digest(y),
        "source_features_sha256": digest(raw),
        "source_labels_sha256": digest(labels),
    }
    provider_profiles = [
        {
            **info,
            "provider": name,
            "fit_brier": brier(p[:, i], y),
            "zero_fraction": float(np.mean(p[:, i] == 0)),
            "mean_max_probability": float(p[:, i].max(1).mean(dtype=np.float64)),
        }
        for i, name in enumerate(models)
    ]
    stats = {**info, **cells, **profile_fit(p, y)}
    fit_arrays = {key + "_inputs": p, key + "_labels": y, key + "_rows": fit}
    ds = None
    if score:
        ds = Dataset(
            name=suite,
            task="multiclass",
            n_classes=p.shape[-1],
            model_ids=list(models),
            fit_preds=p,
            fit_y=y,
            test_preds=predictions(test),
            test_y=labels[test],
            fit_oracle=None,
            test_oracle=None,
            meta={"primary_metric": "brier"},
        )
    return stats, provider_profiles, metadata, fit_arrays, (suite, seed, ds, test)


def frozen_files(path, files, arrays, *, score):
    path.mkdir(parents=True, exist_ok=True)
    for name, contents in files.items():
        file = path / name
        if file.exists():
            if file.read_text() != contents:
                raise ValueError(f"Frozen artifact differs: {file}")
        elif score:
            raise FileNotFoundError(
                f"Run --profiles-only before scoring: missing {file}"
            )
        else:
            file.write_text(contents)
    tensor = path / "fit_banks.npz"
    if tensor.exists():
        with np.load(tensor) as saved:
            if set(saved.files) != set(arrays):
                raise ValueError(f"Unexpected archived keys in {tensor}")
            for key, value in arrays.items():
                np.testing.assert_array_equal(saved[key], value)
    elif score:
        raise FileNotFoundError(f"Run --profiles-only before scoring: missing {tensor}")
    else:
        np.savez_compressed(tensor, **arrays)


def write_scores(out, banks, build_seconds):
    out.mkdir(parents=True, exist_ok=True)
    scores, arrays, overlap = [], {}, {}
    start = perf_counter()
    for suite, seed, ds, test in banks:
        scores.extend(
            {"suite": suite, "seed": seed, **row}
            for row in evaluate_methods(ds, seed=seed)
        )
        key = f"{suite}_{seed}"
        arrays.update(
            {
                key + "_inputs": ds.test_preds,
                key + "_labels": ds.test_y,
                key + "_rows": test,
            }
        )
    for suite in SUITES:
        rows = [rows for name, _, _, rows in banks if name == suite]
        sizes = [len(np.intersect1d(a, b)) for a, b in combinations(rows, 2)]
        overlap[suite] = {
            "n_unique": len(np.unique(rows)),
            "mean_pair_overlap": float(np.mean(sizes)),
            "min_pair_overlap": min(sizes),
            "max_pair_overlap": max(sizes),
        }
    frame = pd.DataFrame(scores)
    mean = (
        frame[frame.method == "mean"]
        .set_index(["suite", "seed"])
        .brier.rename("mean_brier")
    )
    frame = frame.join(mean, on=["suite", "seed"])
    frame["gain_over_mean"] = frame.mean_brier - frame.brier
    frame.to_csv(out / "scores.csv", index=False)
    frame.groupby(["suite", "method"])[
        ["brier", "accuracy", "gain_over_mean", "fit_seconds"]
    ].mean().to_csv(out / "means.csv")
    np.savez_compressed(out / "test_banks.npz", **arrays)
    stats = {
        "fit_counts": len(frame),
        "test_overlap": overlap,
        "base_and_profile_seconds": build_seconds,
        "fusion_and_scoring_seconds": perf_counter() - start,
    }
    (out / "run.json").write_text(json.dumps(stats, indent=2) + "\n")


def run(study, archive_path, *, score):
    raw = {
        "digits": load_digits(return_X_y=True),
        "iris": load_iris(return_X_y=True),
        "seeds": load_seeds(archive_path),
    }
    profiles, provider_profiles, records, banks, arrays = [], [], [], [], {}
    tick = perf_counter()
    for suite, (root, n_base, n_fit) in SUITES.items():
        features, labels = raw[suite]
        for seed in range(root, root + 8):
            stats, individual, metadata, tensors, bank = make_bank(
                suite, seed, features, labels, n_base, n_fit, score=score
            )
            profiles.append(stats)
            provider_profiles.extend(individual)
            records.append(metadata)
            arrays.update(tensors)
            banks.append(bank)
    manifest = {
        "schema_version": 1,
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "seeds_source": {
            "citation": "Charytanowicz, M., Niewczas, J., Kulczycki, P., Kowalski, P., & Lukasik, S. (2010). Seeds. UCI Machine Learning Repository.",
            "doi": "10.24432/C5H30K",
            "license": "CC BY 4.0",
            "archive_url": SEEDS_ZIP,
            "archive_sha256": ZIP_SHA256,
            "seeds_dataset_txt_sha256": DATA_SHA256,
            "transforms": "Numerical features retained in published order, cultivar labels 1..3 recoded to 0..2.",
        },
        "banks": records,
    }
    files = {
        "fit_profiles.csv": pd.DataFrame(profiles).to_csv(index=False),
        "fit_providers.csv": pd.DataFrame(provider_profiles).to_csv(index=False),
        "fit_manifest.json": json.dumps(manifest, indent=2) + "\n",
    }
    frozen_files(study / "external-fit", files, arrays, score=score)
    if score:
        write_scores(study / "external-results", banks, perf_counter() - tick)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--profiles-only",
        action="store_true",
        help="Generate and publish fusion-fit artifacts only",
    )
    mode.add_argument(
        "--score",
        action="store_true",
        help="Recheck frozen fit artifacts and score separate held-out banks",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=HERE,
        help="Study root containing external-fit and external-results",
    )
    parser.add_argument(
        "--seeds-archive",
        type=Path,
        default=Path(gettempdir()) / "fb-seeds-uci-236.zip",
    )
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.out, args.seeds_archive, score=args.score)

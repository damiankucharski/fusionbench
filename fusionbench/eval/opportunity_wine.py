"""Cross-fitted Wine recognition probabilities for external ensemble evaluation.

A specimen occurs in exactly one outer test fold. Within each fold its validation
specimens and test specimens are withheld from every provider fit, including
feature selection. The same six final providers predict both held-out splits.
"""

import argparse
import json
from dataclasses import dataclass
from hashlib import sha256
from numbers import Integral
from pathlib import Path

import numpy as np
import sklearn
from sklearn.datasets import load_wine
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

PROVIDER_NAMES = (
    "logistic",
    "gaussian_nb",
    "knn",
    "tree",
    "extra_trees",
    "selected_lda",
)


@dataclass(frozen=True)
class ProviderJob:
    train_ids: np.ndarray
    predict_ids: np.ndarray


@dataclass(frozen=True)
class WinePlan:
    fold: int
    fit_ids: np.ndarray
    validation_ids: np.ndarray
    test_ids: np.ndarray
    inner_holdout: np.ndarray

    def provider_jobs(self):
        for inner in range(4):
            held = self.inner_holdout[self.fit_ids] == inner
            yield ProviderJob(self.fit_ids[~held], self.fit_ids[held])
        yield ProviderJob(
            self.fit_ids, np.concatenate((self.validation_ids, self.test_ids))
        )


@dataclass(frozen=True)
class WineSplit:
    row_ids: np.ndarray
    preds: np.ndarray
    y: np.ndarray


@dataclass(frozen=True)
class WineFold:
    plan: WinePlan
    fit: WineSplit
    validation: WineSplit
    test: WineSplit


def _plans(labels):
    row_ids = np.arange(len(labels))
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=7709)
    for fold, (development, test_ids) in enumerate(outer.split(row_ids, labels)):
        fit_ids, validation_ids = train_test_split(
            development,
            test_size=0.25,
            stratify=labels[development],
            random_state=7909 + fold,
        )
        fit_ids = np.sort(fit_ids)
        inner = StratifiedKFold(n_splits=4, shuffle=True, random_state=7809 + fold)
        inner_holdout = np.full(len(labels), -1, dtype=np.int16)
        for index, (_, held) in enumerate(inner.split(fit_ids, labels[fit_ids])):
            inner_holdout[fit_ids[held]] = index
        yield WinePlan(fold, fit_ids, np.sort(validation_ids), test_ids, inner_holdout)


def wine_plans():
    """Return row-level provenance, without fitting any estimator."""
    return tuple(_plans(load_wine().target))


def wine_hashes():
    """Hash raw data and the ordered [outer fold, original row, assignment] manifest."""
    x, y = load_wine(return_X_y=True)
    assignments = np.full((5, len(y), 2), -1, dtype="<i2")
    for plan in _plans(y):
        for role, ids in enumerate((plan.fit_ids, plan.validation_ids, plan.test_ids)):
            assignments[plan.fold, ids, 0] = role
        assignments[plan.fold, :, 1] = plan.inner_holdout
    return {
        "features": sha256(np.asarray(x, dtype="<f8", order="C").tobytes()).hexdigest(),
        "labels": sha256(np.asarray(y, dtype="<i8", order="C").tobytes()).hexdigest(),
        "splits": sha256(assignments.tobytes()).hexdigest(),
    }


def _providers():
    return (
        make_pipeline(
            StandardScaler(), LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
        ),
        GaussianNB(var_smoothing=1e-9),
        make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(
                n_neighbors=9, weights="distance", p=2, algorithm="brute"
            ),
        ),
        DecisionTreeClassifier(max_depth=3, min_samples_leaf=4, random_state=7717),
        ExtraTreesClassifier(
            n_estimators=96,
            max_features="sqrt",
            min_samples_leaf=2,
            random_state=7718,
            n_jobs=1,
        ),
        make_pipeline(
            SelectKBest(f_classif, k=6),
            StandardScaler(),
            LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
        ),
    )


def build_wine_fold(fold):
    """Return fit OOF probabilities, held-out probabilities and the exact provider splits.

    Each probability tensor has shape (samples, providers, classes). Its rows
    follow the original specimen IDs stored on the split. Class columns are 0, 1, 2.
    """
    if isinstance(fold, bool) or not isinstance(fold, Integral) or fold not in range(5):
        raise ValueError("Wine fold must be an integer from 0 to 4")
    x, y = load_wine(return_X_y=True)
    plan = tuple(_plans(y))[fold]
    classes = np.arange(3)
    preds = np.full((len(y), len(PROVIDER_NAMES), len(classes)), np.nan)
    for job in plan.provider_jobs():
        for column, model in enumerate(_providers()):
            model.fit(x[job.train_ids], y[job.train_ids])
            if not np.array_equal(model.classes_, classes):
                raise ValueError("Wine provider did not see all expected classes")
            preds[job.predict_ids, column] = model.predict_proba(x[job.predict_ids])
    if (
        not np.isfinite(preds).all()
        or preds.min() < 0
        or not np.allclose(preds.sum(axis=-1), 1.0, rtol=0, atol=1e-12)
    ):
        raise ValueError("Incomplete or invalid Wine provider probabilities")

    def split(ids):
        return WineSplit(ids, preds[ids], y[ids])

    return WineFold(
        plan, split(plan.fit_ids), split(plan.validation_ids), split(plan.test_ids)
    )


def write_wine_inputs(output):
    """Write all frozen matrices and a row-level provenance manifest."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    entries = []
    arrays = {}
    with threadpool_limits(limits=1):
        for fold in range(5):
            data = build_wine_fold(fold)
            jobs = [
                {
                    "train_ids": job.train_ids.tolist(),
                    "predict_ids": job.predict_ids.tolist(),
                }
                for job in data.plan.provider_jobs()
            ]
            split_metadata = {}
            for name, split in (
                ("fit", data.fit),
                ("validation", data.validation),
                ("iid", data.test),
            ):
                prefix = f"fold{fold}_{name}"
                arrays[f"{prefix}_preds"] = np.asarray(
                    split.preds, dtype="<f8", order="C"
                )
                arrays[f"{prefix}_labels"] = np.asarray(split.y, dtype="<i8")
                arrays[f"{prefix}_ids"] = np.asarray(split.row_ids, dtype="<i8")
                split_metadata[name] = {
                    "row_ids": split.row_ids.tolist(),
                    "shape": list(split.preds.shape),
                    "sha256": sha256(arrays[f"{prefix}_preds"].tobytes()).hexdigest(),
                }
            entries.append(
                {"fold": fold, "provider_jobs": jobs, "splits": split_metadata}
            )
    manifest = {
        "source": wine_hashes(),
        "numpy_version": np.__version__,
        "sklearn_version": sklearn.__version__,
        "providers": list(PROVIDER_NAMES),
        "class_order": [0, 1, 2],
        "folds": entries,
    }
    np.savez_compressed(output / "wine-inputs.npz", **arrays)
    (output / "wine-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def run_wine(output, *, independent_refresh=False):
    """Use each frozen provider bank for all searches, budget checkpoints and baselines."""
    import pandas as pd

    from fusionbench.eval.mutation_opportunity import evaluate_bank

    output = Path(output)
    if (
        not (output / "wine-inputs.npz").exists()
        or not (output / "wine-manifest.json").exists()
    ):
        write_wine_inputs(output)
    manifest = json.loads((output / "wine-manifest.json").read_text())
    if manifest["source"] != wine_hashes():
        raise ValueError("Wine source or split manifest differs from the stored bank")
    plans = wine_plans()
    labels = load_wine().target
    if [entry["fold"] for entry in manifest["folds"]] != list(range(5)):
        raise ValueError("Wine bank must contain all five ordered outer folds")
    options = (
        {"arms": ("refresh_independent",), "include_baselines": False}
        if independent_refresh
        else {}
    )
    rows = []
    with np.load(output / "wine-inputs.npz") as saved, threadpool_limits(limits=1):
        for entry in manifest["folds"]:
            fold = entry["fold"]
            plan = plans[fold]
            ids = {
                "fit": plan.fit_ids,
                "validation": plan.validation_ids,
                "iid": plan.test_ids,
            }
            splits = {}
            for name, expected_ids in ids.items():
                meta = entry["splits"][name]
                prefix = f"fold{fold}_{name}"
                z, y = saved[f"{prefix}_preds"], saved[f"{prefix}_labels"]
                if (
                    meta["row_ids"] != expected_ids.tolist()
                    or not np.array_equal(saved[f"{prefix}_ids"], expected_ids)
                    or not np.array_equal(y, labels[expected_ids])
                ):
                    raise ValueError(f"Wine row or label mismatch in {prefix}")
                if sha256(z.tobytes()).hexdigest() != meta["sha256"]:
                    raise ValueError(f"Wine probability digest mismatch in {prefix}")
                splits[name] = (z, y, None)
            rows.extend(
                evaluate_bank("wine", fold, splits, gp_seeds=(661, 673), **options)
            )
    filename = (
        "wine-refresh-independent-runs.csv.gz"
        if independent_refresh
        else "wine-runs.csv.gz"
    )
    pd.DataFrame(rows).to_csv(output / filename, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--independent-refresh", action="store_true")
    args = parser.parse_args()
    if args.prepare_only:
        write_wine_inputs(args.out)
    else:
        run_wine(args.out, independent_refresh=args.independent_refresh)

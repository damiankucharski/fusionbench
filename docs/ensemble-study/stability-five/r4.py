"""Cross-validated choice of one fitted native model for deployment.

The simplex and GP each use the same three training folds to predict the fourth.
One paired gain per example determines the decision. The row standard error is a
margin heuristic: examples across folds share partially overlapping training sets.
"""

from hashlib import sha256
from time import perf_counter

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

N_FOLDS = 4
FOLD_SEED = 7304
FAMILIES = ("simplex", "regularized")
POLICIES = ("cf_gain", "cf_margin")


def _new_model(family):
    if family == "simplex":
        from fusionbench.methods.simplex_tree import SimplexTree

        return SimplexTree(regularized=True)
    from fusionbench.methods.native_pattern import RegularizedNativePatternOKAPI

    return RegularizedNativePatternOKAPI(seed=0, split_seed=0)


def _fit(family, x, labels):
    classes = x.shape[-1]
    task = "binary" if classes == 2 else "multiclass"
    return _new_model(family).fit(x, labels, task=task, n_classes=classes)


def _folds(labels):
    if len(labels) < N_FOLDS:
        raise ValueError("Four-fold selection requires at least four training rows")
    kwargs = {"n_splits": N_FOLDS, "shuffle": True, "random_state": FOLD_SEED}
    _, class_counts = np.unique(labels, return_counts=True)
    # sklearn cannot stratify when even the largest class has fewer than four rows.
    if class_counts.max() < N_FOLDS:
        return "kfold", KFold(**kwargs).split(labels)
    return "stratified", StratifiedKFold(**kwargs).split(labels, labels)


def _crossfit(x, labels):
    splitter, folds = _folds(labels)
    classes = x.shape[-1]
    fold_id = np.full(len(labels), -1, dtype="int8")
    losses = {family: np.empty(len(labels)) for family in FAMILIES}
    fold_rows = []
    times = dict.fromkeys(FAMILIES, 0.0)
    for fold, (train, held) in enumerate(folds):
        fold_id[held] = fold
        fold_rows.append(
            {
                "train": len(train),
                "held": len(held),
                "train_class_counts": np.bincount(
                    labels[train], minlength=classes
                ).tolist(),
                "held_class_counts": np.bincount(
                    labels[held], minlength=classes
                ).tolist(),
            }
        )
        for family in FAMILIES:
            start = perf_counter()
            model = _fit(family, x[train], labels[train])
            predicted = model.predict(x[held])
            loss = np.square(predicted - np.eye(classes)[labels[held]]).sum(axis=1)
            if predicted.shape != (len(held), classes) or not np.isfinite(loss).all():
                raise ValueError(f"Invalid held-fold predictions for {family}")
            losses[family][held] = loss
            times[family] += perf_counter() - start
    if np.any(fold_id == -1):
        raise RuntimeError("Every source row must be predicted out of fold")
    delta = losses["simplex"] - losses["regularized"]
    gain = float(delta.mean())
    se = float(delta.std(ddof=1) / np.sqrt(len(delta)))
    choices = {
        "cf_gain": "regularized" if gain > 0.0 else "simplex",
        "cf_margin": "regularized" if gain > se else "simplex",
    }
    info = {
        "splitter": splitter,
        "fold_seed": FOLD_SEED,
        "fold_id_sha256": sha256(fold_id.tobytes()).hexdigest(),
        "fold_rows": fold_rows,
        "oof_brier": {name: float(value.mean()) for name, value in losses.items()},
        "oof_gain": gain,
        "oof_se": se,
        "oof_seconds": times,
        "oof_fits": dict.fromkeys(FAMILIES, N_FOLDS),
    }
    return choices, info


def _fit_requested(x, y, arms):
    x = np.asarray(x)
    labels = np.asarray(y).reshape(-1)
    if x.ndim != 3 or len(labels) != len(x) or labels.dtype.kind not in "iub":
        raise ValueError(
            "Expected aligned class probabilities [N,M,C] and integer labels"
        )
    labels = labels.astype(np.intp, copy=False)
    if not np.all((labels >= 0) & (labels < x.shape[-1])):
        raise ValueError("Labels must index the supplied class axis")
    chosen = {name: name for name in FAMILIES}
    info = {"oof_fits": dict.fromkeys(FAMILIES, 0)}
    if any(arm in POLICIES for arm in arms):
        choices, info = _crossfit(x, labels)
        chosen.update(choices)
    full = {}
    seconds = {}
    for family in dict.fromkeys(chosen[arm] for arm in arms):
        start = perf_counter()
        full[family] = _fit(family, x, labels)
        seconds[family] = perf_counter() - start
    info.update(
        selected_family={arm: chosen[arm] for arm in arms},
        full_fit_seconds=seconds,
        full_fits={family: int(family in full) for family in FAMILIES},
    )
    return {arm: full[chosen[arm]] for arm in arms}, info


def fit_variants(x, y):
    """Fit the two standalone models and both matched cross-fit selectors."""
    return _fit_requested(x, y, (*FAMILIES, *POLICIES))


def fit_variant(x, y, arm):
    """Train a single confirmation arm and return its deployed predictor and metadata."""
    if arm not in (*FAMILIES, *POLICIES):
        raise ValueError(f"Unknown R4 method {arm!r}")
    models, info = _fit_requested(x, y, (arm,))
    return models[arm], info

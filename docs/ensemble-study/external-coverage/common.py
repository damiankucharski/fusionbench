"""Shared fit-side profiles and frozen fusion methods for external coverage."""

from time import perf_counter

import numpy as np

from fusionbench.eval.metrics import normalize_proba
from fusionbench.eval.transfer_grid import brier
from fusionbench.methods.averaging import SimpleAverage, TopN
from fusionbench.methods.native_pattern import NativePatternOKAPI
from fusionbench.methods.stacking import StackingLogit, StackingLR
from fusionbench.types import Dataset

TEMPERATURES = (1.0, 0.25, 0.5, 0.75, 1.5, 2.0)


def _temper(mean, temperature):
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if temperature == 1:
        return normalize_proba(mean)
    # Scale relative to the row maximum before the power. The common factor
    # cancels in normalization and at least one class retains unit mass.
    relative = mean / mean.max(axis=1, keepdims=True)
    mass = relative ** (1.0 / temperature)
    return normalize_proba(mass / mass.sum(axis=1, keepdims=True))


def mean_temperature(preds, temperature):
    """Raise the arithmetic pool to power 1/T and use the suite's probability normalization."""
    return _temper(np.asarray(preds).mean(axis=1).astype(float, copy=False), temperature)


def fit_mean_temperature(fit_preds, fit_y):
    """Minimize full Brier on fit rows. Exact ties prefer 1, then ascending T."""
    mean = np.asarray(fit_preds).mean(axis=1).astype(float, copy=False)
    return min(TEMPERATURES, key=lambda temperature: brier(_temper(mean, temperature), fit_y))


def profile_fit(fit_preds, fit_y):
    """Compute fit-only provider statistics on the original probability values.

    The disagreement averages all unordered distinct provider pairs and rows.
    A unanimous error requires every provider to choose the same wrong class.
    Top-1 ties use the lowest class index.
    """
    x = np.asarray(fit_preds)
    y = np.asarray(fit_y)
    if x.ndim != 3 or min(x.shape) == 0 or x.shape[0] != len(y) or x.shape[2] < 2:
        raise ValueError("Expected a nonempty [row, provider, class] bank and matching labels")
    n, providers, classes = x.shape
    marginal = [brier(x[:, i, :], y) for i in range(providers)]
    top = x.argmax(axis=2)
    votes = np.zeros((n, classes), dtype=np.int64)
    np.add.at(votes, (np.arange(n)[:, None], top), 1)
    agreement = np.sum(votes * (votes - 1), axis=1)
    disagreement = float(np.mean(1 - agreement / (providers * (providers - 1)))) if providers > 1 else None
    return {
        "mean_provider_brier": float(np.mean(marginal)),
        "best_provider_brier": min(marginal),
        "mean_pairwise_top1_disagreement": disagreement,
        "zero_fraction": float(np.mean(x == 0)),
        "mean_max_probability": float(x.max(axis=2).mean(dtype=np.float64)),
        "unanimous_wrong_fraction": float(np.mean((votes.max(axis=1) == providers) & (top[:, 0] != y))),
    }


def evaluate_methods(dataset: Dataset, *, seed: int):
    """Evaluate five core methods and four diagnostics on one held-out split.

    Return one flat dictionary per method. Fit durations exclude prediction
    and test scoring. The fixed sharpener uses T=0.5 for every bank.
    """
    from symmetric import PooledScaleStackingLR, SymmetricStackingLR

    methods = {
        "native": NativePatternOKAPI(seed=seed, l2_strength=0.25),
        "mean": SimpleAverage(),
        "top_n": TopN("brier"),
        "prob_lr": StackingLR(c_value=1),
        "logit": StackingLogit(c_value=1),
        "symmetric_lr": SymmetricStackingLR(c_value=1),
        "pooled_lr": PooledScaleStackingLR(c_value=1),
    }
    seconds = {}
    for label, model in methods.items():
        start = perf_counter()
        model.fit(dataset.fit_preds, dataset.fit_y, task=dataset.task, n_classes=dataset.n_classes)
        seconds[label] = perf_counter() - start
    start = perf_counter()
    temperature = fit_mean_temperature(dataset.fit_preds, dataset.fit_y)
    seconds["temperature"] = perf_counter() - start
    seconds["sharpener"] = 0.0

    predictions = {label: model.predict(dataset.test_preds) for label, model in methods.items()}
    predictions["temperature"] = mean_temperature(dataset.test_preds, temperature)
    predictions["sharpener"] = mean_temperature(dataset.test_preds, 0.5)
    rows = []
    for label, p in predictions.items():
        rows.append({
            "method": label,
            "brier": brier(p, dataset.test_y),
            "accuracy": float(np.mean(p.argmax(axis=1) == dataset.test_y)),
            "fit_seconds": seconds[label],
            "native_choice": methods["native"].chosen_ if label == "native" else None,
            "top_n": int(methods["top_n"].n_selected_) if label == "top_n" else None,
            "temperature": {"temperature": temperature, "sharpener": 0.5}.get(label),
        })
    return rows

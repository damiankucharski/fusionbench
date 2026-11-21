"""Run methods x scenarios and collect per-(scenario, method) records."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fusionbench.eval.cost import Timer
from fusionbench.eval.metrics import HIGHER_BETTER, METRIC_NAMES, score
from fusionbench.methods.averaging import TopN
from fusionbench.methods.registry import default_methods
from fusionbench.types import Dataset


def _regret(primary: str, oracle_val: float, method_val: float) -> float:
    """Non-negative shortfall of a method vs the oracle on the primary metric."""
    return (oracle_val - method_val) if HIGHER_BETTER[primary] else (method_val - oracle_val)


def evaluate_dataset(ds: Dataset, methods=None) -> pd.DataFrame:
    methods = methods if methods is not None else default_methods()
    primary = ds.meta.get("primary_metric", "macro_f1")
    rows = []

    # Oracle ceiling (analytic Bayes posterior), if present.
    oracle_test = None
    if ds.test_oracle is not None:
        o_test = score(ds.test_y, ds.test_oracle, ds.n_classes)
        # fit_oracle is absent on degraded datasets; only the test ceiling matters.
        o_fit = score(ds.fit_y, ds.fit_oracle, ds.n_classes) if ds.fit_oracle is not None else o_test
        oracle_test = o_test[primary]
        rows.append(_row(ds, "oracle", primary, o_fit, o_test, 0.0, 0.0, oracle_test))

    for method in methods:
        ft, pt = Timer(), Timer()
        try:
            with ft.measure():
                options = {"primary_metric": primary} if isinstance(method, TopN) else {}
                method.fit(ds.fit_preds, ds.fit_y, task=ds.task, n_classes=ds.n_classes, **options)
            fit_pred = method.predict(ds.fit_preds)
            with pt.measure():
                test_pred = method.predict(ds.test_preds)
            fit_m = score(ds.fit_y, fit_pred, ds.n_classes)
            test_m = score(ds.test_y, test_pred, ds.n_classes)
        except Exception as exc:  # noqa: BLE001 — record and continue
            rows.append({"scenario": ds.name, "method": method.name, "primary_metric": primary,
                         "regret": np.nan, "error": repr(exc)})
            continue
        rows.append(_row(ds, method.name, primary, fit_m, test_m,
                         ft.elapsed, pt.elapsed, oracle_test))
        if isinstance(method, TopN):
            rows[-1].update(n_selected=method.n_selected_, fit_metric=method.metric_)

    df = pd.DataFrame(rows)
    df["expected_winner"] = df["method"].isin(ds.expected_winners)
    return df


def _row(ds, name, primary, fit_m, test_m, fit_t, pred_t, oracle_test):
    row = {
        "scenario": ds.name,
        "task": ds.task,
        "n_classes": ds.n_classes,
        "primary_metric": primary,
        "method": name,
        "fit_time_s": fit_t,
        "predict_time_s": pred_t,
        "error": "",
    }
    for m in METRIC_NAMES:
        row[f"fit_{m}"] = fit_m[m]
        row[f"test_{m}"] = test_m[m]
    row["val_test_gap"] = fit_m[primary] - test_m[primary]
    if oracle_test is not None and name != "oracle":
        row["regret"] = _regret(primary, oracle_test, test_m[primary])
    else:
        row["regret"] = 0.0 if name == "oracle" else np.nan
    return row


def run_suite(scenarios, methods=None, n_per_split: int = 50_000, seed: int = 0
              ) -> pd.DataFrame:
    frames = []
    for sc in scenarios:
        ds = sc.build(n_per_split=n_per_split, seed=seed)
        frames.append(evaluate_dataset(ds, methods))
    return pd.concat(frames, ignore_index=True)

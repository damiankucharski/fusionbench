"""Run methods x scenarios and collect per-(scenario, method) records."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fusionbench.eval.cost import Timer
from fusionbench.eval.metrics import HIGHER_BETTER, METRIC_NAMES, score
from fusionbench.methods.averaging import TopN
from fusionbench.methods.registry import FACTORIES, default_methods
from fusionbench.types import Dataset


def _regret(primary: str, oracle_val: float, method_val: float) -> float:
    """Empirical difference on this test split, positive when the oracle scores better."""
    return (
        (oracle_val - method_val)
        if HIGHER_BETTER[primary]
        else (method_val - oracle_val)
    )


def evaluate_dataset(ds: Dataset, methods=None) -> pd.DataFrame:
    """Fit and score the selected method instances or registry IDs on one bank."""
    if methods is None:
        methods = default_methods()
    elif isinstance(methods, str):
        methods = [methods]
    primary = ds.meta.get("primary_metric", "macro_f1")
    rows = []

    oracle_test = None
    if ds.test_oracle is not None:
        o_test = score(ds.test_y, ds.test_oracle, ds.n_classes)
        o_fit = (
            score(ds.fit_y, ds.fit_oracle, ds.n_classes)
            if ds.fit_oracle is not None
            else dict.fromkeys(METRIC_NAMES, np.nan)
        )
        oracle_test = o_test[primary]
        rows.append(_row(ds, "oracle", primary, o_fit, o_test, 0.0, 0.0, oracle_test))

    for method in methods:
        if isinstance(method, str):
            if method not in FACTORIES:
                raise ValueError(f"Unknown fusion method {method!r}")
            method = FACTORIES[method]()
            if not method.available():
                raise ValueError(f"Requested method {method.name!r} is unavailable")
        ft, pt = Timer(), Timer()
        try:
            with ft.measure():
                options = (
                    {"primary_metric": primary} if isinstance(method, TopN) else {}
                )
                method.fit(
                    ds.fit_preds,
                    ds.fit_y,
                    task=ds.task,
                    n_classes=ds.n_classes,
                    **options,
                )
            fit_pred = method.predict(ds.fit_preds)
            with pt.measure():
                test_pred = method.predict(ds.test_preds)
            fit_m = score(ds.fit_y, fit_pred, ds.n_classes)
            test_m = score(ds.test_y, test_pred, ds.n_classes)
        except Exception as exc:  # noqa: BLE001 — record and continue
            rows.append(
                {
                    "scenario": ds.name,
                    "method": method.name,
                    "primary_metric": primary,
                    "regret": np.nan,
                    "error": repr(exc),
                }
            )
            continue
        rows.append(
            _row(
                ds,
                method.name,
                primary,
                fit_m,
                test_m,
                ft.elapsed,
                pt.elapsed,
                oracle_test,
            )
        )
        if isinstance(method, TopN):
            rows[-1].update(n_selected=method.n_selected_, fit_metric=method.metric_)

    if not rows:
        raise ValueError(
            "Select a fusion method to evaluate on a dataset without an oracle"
        )
    df = pd.DataFrame(rows)
    df["scenario_id"] = ds.meta.get("scenario_id", ds.name)
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


def run_suite(
    scenarios=None, methods=None, n_per_split: int = 50_000, seed: int = 0
) -> pd.DataFrame:
    """Evaluate registered IDs or configured Scenario objects. The default covers standard-v6."""
    from fusionbench.scenarios import get_scenario
    from fusionbench.scenarios.registry import STANDARD_V6_SCENARIOS

    if scenarios is None:
        scenarios = STANDARD_V6_SCENARIOS
    elif isinstance(scenarios, str):
        scenarios = [scenarios]
    if methods is not None and not isinstance(methods, str):
        methods = tuple(methods)
    frames = []
    for sc in scenarios:
        if isinstance(sc, str):
            sc = get_scenario(sc)
        ds = sc.build(n_per_split=n_per_split, seed=seed)
        frame = evaluate_dataset(ds, methods)
        frame["data_seed"] = seed
        frames.append(frame)
    if not frames:
        raise ValueError("Select at least one scenario")
    return pd.concat(frames, ignore_index=True)


def summarize_suite(results, *, weights=None):
    """Average test Brier across draws within scenarios, then over the selected scenarios.

    Return a selected-panel score and a score per represented standard mechanism group.
    Supply the scenario weights from fusionbench.taxonomy to score a particular application.
    A missing or failed method on any selected scenario leaves that panel's score undefined.
    """
    from collections.abc import Mapping

    from fusionbench.taxonomy import MECHANISM_GROUPS, scenario_weights

    rows = results.loc[results["method"] != "oracle"].copy()
    keys = "scenario_id" if "scenario_id" in rows else "scenario"
    if "data_seed" in rows and (
        rows["data_seed"].isna().any()
        or rows.duplicated([keys, "method", "data_seed"]).any()
    ):
        raise ValueError(
            "A benchmark requires one result per scenario, method and data_seed; select one configuration before aggregating"
        )
    if weights is None:
        shares = dict.fromkeys(rows[keys].unique(), 1.0)
    elif isinstance(weights, Mapping):
        shares = scenario_weights(scenarios=weights)
    else:
        raise TypeError("Provide an ID-to-weight mapping or omit weights")
    if not shares or rows.empty:
        raise ValueError("Cannot summarize an empty benchmark")
    # Preserve failed records and distinguish a reported error from a zero loss.
    if "test_brier" not in rows:
        rows["test_brier"] = np.nan
    rows.loc[rows["error"].fillna("").ne(""), "test_brier"] = np.nan
    means = (
        rows.groupby([keys, "method"])["test_brier"]
        .agg(lambda v: v.mean(skipna=False))
        .unstack("method")
    )
    if "data_seed" in rows:
        # Keep the paired seed set common to methods when combining saved run results.
        expected = rows.groupby(keys)["data_seed"].nunique()
        counts = rows.groupby([keys, "method"])["data_seed"].nunique().unstack("method")
        means = means.where(counts.eq(expected, axis=0))
    panels = {"selected": shares}
    for group, names in MECHANISM_GROUPS.items():
        if local := {s: shares[s] for s in names if s in shares}:
            panels[group] = local
    records = []
    for group, local in panels.items():
        subset = means.reindex(local)
        mass = list(local.values())
        for method, scores in subset.items():
            seen = int(np.isfinite(scores).sum())
            records.append(
                {
                    "group": group,
                    "method": method,
                    "scenarios": len(local),
                    "scored": seen,
                    "mean_brier": np.average(scores, weights=mass)
                    if seen == len(local)
                    else np.nan,
                }
            )
    return pd.DataFrame(records)

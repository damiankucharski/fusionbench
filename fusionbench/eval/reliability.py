"""Reliability diagnostics — the benchmark's headline guarantee.

Because both splits come from the identical i.i.d. process at large N, selecting a
method by its fit-split score should pick (essentially) the test-best method, and
the fit->test ranking should be preserved. These functions quantify that:
selection-top-1 agreement, Spearman rank correlation of the primary metric, and
the worst validation-test gap. A scenario where these hold confirms reliability
is controlled, so any method's shortfall is expressiveness, not selection luck.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return float("nan")
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _primary_cols(group: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, bool]:
    primary = group["primary_metric"].iloc[0]
    higher_better = primary not in ("log_loss", "ece", "brier")
    fit = group[f"fit_{primary}"].to_numpy()
    test = group[f"test_{primary}"].to_numpy()
    return fit, test, higher_better


def reliability_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-scenario reliability diagnostics over the competing methods."""
    rows = []
    valid = df[(df["method"] != "oracle") & (df["error"] == "")]
    for scenario, group in valid.groupby("scenario", sort=False):
        fit, test, higher_better = _primary_cols(group)
        methods = group["method"].to_numpy()
        pick = (np.argmax if higher_better else np.argmin)
        fit_best = methods[pick(fit)]
        test_best = methods[pick(test)]
        rows.append({
            "scenario": scenario,
            "primary_metric": group["primary_metric"].iloc[0],
            "n_methods": len(group),
            "selected_by_fit": fit_best,
            "actual_test_best": test_best,
            "selection_correct": bool(fit_best == test_best),
            "spearman_fit_test": _spearman(fit, test),
            "max_abs_gap": float(np.max(np.abs(fit - test))),
        })
    return pd.DataFrame(rows)

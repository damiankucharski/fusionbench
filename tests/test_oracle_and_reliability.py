"""Empirical checks on oracle scores and ranking stability for fixed benchmark draws."""

from dataclasses import replace

import pytest

from fusionbench.eval.metrics import METRIC_NAMES
from fusionbench.eval.reliability import reliability_summary
from fusionbench.eval.runner import evaluate_dataset
from fusionbench.methods.registry import CORE, default_methods
from fusionbench.scenarios.compositional_switch import CompositionalSwitch
from fusionbench.scenarios.registry import default_suite

N = 8000


def test_missing_fit_oracle_keeps_test_scores_and_leaves_fit_metrics_missing():
    dataset = CompositionalSwitch().build(n_per_split=64, seed=0)
    full = evaluate_dataset(dataset, methods=[]).iloc[0]
    test_only = evaluate_dataset(replace(dataset, fit_oracle=None), methods=[]).iloc[0]
    fit_columns = [f"fit_{metric}" for metric in METRIC_NAMES]
    test_columns = [f"test_{metric}" for metric in METRIC_NAMES]
    assert test_only.index.equals(full.index)
    assert test_only[fit_columns + ["val_test_gap"]].isna().all()
    assert test_only[test_columns].equals(full[test_columns])
    assert test_only["method"] == "oracle" and test_only["regret"] == 0.0


@pytest.fixture(scope="module")
def results():
    # Core methods only: fast, deterministic, and independent of the optional
    # okapi/nsga3/tpot dependencies (those are validated separately).
    methods = default_methods(CORE)
    frames = []
    for sc in default_suite():
        frames.append(evaluate_dataset(sc.build(n_per_split=N, seed=0), methods))
    import pandas as pd
    return pd.concat(frames, ignore_index=True)


def test_oracle_not_beaten(results):
    comp = results[(results.method != "oracle") & (results.error == "")]
    # Allow a small tolerance for finite-sample noise at the reduced test N.
    assert comp["regret"].min() > -0.025, comp.loc[comp.regret.idxmin()].to_dict()


def test_oracle_present_for_all(results):
    scen = set(results["scenario"])
    have = set(results[results.method == "oracle"]["scenario"])
    assert scen - {"similar_classes_38"} == have
    assert "similar_classes_38" in scen  # Its posterior given reported scores is unknown.


def test_reliability_selection(results):
    rel = reliability_summary(results)
    # Selecting by fit should pick the test-best method on most scenarios
    # (ties on saturated/mixture are harmless), and ranks should correlate.
    assert rel["selection_correct"].mean() >= 0.6
    sp = rel["spearman_fit_test"].dropna()
    assert sp.mean() >= 0.5


def test_max_gap_small(results):
    rel = reliability_summary(results)
    # Fit/test gaps are intentionally large for explicitly shifted domains.
    shifted = {s.name for s in default_suite()
               if s._registry_metadata["scenario_id"] in ("environment_shortcut", "cloned_shortcut")}
    assert rel.loc[~rel.scenario.isin(shifted), "max_abs_gap"].max() < 0.05

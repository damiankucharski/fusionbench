"""The two methodological guarantees of the benchmark.

1. The analytic Bayes oracle is a genuine ceiling: no fitted method beats it on
   the scenario's primary metric (beyond sampling noise).
2. Validation-test reliability holds: selecting a method by its fit score picks
   (essentially) the test-best method, and the fit->test ranking is preserved.
"""

import numpy as np
import pytest

from fusionbench.eval.reliability import reliability_summary
from fusionbench.eval.report import add_primary
from fusionbench.eval.runner import evaluate_dataset
from fusionbench.scenarios.registry import default_suite

N = 8000


@pytest.fixture(scope="module")
def results():
    frames = []
    for sc in default_suite():
        frames.append(evaluate_dataset(sc.build(n_per_split=N, seed=0)))
    import pandas as pd
    return pd.concat(frames, ignore_index=True)


def test_oracle_not_beaten(results):
    comp = results[(results.method != "oracle") & (results.error == "")]
    # Allow a small tolerance for finite-sample noise at the reduced test N.
    assert comp["regret"].min() > -0.025, comp.loc[comp.regret.idxmin()].to_dict()


def test_oracle_present_for_all(results):
    scen = set(results["scenario"])
    have = set(results[results.method == "oracle"]["scenario"])
    assert scen == have


def test_reliability_selection(results):
    rel = reliability_summary(results)
    # Selecting by fit should pick the test-best method on most scenarios
    # (ties on saturated/mixture are harmless), and ranks should correlate.
    assert rel["selection_correct"].mean() >= 0.6
    sp = rel["spearman_fit_test"].dropna()
    assert sp.mean() >= 0.5


def test_max_gap_small(results):
    rel = reliability_summary(results)
    # No scenario should show a large validation-test gap given the controlled DGP.
    assert rel["max_abs_gap"].max() < 0.05

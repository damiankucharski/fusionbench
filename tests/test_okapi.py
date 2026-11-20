"""OKAPI adapter smoke test (skipped when okapi/torch are not installed).

Validates that the real OKAPI search runs end-to-end and, in its target regime
(a logical-gate pathology with competent base models), discovers a gating rule
that clearly beats naive averaging and stays parsimonious.
"""

import numpy as np
import pytest

pytest.importorskip("okapi")
pytest.importorskip("torch")

from fusionbench.eval.metrics import score  # noqa: E402
from fusionbench.methods.averaging import SimpleAverage  # noqa: E402
from fusionbench.methods.okapi import OKAPIMethod  # noqa: E402
from fusionbench.scenarios.gates import HardRescue  # noqa: E402


def test_okapi_finds_gate():
    ds = HardRescue().build(n_per_split=8000, seed=0)

    # Single-level gate search, as in the existing depth-control experiment.
    ok = OKAPIMethod(max_depth=3, generations=100, n_restarts=1, n_seeds=20, seed=0)
    ok.fit(ds.fit_preds, ds.fit_y, task=ds.task, n_classes=ds.n_classes)
    ok_f1 = score(ds.test_y, ok.predict(ds.test_preds), ds.n_classes)["macro_f1"]

    avg = SimpleAverage().fit(ds.fit_preds, ds.fit_y, task=ds.task, n_classes=ds.n_classes)
    avg_f1 = score(ds.test_y, avg.predict(ds.test_preds), ds.n_classes)["macro_f1"]

    # OKAPI should discover an OR/max-style gate: clearly above naive averaging,
    # and compact. (Exact closeness to the oracle is budget-dependent, so we
    # assert the robust facts only.)
    assert ok_f1 > avg_f1 + 0.12, f"okapi {ok_f1:.3f} not clearly above avg {avg_f1:.3f}"
    # Compact relative to a dense K*C-weight stacker (seeding can add material).
    assert ok.best_tree.depth <= 3
    assert ok.best_nodes <= 30, f"tree not parsimonious: {ok.best_nodes} nodes"

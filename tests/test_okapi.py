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

    ok = OKAPIMethod(generations=100, n_restarts=1, n_seeds=20, seed=0, max_nodes=30)
    ok.fit(ds.fit_preds, ds.fit_y, task=ds.task, n_classes=ds.n_classes)
    ok_f1 = score(ds.test_y, ok.predict(ds.test_preds), ds.n_classes)["macro_f1"]

    avg = SimpleAverage().fit(
        ds.fit_preds, ds.fit_y, task=ds.task, n_classes=ds.n_classes
    )
    avg_f1 = score(ds.test_y, avg.predict(ds.test_preds), ds.n_classes)["macro_f1"]

    # OKAPI should discover an OR/max-style gate: clearly above naive averaging,
    # and compact. (Exact closeness to the oracle is budget-dependent, so we
    # assert the robust facts only.)
    assert ok_f1 > avg_f1 + 0.12, (
        f"okapi {ok_f1:.3f} not clearly above avg {avg_f1:.3f}"
    )
    # The explicit upstream structural budget applies to the seeded population and offspring.
    assert ok.best_nodes <= 30, f"tree not parsimonious: {ok.best_nodes} nodes"


def test_interleaved_tasks_restore_caller_state():
    from okapi.globals import postprocessing_function, set_postprocessing_function

    rng = np.random.default_rng(23)
    params = dict(generations=2, n_restarts=1, n_seeds=3, seed=4, fitness="brier")
    previous = postprocessing_function._postprocessing_function
    original_rng = np.random.get_state()
    try:
        set_postprocessing_function(np.negative)
        first = rng.dirichlet([2, 2], size=(64, 3))
        a = OKAPIMethod(**params).fit(
            first, np.arange(64) % 2, task="binary", n_classes=2
        )
        p_binary = a.predict(first)
        second = rng.dirichlet([2, 2, 2], size=(64, 4))
        b = OKAPIMethod(**params).fit(
            second, np.arange(64) % 3, task="multiclass", n_classes=3
        )
        p_multi = b.predict(second)
        np.testing.assert_array_equal(a.predict(first), p_binary)
        np.testing.assert_array_equal(b.predict(second), p_multi)
        np.testing.assert_allclose(p_multi.sum(1), 1)
        np.testing.assert_array_equal(
            postprocessing_function(np.array([1, 2])), [-1, -2]
        )
        current = np.random.get_state()
        assert current[0] == original_rng[0] and current[2:] == original_rng[2:]
        np.testing.assert_array_equal(current[1], original_rng[1])
    finally:
        set_postprocessing_function(previous)
        np.random.set_state(original_rng)


def test_single_provider_returns_a_real_one_leaf_tree():
    p = np.random.default_rng(20).dirichlet([1, 1], size=(30, 1))
    model = OKAPIMethod().fit(p, np.arange(30) % 2, task="binary", n_classes=2)
    assert model.best_nodes == 1
    np.testing.assert_allclose(model.predict(p), p[:, 0, :])

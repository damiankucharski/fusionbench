import importlib
from fractions import Fraction

import numpy as np
import pytest

from fusionbench.eval.structural_key_audit import MaskAudit, exact_brier


def test_hit_comparison_resolves_losses_below_float_precision():
    p = np.array([1.0, np.nextafter(0.0, 1.0)])
    y = np.zeros(2)
    assert float(exact_brier(p, y)) == 0.5
    assert exact_brier(p, y) > Fraction(1, 2)


@pytest.mark.parametrize(
    "policy,expected",
    [("legacy", [True, False, False]), ("fixed", [True, True, False])],
)
def test_monitor_detects_alias_even_when_all_fit_predictions_match(
    monkeypatch, policy, expected
):
    pytest.importorskip("okapi")
    if not hasattr(importlib.import_module("okapi.okapi"), "_unique_tree_mask"):
        pytest.skip("requires the OKAPI structural-key revision")
    from okapi.backend.backend import Backend
    from okapi.backend.numpy_backend import NumpyBackend
    from okapi.globals import postprocessing_function
    from okapi.node import MaxNode, MinNode, ValueNode
    from okapi.tree import Tree

    monkeypatch.setattr(Backend, "_current_backend", NumpyBackend)
    monkeypatch.setattr(postprocessing_function, "_postprocessing_function", np.asarray)
    x = {name: np.zeros((2, 1)) for name in "ABCD"}

    def make_tree(n):
        a, *other = [ValueNode(None, value, name) for name, value in x.items()]
        a.add_child(MinNode(other[:n]))
        a.add_child(MaxNode(other[n:]))
        return Tree(a)

    a, b = make_tree(2), make_tree(1)
    monitor = MaskAudit(policy, x)
    assert monitor([a, b, a.copy()]) == expected
    assert monitor.suppressed == 1 and monitor.changed == 0
    assert monitor.collisions[0]["different_boolean_corners"] == 4

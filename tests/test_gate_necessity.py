"""The frozen tree reader needs exact weights, child order and parent tie-breaking."""

import numpy as np
import pytest

from fusionbench.eval.gate_necessity import model_outputs, native_tree
from fusionbench.eval.native_switch import tensors


def test_frozen_gate_keeps_incoming_prefix_and_downstream_update():
    pytest.importorskip("okapi.gate_lineage")
    from okapi.backend.backend import Backend
    from okapi.gate_lineage import tree_with_gate_cutoff
    from okapi.globals import set_postprocessing_function

    assert callable(tree_with_gate_cutoff)
    Backend.set_backend("numpy")
    set_postprocessing_function(np.asarray)

    def variable(i, *ops):
        return {"type": "ValueNode", "id": f"p{i}.npy", "children": list(ops)}

    def op(kind, children, **params):
        return {"type": kind, "children": children, **params}

    x = np.array(
        [
            [0.25, 0.75, 0.75, 0.25],
            [0.75, 0.25, 0.5, 0.75],
            [0.5, 0.5, 0.25, 0.5],
            [0.875, 0.125, 0.375, 0.75],
        ]
    )
    prefix = op("WeightedMeanNode", [variable(1)], weights=[0.25, 0.75])
    gate = op("CloseThresholdNode", [variable(2)], threshold=0.375, close=True)
    suffix = op("MaxNode", [variable(3)])
    tree = native_tree(variable(0, prefix, gate, suffix), tensors(x))
    a = 0.25 * x[:, 0] + 0.75 * x[:, 1]
    result = model_outputs(tree, [(1,)])
    expected = np.maximum(x[:, 3], a)
    np.testing.assert_array_equal(result["bypassed"].predict().ravel(), expected)
    for method, cut in [("intact", 0.375), ("cutoff_half", 0.5)]:
        kept = np.where(abs(a - cut) <= abs(x[:, 2] - cut), a, x[:, 2])
        np.testing.assert_array_equal(
            result[method].predict().ravel(), np.maximum(x[:, 3], kept)
        )
    assert result["intact"] is tree

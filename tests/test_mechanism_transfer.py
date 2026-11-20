"""Structural checks for the intervention used by the transfer experiment."""

from itertools import product

import numpy as np

from fusionbench.eval.mechanism_transfer import VIEW_SIZE, discover, render, target
from fusionbench.rules import eval_rule


def test_discovered_rule_reconstructs_binary_mechanism_and_diagonal_targets():
    bits = np.array(list(product([False, True], repeat=3)))
    scores = np.where(bits, 0.95, 0.05)
    _, _, _, pair, rule = discover(scores, target(bits))
    assert pair == (0, 1)
    assert rule["op"] == "min"
    assert rule["args"][1]["op"] == "max"
    original = eval_rule(scores[:, :, None], rule)
    aa, bb = scores.copy(), scores.copy()
    aa[:, 1] = aa[:, 0]
    bb[:, 0] = bb[:, 1]
    np.testing.assert_allclose(original, np.maximum(eval_rule(aa[:, :, None], rule),
                                                    eval_rule(bb[:, :, None], rule)))


def test_opposite_polarity_requires_signed_raw_view_exchange():
    bits = np.array([[True, False, True], [False, False, True]])
    noise = np.zeros((2, 3, VIEW_SIZE))
    raw = render(bits, noise, -1).reshape(2, 3, VIEW_SIZE)
    exchanged = raw.copy()
    exchanged[:, :2] = -raw[:, [1, 0]]
    swapped_bits = bits[:, [1, 0, 2]]
    np.testing.assert_allclose(exchanged.reshape(2, -1), render(swapped_bits, noise, -1))
    np.testing.assert_array_equal(target(bits), target(swapped_bits))

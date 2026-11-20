import numpy as np

from fusionbench.eval.blind_circuit import BITS, RawCircuit, cell_id, fit_rule, library


def test_search_matches_all_represented_boolean_functions_and_excludes_xor():
    tables, rules = library()
    z = 2 * BITS - 1
    assert len(np.unique(tables, axis=0)) == len(tables)
    for table, rule in zip(tables, rules, strict=True):
        np.testing.assert_array_equal(rule.score(z) > 0, table[cell_id(z)])
    for y, expected_errors in [
        (((BITS[:, 0] & BITS[:, 1]) | (BITS[:, 2] & BITS[:, 3])), 0),
        ((BITS[:, 0] ^ BITS[:, 1]), 4),
    ]:
        _, errors = fit_rule(z, y.astype(bool))
        assert errors == expected_errors


def test_compilation_preserves_signed_leaf_scores_and_drops_redundant_axes():
    rng = np.random.default_rng(23)
    matrix = rng.normal(size=(4, 4))
    offset = rng.normal(size=4)
    z = 2 * BITS - 1
    y = ((z[:, 0] < 0) | (z[:, 1] > 0)) & (z[:, 3] > 0)
    rule, errors = fit_rule(z, y)
    assert errors == 0
    net = RawCircuit.from_axes(rule, matrix, offset)
    assert net.weights.shape == (3, 4)
    raw = rng.normal(size=(127, 4))
    np.testing.assert_allclose(net.score(raw), rule.score((raw-offset) @ matrix.T), atol=1e-12)
    data = net.to_dict()
    restored = RawCircuit(np.asarray(data["weights"]), np.asarray(data["bias"]), data["reduction"])
    np.testing.assert_array_equal(net.score(raw), restored.score(raw))

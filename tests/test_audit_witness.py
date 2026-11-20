import numpy as np
from fusionbench.eval.audit_witness import permute_within


def test_permutation_keeps_whole_rows_and_score_bins():
    scores = np.repeat([-0.5, 0.5], 20)
    blocks = np.column_stack([np.arange(40), np.arange(40) ** 2])
    moved = permute_within(scores, blocks, [0], np.random.default_rng(7))
    assert np.any(moved[:, 0] != blocks[:, 0])
    np.testing.assert_array_equal(moved[:, 1], moved[:, 0] ** 2)
    for group in [scores < 0, scores > 0]:
        np.testing.assert_array_equal(np.sort(moved[group, 0]), blocks[group, 0])

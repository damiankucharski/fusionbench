import numpy as np
from fusionbench.eval.stratified_audit import sample_indices


def test_equal_budgets_unique_indices_and_quotas():
    p = np.tile(np.linspace(0, 1, 256), (20, 1))
    result, groups = sample_indices(p, (0.08, 0.42), 32, np.random.default_rng(4))
    for idx in result.values():
        assert idx.shape == (20, 32)
        assert all(len(np.unique(row)) == 32 for row in idx)
    for n in [8, 16, 32]:
        strata = np.take_along_axis(groups, result["fixed_allocation"][:, :n], axis=1)
        expected = np.broadcast_to(np.array([n // 2, n // 4, n // 4]), (20, 3))
        actual = np.stack([(strata == j).sum(1) for j in range(3)], axis=1)
        np.testing.assert_array_equal(actual, expected)


def test_exhausted_strata_fill_without_replacement():
    result, _ = sample_indices(np.full((3, 12), 0.5), (0.04, 0.42), 12, np.random.default_rng(2))
    for idx in result.values():
        np.testing.assert_array_equal(np.sort(idx, axis=1), np.tile(np.arange(12), (3, 1)))

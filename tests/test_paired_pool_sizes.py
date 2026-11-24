import numpy as np
import pytest

from fusionbench.interventions import duplicate_provider, paired_pool_sizes


@pytest.mark.parametrize("scenario", ["committee", "multiclass_committee", "needle", "deadweight"])
def test_nested_provider_pools_keep_observations(scenario):
    banks = paired_pool_sizes(scenario, [3, 6], seed=42, n_per_split=128)
    for split in ["fit", "test"]:
        small, large = banks[3], banks[6]
        x, wide = getattr(small, split + "_preds"), getattr(large, split + "_preds")
        np.testing.assert_array_equal(x, wide[:, :small.n_models])
        np.testing.assert_array_equal(getattr(small, split + "_y"), getattr(large, split + "_y"))
        cloned = duplicate_provider(small, provider=1, copies=2)
        expanded = getattr(cloned, split + "_preds")
        np.testing.assert_array_equal(expanded[:, -1], x[:, 1])
        np.testing.assert_array_equal(expanded[:, -2], x[:, 1])
        np.testing.assert_array_equal(getattr(cloned, split + "_oracle"), getattr(small, split + "_oracle"))
        if scenario in ("needle", "deadweight"):
            np.testing.assert_array_equal(getattr(small, split + "_oracle"), getattr(large, split + "_oracle"))

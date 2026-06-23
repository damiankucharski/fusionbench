"""Dataset save/load round-trip preserves predictions, labels and oracle."""

import numpy as np

from fusionbench.scenarios.committee import Committee
from fusionbench.store import load_dataset, save_dataset


def test_round_trip(tmp_path):
    ds = Committee(k=5).build(n_per_split=1000, seed=0)
    path = save_dataset(ds, tmp_path)
    back = load_dataset(path)

    assert back.name == ds.name
    assert back.task == ds.task
    assert back.model_ids == ds.model_ids
    np.testing.assert_allclose(back.fit_preds, ds.fit_preds, rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(back.test_y, ds.test_y)
    np.testing.assert_allclose(back.test_oracle, ds.test_oracle, rtol=1e-5, atol=1e-6)


def test_rows_sum_to_one():
    ds = Committee(k=4).build(n_per_split=500, seed=0)
    sums = ds.fit_preds.sum(axis=2)
    np.testing.assert_allclose(sums, 1.0, atol=1e-5)

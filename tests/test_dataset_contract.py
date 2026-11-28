from dataclasses import replace

import numpy as np
import pytest

from fusionbench.scenarios.committee import Committee
from fusionbench.types import Dataset


@pytest.fixture
def dataset():
    p = np.array([[[0.25, 0.75], [1, 0]], [[0.4, 0.6], [0, 1]]], dtype=np.float32)
    y = np.array([1, 0], dtype=np.int32)
    return Dataset("tiny", "binary", 2, ["first", "second"], p, y, p[:1], y[:1])


@pytest.mark.parametrize(
    "key,value",
    [
        ("task", "multilabel"),
        ("n_classes", 1),
        ("n_classes", 2.0),
        ("n_classes", 3),
        ("model_ids", []),
        ("model_ids", ["one", "one"]),
        ("fit_y", np.array([], dtype=int)),
        ("fit_y", np.array([[1], [0]])),
        ("test_y", np.array([1.0])),
        ("test_y", np.array([-1])),
        ("fit_y", np.array([2, 0], dtype=np.uint64)),
        ("fit_preds", np.ones((2, 2))),
        ("fit_preds", np.ones((2, 1, 2))),
        ("test_preds", np.ones((1, 2, 3))),
        ("test_preds", np.ones((2, 2, 2))),
        ("test_oracle", np.ones((1, 2, 1))),
    ],
)
def test_metadata_and_axes(dataset, key, value):
    with pytest.raises(ValueError, match=key):
        replace(dataset, **{key: value})


@pytest.mark.parametrize("field", ["fit_preds", "test_oracle"])
@pytest.mark.parametrize(
    "values,reason",
    [
        ((np.nan, 0.5), "finite"),
        ((np.inf, -np.inf), "finite"),
        ((-0.2, 1.2), "in .0, 1."),
        ((0.4, 0.5), "sum to 1"),
        (("0", "1"), "probabilities"),
    ],
)
def test_every_row_is_a_probability_vector(dataset, field, values, reason):
    if field == "test_oracle":
        invalid = np.array([values])
    else:
        invalid = dataset.fit_preds.astype(np.asarray(values).dtype)
        invalid[-1, -1] = values
    with pytest.raises(ValueError, match=f"{field}.*{reason}"):
        replace(dataset, **{field: invalid})


def test_arrays_keep_precision_and_storage(dataset):
    hard = np.array([[[1, 0, 0]], [[0, 0, 1]]], dtype=np.int8)
    labels = np.array([False, True])
    ds = Dataset(
        "three", "multiclass", np.int32(3), ["hard"], hard, labels, hard, labels
    )
    assert ds.fit_preds is hard and ds.test_y is labels

    # Noncontiguous oracle views and float32 row-sum roundoff remain valid.
    oracle = dataset.fit_preds[:, 0, :]
    oracle[0, 1] += 2e-6
    updated = replace(dataset, fit_oracle=oracle)
    assert updated.fit_preds is dataset.fit_preds and updated.fit_oracle is oracle


def test_custom_scenario_cannot_emit_an_invalid_oracle():
    class BrokenOracle(Committee):
        def bayes_oracle(self, preds, latent):
            return np.zeros((len(preds), self.n_classes))

    with pytest.raises(ValueError, match="fit_oracle.*sum to 1"):
        BrokenOracle().build(n_per_split=3, seed=9)

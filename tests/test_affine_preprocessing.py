from itertools import product

import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler

from fusionbench.methods.affine_preprocessing import fold_standardized_pca_affines


@pytest.mark.parametrize(
    ("with_mean", "with_std", "whiten"), list(product((False, True), repeat=3))
)
def test_fitted_pipeline_and_folded_multireadout_agree_on_new_rows(
    with_mean, with_std, whiten
):
    rng = np.random.default_rng(64)
    train = rng.normal(size=(91, 7)) * np.arange(1, 8) + 3
    train[:, -1] = -14
    scaler = StandardScaler(with_mean=with_mean, with_std=with_std).fit(train)
    pca = PCA(n_components=5, whiten=whiten, svd_solver="full").fit(
        scaler.transform(train)
    )
    weights, bias = rng.normal(size=(4, 5)), rng.normal(size=4)
    raw_weights, raw_bias = fold_standardized_pca_affines(scaler, pca, weights, bias)
    query = rng.normal(size=(32, 7)) + 10
    for x in [train, query]:
        transformed = pca.transform(scaler.transform(x))
        np.testing.assert_allclose(
            x @ raw_weights.T + raw_bias,
            transformed @ weights.T + bias,
            rtol=1e-12,
            atol=1e-12,
        )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_zero_variance_whitening_uses_fitted_dtype_epsilon(dtype):
    train = np.array([[-2, 0], [-1, 0], [1, 0], [2, 0]], dtype=dtype)
    scaler = StandardScaler().fit(train)
    pca = PCA(whiten=True, svd_solver="full").fit(scaler.transform(train))
    assert pca.explained_variance_[-1] == 0
    weights = np.array([0, 0.25], dtype=dtype)
    raw_weights, raw_bias = fold_standardized_pca_affines(scaler, pca, weights, 0.125)
    new_x = np.array([[7, 0], [-1, 1], [4, -2]], dtype=dtype)
    expected = pca.transform(scaler.transform(new_x)) @ weights + 0.125
    np.testing.assert_allclose(
        new_x @ raw_weights + raw_bias, expected, rtol=2 * np.finfo(dtype).eps
    )


def test_rejects_unfitted_incompatible_and_nonfinite_inputs():
    rng = np.random.default_rng(22)
    x = rng.normal(size=(16, 4))
    scaler = StandardScaler().fit(x)
    pca = PCA(n_components=2).fit(scaler.transform(x))
    with pytest.raises(NotFittedError):
        fold_standardized_pca_affines(StandardScaler(), pca, [1, 2], 0)
    with pytest.raises(ValueError, match="dimensions"):
        fold_standardized_pca_affines(StandardScaler().fit(x[:, :3]), pca, [1, 2], 0)
    with pytest.raises(ValueError, match="axis"):
        fold_standardized_pca_affines(scaler, pca, [1, 2, 3], 0)
    with pytest.raises(ValueError, match="bias"):
        fold_standardized_pca_affines(scaler, pca, np.ones((2, 2)), [1, 2, 3])
    for vector, offset in [([1, np.inf], 0), ([1, 2], np.nan)]:
        with pytest.raises(ValueError, match="finite"):
            fold_standardized_pca_affines(scaler, pca, vector, offset)

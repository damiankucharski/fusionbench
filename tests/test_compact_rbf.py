from itertools import product

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from fusionbench.methods.compact_rbf import compile_standardized_rbf


def fit_pipeline(x, y, scaler, basis):
    standardized = scaler.fit_transform(x)
    embedded = basis.fit_transform(standardized)
    readout = LogisticRegression(C=2).fit(embedded, y)
    return scaler, basis, readout


@pytest.mark.parametrize(("center", "scale"), list(product((False, True), repeat=2)))
@pytest.mark.parametrize(
    "settings,expected_gamma",
    [({}, 0.2), ({"gamma": 0.08}, 0.08), ({"kernel_params": {"gamma": 0.4}}, 0.4)],
)
def test_folded_rbf_matches_second_class_on_new_raw_rows(
    center, scale, settings, expected_gamma
):
    rng = np.random.default_rng(1984)
    train = rng.normal(size=(75, 5)) * [0.8, 2.0, 0.4, 2.5, 0.0] + [4, -1, 6, 1, 3]
    labels = np.where(train[:, 0] - train[:, 3] > 3, "upper", "lower")
    scaler, basis, readout = fit_pipeline(
        train,
        labels,
        StandardScaler(with_mean=center, with_std=scale),
        Nystroem(n_components=17, random_state=187, **settings),
    )
    net = compile_standardized_rbf(scaler, basis, readout)
    query = train[:20] + rng.normal(size=(20, 5))
    expected = readout.predict_proba(basis.transform(scaler.transform(query)))[:, 1]
    np.testing.assert_allclose(
        net.predict_positive(query), expected, atol=1e-12, rtol=1e-12
    )
    assert net.gamma == expected_gamma
    assert net.centers.shape == (17, 5) and net.weights.shape == (17,)
    # Deployment parameters are independent of subsequent refits or edits.
    basis.components_[:] = 0
    basis.normalization_[:] = 1
    readout.coef_[:] = 0
    np.testing.assert_allclose(
        net.predict_positive(query), expected, atol=1e-12, rtol=1e-12
    )


def test_rejects_incompatible_pipelines_and_invalid_query():
    rng = np.random.default_rng(9)
    x = rng.normal(size=(90, 4))
    y = np.arange(len(x)) % 2
    scaler, basis, readout = fit_pipeline(
        x, y, StandardScaler(), Nystroem(n_components=8, random_state=4)
    )
    with pytest.raises(NotFittedError):
        compile_standardized_rbf(StandardScaler(), basis, readout)
    with pytest.raises(ValueError, match="RBF"):
        compile_standardized_rbf(
            scaler, Nystroem(kernel="linear", n_components=8).fit(x), readout
        )
    with pytest.raises(ValueError, match="feature count"):
        compile_standardized_rbf(StandardScaler().fit(x[:, :3]), basis, readout)
    with pytest.raises(ValueError, match="dimensions"):
        compile_standardized_rbf(scaler, basis, LogisticRegression().fit(x, y))
    embedded = basis.transform(scaler.transform(x))
    with pytest.raises(ValueError, match="two classes"):
        compile_standardized_rbf(
            scaler, basis, LogisticRegression().fit(embedded, np.arange(90) % 3)
        )
    net = compile_standardized_rbf(scaler, basis, readout)
    with pytest.raises(ValueError, match="feature count"):
        net.predict_positive(x[:, :3])
    with pytest.raises(ValueError, match="finite"):
        net.predict_positive(np.full((3, 4), np.nan))

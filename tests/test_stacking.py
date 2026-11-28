import numpy as np
import pytest

from fusionbench.methods.stacking import StackingLogit, StackingLR


@pytest.mark.parametrize("stacker", (StackingLR, StackingLogit))
def test_missing_classes_preserve_canonical_columns_across_refits(stacker):
    x = np.random.default_rng(41).dirichlet([1, 1, 1], size=(18, 2))
    model = stacker().fit(x, np.tile([0, 2], 9), task="multiclass", n_classes=3)
    p = model.predict(x)
    assert p.shape == (18, 3)
    assert np.all(p[:, 1] <= 1e-11)
    np.testing.assert_allclose(p.sum(axis=1), 1)

    model.fit(x, np.full(18, 2), task="multiclass", n_classes=3)
    assert np.all(model.predict(x)[:, 2] >= 1 - 1e-11)
    model.fit(x, np.tile([0, 1, 2], 6), task="multiclass", n_classes=3)
    assert np.all(model.predict(x) > 0)
    with pytest.raises(ValueError, match="axes"):
        model.predict(x.reshape((18, 3, 2)))

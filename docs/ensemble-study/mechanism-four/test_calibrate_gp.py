"""Normal native adapter and cheap mean identities and deployed predictions."""
import numpy as np
import pytest
from calibrate_gp import fit_calibrated

from fusionbench.methods.averaging import SimpleAverage
from fusionbench.methods.okapi import OKAPIMethod


@pytest.mark.parametrize("classes", [2, 4])
@pytest.mark.parametrize("kind", ["gp", "mean"])
def test_adapter_forecast_matches_temperature_of_original(classes, kind):
    rng = np.random.default_rng(5830)
    x = rng.dirichlet(np.ones(classes), size=(90, 3))
    y = x.mean(1).argmax(1)
    task = "binary" if classes == 2 else "multiclass"
    base = OKAPIMethod(generations=2, n_restarts=1, n_seeds=0, fitness="brier") if kind == "gp" else SimpleAverage()
    base.fit(x, y, task=task, n_classes=classes)
    # Use labels matched to the fitted tree to force a nonidentity grid point.
    obs = base.predict(x).argmax(1)
    q = rng.dirichlet(np.ones(classes), size=(15, 3))
    q[0, :, :] = np.eye(classes)[0]
    before = base.predict(q)
    fitted = fit_calibrated(base, x, obs, groups=np.arange(len(x)) // 3, penalized=False)
    assert fitted.calibration_["beta"] == 2. and fitted.calibration_["n_eff"] == 30
    p = np.clip(before, 1e-6, 1 - 1e-6) ** 2
    p /= p.sum(1, keepdims=True)
    np.testing.assert_allclose(fitted.predict(q), p, rtol=0, atol=2e-12)
    np.testing.assert_array_equal(base.predict(q), before)


@pytest.mark.parametrize("classes", [2, 4])
def test_mean_identity_bitwise_even_with_zero_probabilities(classes):
    labels = np.arange(64) % classes
    x = np.tile(np.eye(classes)[labels][:, None, :], (1, 3, 1))
    mean = SimpleAverage().fit(x, labels, task="binary" if classes == 2 else "multiclass", n_classes=classes)
    fit = fit_calibrated(mean, x, labels)
    assert fit.calibration_["beta"] == 1.
    np.testing.assert_array_equal(fit.predict(x), mean.predict(x))

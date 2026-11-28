import numpy as np
import pytest

from fusionbench.scenarios.checkpoint_bank import (
    _residual_scales,
    generate_scene_pool,
    select_checkpoints,
    signal_provider_index,
)


def pool(share, law="equal"):
    return generate_scene_pool(
        17, law=law, n_fit_scenes=5, n_audit_scenes=3, scene_error_share=share
    )


def test_historical_coefficients_and_both_covariance_components():
    assert _residual_scales(None) == (0.5, 0.5, 0.85, 0.5)
    lag = 0.65 ** np.arange(8)
    cov = 0.5 + (0.85**2 + 0.5**2) * lag
    for share in (0, 0.17, np.float64(0.7), 1):
        u, v, h, j = _residual_scales(share)
        np.testing.assert_allclose(
            u * u + v * v + (h * h + j * j) * lag, cov, rtol=1e-15
        )
        np.testing.assert_allclose(u * u + h * h * lag, share * cov, rtol=1e-15)
    # Under the historical law, static and AR noise share different fractions.
    assert not np.isclose(0.5**2 / 0.5, 0.85**2 / (0.85**2 + 0.5**2))


def test_shared_draws_interpolate_endpoint_residuals_after_removing_signal():
    a, b = pool(0), pool(1)
    share = 0.37
    middle = pool(share)

    def residual(p, base):
        return np.log(p / (1 - p)) - base

    for name, block in middle.items():
        q = block.posterior[:, None, None]
        base = (1.3 / 1.8) * np.log(q / (1 - q))

        row, scene = residual(a[name].P, base), residual(b[name].P, base)
        np.testing.assert_allclose(
            residual(block.P, base),
            np.sqrt(1 - share) * row + np.sqrt(share) * scene,
            atol=2e-12,
        )
        # At full sharing, subtracting z gives the same logit residual on
        # every row of the scene, even though the probabilities vary with z.
        scene = scene.reshape(-1, 8, 8, 8)
        np.testing.assert_allclose(np.diff(scene, axis=1), 0, atol=2e-12)
        assert np.any(abs(np.diff(row.reshape(-1, 8, 8, 8), axis=1)) > 0.1)


@pytest.mark.parametrize("share", [0, 0.43, 1])
def test_share_leaves_labels_and_planted_provider_unchanged(share):
    original = generate_scene_pool(17, law="signal", n_fit_scenes=5, n_audit_scenes=3)
    equal, signal = pool(share), pool(share, "signal")
    index = signal_provider_index(17)
    other = np.arange(8) != index
    for name, base in original.items():
        for key in ("y", "posterior", "scene"):
            np.testing.assert_array_equal(getattr(base, key), getattr(equal[name], key))
            np.testing.assert_array_equal(
                getattr(base, key), getattr(signal[name], key)
            )
        np.testing.assert_array_equal(signal[name].P[:, index], base.P[:, index])
        np.testing.assert_array_equal(signal[name].P[:, other], equal[name].P[:, other])
        assert not np.array_equal(equal[name].P[:, other], base.P[:, other])
    frozen = select_checkpoints(signal["A"], 8)
    assert frozen[index] == 0
    np.testing.assert_array_equal(
        frozen[other], select_checkpoints(equal["A"], 8)[other]
    )


@pytest.mark.parametrize(
    "bad", [-0.01, 1.001, True, np.bool_(False), np.nan, np.inf, "0.5"]
)
def test_scene_error_share_rejects_inputs_outside_its_domain(bad):
    with pytest.raises(ValueError, match="scene_error_share"):
        pool(bad)

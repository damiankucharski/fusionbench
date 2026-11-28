import numpy as np
import pytest

from fusionbench.methods.environment_screen import EnvironmentScreenedOKAPI
from fusionbench.methods.native_pattern import RegularizedNativePatternOKAPI

pytest.importorskip("okapi.environment_stability")


def example():
    # Provider 1 changes performance by environment on the diagnostic half only.
    # Pooling both halves erases that environmental contrast.
    environment = np.tile(np.repeat([0, 1], 32), 2)
    context = np.tile(np.repeat([0, 1], 16), 4)
    y = np.arange(128) % 2
    good = np.where(y, 0.875, 0.125)
    drift = np.logical_xor(environment, np.arange(128) >= 64)
    p = np.column_stack((good, np.where(drift, 1 - good, good), 0.25 + 0.25 * context))
    return np.stack([1 - p, p], axis=2), y, environment, context


def test_diagnostic_subset_full_refit_and_compact_deployment():
    x, y, env, ctx = example()
    options = {"task": "binary", "n_classes": 2}
    method = EnvironmentScreenedOKAPI(seed=17).fit(
        x, y, **options, environments=env, contexts=ctx, screen_indices=np.arange(64)
    )
    assert (
        method.drop_ == 1 and method.screen_info_["reason"] == "unique_above_threshold"
    )
    np.testing.assert_array_equal(method.kept_indices_, [0, 2])
    assert method.model_.counts_["refit_rows"] == 128
    assert np.sum(method.screen_info_["cell_counts"]) == 64
    compact = np.ascontiguousarray(x[:, [0, 2], :])
    direct = RegularizedNativePatternOKAPI(
        seed=17, split_seed=0, gp_validation_top_k=8
    ).fit(compact, y, **options)
    np.testing.assert_array_equal(method.predict(x), direct.predict(compact))
    np.testing.assert_array_equal(method.predict_retained(compact), method.predict(x))
    with pytest.raises(ValueError, match="original provider"):
        method.predict(compact)
    with pytest.raises(ValueError, match="retained"):
        method.predict_retained(x)


def test_missing_metadata_abstains_and_matches_native():
    x, y, _, _ = example()
    options = {"task": "binary", "n_classes": 2}
    method = EnvironmentScreenedOKAPI().fit(x, y, **options)
    assert method.drop_ is None
    assert method.screen_info_["reason"] == "missing_environments"
    direct = RegularizedNativePatternOKAPI(
        seed=0, split_seed=0, gp_validation_top_k=8
    ).fit(x, y, **options)
    np.testing.assert_array_equal(method.predict_retained(x), direct.predict(x))


def test_raw_mode_ignores_context_and_single_remaining_provider_works():
    x, y, env, _ = example()
    x = x[:, :2, 1]  # The canonical two-dimensional binary input form.
    mask = np.arange(len(y)) < 64
    method = EnvironmentScreenedOKAPI(mode="raw").fit(
        x,
        y,
        task="binary",
        n_classes=2,
        environments=env,
        contexts=object(),
        screen_indices=mask,
    )
    assert method.drop_ == 1
    np.testing.assert_array_equal(method.screen_indices_, np.flatnonzero(mask))
    np.testing.assert_array_equal(method.predict_retained(x[:, :1])[:, 1], x[:, 0])
    np.testing.assert_array_equal(method.predict(x), method.predict_retained(x[:, :1]))


@pytest.mark.parametrize("rows", [[], [128], [-1], [0, 0], [0.5], [True], [[0, 1]]])
def test_invalid_diagnostic_subset(rows):
    x, y, env, ctx = example()
    with pytest.raises(ValueError, match="screen|source row"):
        EnvironmentScreenedOKAPI().fit(
            x,
            y,
            task="binary",
            n_classes=2,
            environments=env,
            contexts=ctx,
            screen_indices=rows,
        )


def test_source_ids_must_align_before_subsetting_and_classes_must_match():
    x, y, env, ctx = example()
    method = EnvironmentScreenedOKAPI()
    with pytest.raises(ValueError, match="environment ID"):
        method.fit(
            x,
            y,
            task="binary",
            n_classes=2,
            environments=env[:64],
            contexts=ctx,
            screen_indices=np.arange(64),
        )
    with pytest.raises(ValueError, match="context ID"):
        method.fit(
            x,
            y,
            task="binary",
            n_classes=2,
            environments=env,
            contexts=ctx[:64],
            screen_indices=np.arange(64),
        )
    with pytest.raises(ValueError, match="class axis|Class axis"):
        method.fit(x, y, task="multiclass", n_classes=4, environments=env, contexts=ctx)
    with pytest.raises(ValueError, match="before prediction"):
        method.predict(x)

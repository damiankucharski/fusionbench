import numpy as np
import pytest

from fusionbench.methods.transfer_pattern import TransferPatternOKAPI, validation_probes


def test_source_probes_preserve_rows_and_are_repeatable():
    x = np.random.default_rng(6).dirichlet([1, 2, 3], size=(120, 5))
    original = x.copy()
    swapped, warped = validation_probes(x, 44)
    changed = np.any(x != swapped, axis=2).sum(axis=1)
    assert set(changed) == {0, 1}
    np.testing.assert_allclose(warped.sum(2), 1.0)
    np.testing.assert_array_equal(warped.argmax(2), x.argmax(2))
    np.testing.assert_array_equal(x, original)
    for result, replay in zip((swapped, warped), validation_probes(x, 44), strict=True):
        np.testing.assert_array_equal(result, replay)


def test_transfer_deploys_one_tree_using_only_heldout_source_rows():
    pytest.importorskip("okapi.transfer_selection")
    from fusionbench.eval.suite_point_gp import predict_tree
    from fusionbench.methods.registry import default_methods

    assert isinstance(
        default_methods(["okapi_transfer_pattern"])[0], TransferPatternOKAPI
    )
    rng = np.random.default_rng(729)
    x = rng.dirichlet([1, 2], size=(128, 3))
    y = rng.integers(0, 2, size=128)
    method = TransferPatternOKAPI(seed=33).fit(
        x[:100], y[:100], task="binary", n_classes=2
    )
    indices = method.base_.validation_indices_
    for name, tree in method.discovery_trees_.items():
        for data, skill in zip(
            (x[indices], *validation_probes(x[indices], 33)),
            method.scores_[name],
            strict=True,
        ):
            loss = (
                np.square(predict_tree(tree, data) - np.eye(2)[y[indices]])
                .sum(1)
                .mean()
            )
            assert skill == pytest.approx(1 - 0.5 * loss)
    assert method.counts_["additional_score_calls"] == 9
    np.testing.assert_array_equal(
        method.predict(x[100:]), predict_tree(method.trees_[method.chosen_], x[100:])
    )

import numpy as np
import pytest

pytest.importorskip("okapi")
pytest.importorskip("torch")

from fusionbench.methods.okapi import OKAPIMethod
from fusionbench.scenarios.calibration import MulticlassCommittee
from fusionbench.scenarios.committee import Committee, Saturated


@pytest.mark.parametrize("multiclass", [MulticlassCommittee, Saturated])
def test_alternating_binary_multiclass_predict_keeps_model_outputs(multiclass):
    binary = Committee().build(n_per_split=96, seed=487)
    multi = multiclass().build(n_per_split=96, seed=487)
    kwargs = dict(seed=1207, n_restarts=1, generations=3, fitness="brier")
    a = OKAPIMethod(**kwargs).fit(
        binary.fit_preds, binary.fit_y, task=binary.task, n_classes=2
    )
    first = a.predict(binary.test_preds)
    b = OKAPIMethod(**kwargs).fit(
        multi.fit_preds, multi.fit_y, task=multi.task, n_classes=multi.n_classes
    )
    before = b.predict(multi.test_preds)
    np.testing.assert_array_equal(a.predict(binary.test_preds), first)
    np.testing.assert_array_equal(b.predict(multi.test_preds), before)
    np.testing.assert_allclose(before.sum(1), 1)


def test_fit_predict_restores_callers_postprocessor():
    from okapi.globals import postprocessing_function, set_postprocessing_function

    probe = np.array([[0.2, 0.9]])
    original = postprocessing_function._postprocessing_function
    try:
        set_postprocessing_function(np.negative)
        binary = Committee().build(n_per_split=48, seed=488)
        model = OKAPIMethod(
            seed=7,
            n_restarts=1,
            generations=1,
            n_seeds=1,
            fitness="brier",
            operator_point_mutation=True,
        ).fit(binary.fit_preds, binary.fit_y, task="binary", n_classes=2)
        np.testing.assert_array_equal(postprocessing_function(probe), -probe)
        assert model.okapi.operator_point_mutation
        model.predict(binary.test_preds)
        np.testing.assert_array_equal(postprocessing_function(probe), -probe)
    finally:
        set_postprocessing_function(original)

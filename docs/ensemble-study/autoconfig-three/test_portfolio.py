"""Replay parity of public actions against the archived studies, plus OOF contracts."""

import sys
from pathlib import Path

import numpy as np
import pytest

from fusionbench.methods import auto_configuration as ac

HERE = Path(__file__).resolve().parent
for folder in (HERE, HERE.parent / "mechanism-four", HERE.parent / "full-data-five"):
    sys.path.insert(0, str(folder))

from calibrate_gp import fit_calibrated
from interventions import fit_candidate
from portfolio import fit_actions


def _bank(classes, rows=42):
    rng = np.random.default_rng(517)
    x = rng.dirichlet(np.full(classes, 0.8), size=(rows + 10, 3))
    y = rng.integers(classes, size=rows)
    return x[:rows], y, x[rows:]


@pytest.mark.parametrize("classes", [2, 5])
def test_native_flat_and_calibration_match_reference(classes):
    x, y, q = _bank(classes)
    groups = np.arange(len(y)) // 3
    models = fit_actions(
        x, y, groups, ("flat", "previous8", "mean", "cal_mean", "cal_pull")
    )
    old = models["previous8"]
    assert (
        models["flat"].best_tree.structural_key() == old.trees_["flat"].structural_key()
    )
    np.testing.assert_array_equal(
        models["flat"].predict(q), old.component_predictions(q)["flat"]
    )
    old_pull = fit_candidate("pull_mean", x, y, seed=0, groups=groups)
    for key, base in (("cal_mean", models["mean"]), ("cal_pull", old_pull)):
        reference = fit_calibrated(base, x, y, groups)
        assert (
            models[key].best_tree.structural_key()
            == reference.best_tree.structural_key()
        )
        np.testing.assert_array_equal(models[key].predict(q), reference.predict(q))


def test_whole_scene_folds_and_rare_class_posterior():
    x, y, q = _bank(5, 21)
    y[:] = 0
    y[-1] = 3
    groups = np.repeat(["a", "b", "d", "f", "g", "z"], [2, 1, 3, 6, 1, 8])
    splits = list(ac._folds(y, groups))
    np.testing.assert_array_equal(
        np.sort(np.concatenate([held for _, held in splits])), np.arange(len(y))
    )
    for train, held in splits:
        assert not set(groups[train]) & set(groups[held])
    model = ac.ConfiguredOKAPI("source_cv").fit(
        x, y, groups=groups, task="multiclass", n_classes=5
    )
    assert model.selection_["n_units"] == 6
    assert model.selection_["oof_fits"] == 15
    assert sum(f["held_rows"] for f in model.selection_["folds"]) == len(y)
    assert [0] in [fold["train_classes"] for fold in model.selection_["folds"]]
    fitted = ac.fit_actions(x, y, groups, (model.action_,))[model.action_]
    np.testing.assert_array_equal(model.predict(q), fitted.predict(q))
    np.testing.assert_allclose(model.predict(q).sum(1), 1)
    assert model.predict(q).shape == (10, 5)


def test_single_unit_fallback_and_explicit_policy():
    x, y, q = _bank(2)
    choice, detail = ac.cv_choice(x, y, groups=np.ones(len(y)))
    assert choice == "mean" and detail["oof_fits"] == 0
    policy = {"schema_version": 1, "tree": {"action": "top_n"}, "actions": ["top_n"]}
    model = ac.ConfiguredOKAPI(policy).fit(x, y, task="binary", n_classes=2)
    assert model.action_ == "top_n" and model.selection_["path"] == []
    np.testing.assert_array_equal(
        model.predict(q), ac.fit_actions(x, y, actions=("top_n",))["top_n"].predict(q)
    )


def test_one_se_compares_group_means(monkeypatch):
    # The longest group alone would change the selected action under row weighting.
    groups = np.repeat([11, 12, 13, 14, 15, 16], [2, 2, 2, 2, 2, 90])
    p = np.ones((len(groups), 2)) * 0.1
    p[-90:] = 0.9
    x = np.stack([1 - p, p], axis=2)
    y = np.zeros(len(x), dtype=int)

    class Fixed:
        def __init__(self, mean):
            self.mean = mean

        def predict(self, x):
            return x[:, 0, :] if self.mean else np.tile([0.5, 0.5], (len(x), 1))

    def mock_fit(x, y, groups=None):
        return {name: Fixed(mean=(name == "mean")) for name in ac.ACTIONS}

    monkeypatch.setattr(ac, "fit_actions", mock_fit)
    chosen, detail = ac.cv_choice(x, y, groups)
    mean = (5 * 0.02 + 1.62) / 6
    assert chosen == "mean"
    assert detail["oof_brier"]["mean"] == pytest.approx(mean)
    assert detail["paired_se"]["top_n"] == pytest.approx(
        np.std([0.48] * 5 + [-1.12], ddof=1) / np.sqrt(6)
    )
    assert ac.cv_choice(x, y)[0] == "top_n"

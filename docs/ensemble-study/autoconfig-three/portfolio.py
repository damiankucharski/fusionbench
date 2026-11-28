"""Deployable source-only candidates and separate study comparators."""

import sys
from pathlib import Path

from fusionbench.methods.auto_configuration import ACTIONS, cv_choice
from fusionbench.methods.auto_configuration import fit_actions as _fit_actions
from fusionbench.methods.native_pattern import RegularizedNativePatternOKAPI
from fusionbench.methods.stacking import StackingLR

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "full-data-five"))
from engine import fit_full, fit_model

CONTROLS = ("lr", "full", "previous8")
__all__ = ["ACTIONS", "CONTROLS", "cv_choice", "fit_actions"]


def fit_actions(x, y, groups=None, actions=ACTIONS):
    """Fit selected actions and optional fixed comparators on the same source bank."""
    actions = tuple(actions)
    models = _fit_actions(
        x, y, groups, [name for name in actions if name not in CONTROLS]
    )
    if "lr" in actions:
        models["lr"] = fit_model(StackingLR(c_value=1.0), x, y)
    if "full" in actions:
        models["full"] = fit_full(x, y)
    if "previous8" in actions:
        models["previous8"] = fit_model(
            RegularizedNativePatternOKAPI(seed=0, split_seed=0, gp_validation_top_k=8),
            x,
            y,
        )
    return {name: models[name] for name in actions}

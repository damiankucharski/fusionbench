"""Method registry.

``oracle`` is not a fitted method; it is the stored analytic Bayes ceiling and is
handled directly by the runner. The factory list here is the set of *competing*
fusion methods.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from fusionbench.methods.averaging import (
    BestSingle,
    SimpleAverage,
    TopK,
    TopN,
    ValWeighted,
)
from fusionbench.methods.base import Method
from fusionbench.methods.nsga3 import NSGA3
from fusionbench.methods.okapi import OKAPIMethod
from fusionbench.methods.operators import OpClose, OpFar, OpMax, OpMin
from fusionbench.methods.stacking import StackingLogit, StackingLR
from fusionbench.methods.tpot import TPOTMethod

FACTORIES: dict[str, Callable[[], Method]] = {
    "best_single": BestSingle,
    "simple_average": SimpleAverage,
    "top_n": TopN,
    "top_5": lambda: TopK(5),
    "val_weighted": ValWeighted,
    "op_min": OpMin,
    "op_max": OpMax,
    "op_far": OpFar,
    "op_close": OpClose,
    "stacking_lr": StackingLR,
    "stacking_logit": StackingLogit,
    "nsga3": NSGA3,
    "tpot": TPOTMethod,
    "okapi": OKAPIMethod,
}

DEFAULT_METHODS = ("simple_average", "top_n", "stacking_lr")

# All methods available from a base installation.
CORE = [
    "best_single",
    "simple_average",
    "top_5",
    "top_n",
    "val_weighted",
    "op_min",
    "op_max",
    "op_far",
    "op_close",
    "stacking_lr",
    "stacking_logit",
]


def comparison_names(names: list[str] | None) -> list[str] | None:
    """Add the three inexpensive references when the CLI requests OKAPI."""
    if names is None:
        return None
    result = list(dict.fromkeys(names))
    if "okapi" in result:
        for reference in DEFAULT_METHODS:
            if reference not in result:
                result.append(reference)
    return result


def default_methods(
    names: Sequence[str] | None = None, only_available: bool = True
) -> list[Method]:
    """Create the three default references, or the explicitly named methods."""
    if names is None:
        names = DEFAULT_METHODS
    methods = []
    for name in names:
        method = FACTORIES[name]()
        if only_available and not type(method).available():
            continue
        methods.append(method)
    return methods

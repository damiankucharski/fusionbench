"""Method registry.

``oracle`` is not a fitted method; it is the stored analytic Bayes ceiling and is
handled directly by the runner. The factory list here is the set of *competing*
fusion methods.
"""

from __future__ import annotations

from collections.abc import Callable

from fusionbench.methods.averaging import BestSingle, SimpleAverage, TopK, ValWeighted
from fusionbench.methods.base import Method
from fusionbench.methods.clr import StackingCLR
from fusionbench.methods.lattice_brier import ExactLatticeBrier
from fusionbench.methods.nsga3 import NSGA3
from fusionbench.methods.okapi import OKAPIMethod
from fusionbench.methods.operators import OpClose, OpFar, OpMax, OpMin
from fusionbench.methods.shrinkage import ShrunkOKAPI
from fusionbench.methods.stacking import StackingLogit, StackingLR
from fusionbench.methods.tpot import TPOTMethod

FACTORIES: dict[str, Callable[[], Method]] = {
    "best_single": BestSingle,
    "simple_average": SimpleAverage,
    "top_5": lambda: TopK(5),
    "val_weighted": ValWeighted,
    "op_min": OpMin,
    "op_max": OpMax,
    "op_far": OpFar,
    "op_close": OpClose,
    "stacking_lr": StackingLR,
    "stacking_clr": StackingCLR,
    "stacking_logit": StackingLogit,
    "nsga3": NSGA3,
    "tpot": TPOTMethod,
    "okapi": OKAPIMethod,
    "okapi_shrunk": ShrunkOKAPI,
    "lattice_brier_exact": ExactLatticeBrier,
}

# Methods considered "core" (no heavy optional deps) — always run locally.
CORE = ["best_single", "simple_average", "top_5", "val_weighted",
        "op_min", "op_max", "op_far", "op_close",
        "stacking_lr", "stacking_logit"]


def default_methods(names: list[str] | None = None, only_available: bool = True
                    ) -> list[Method]:
    # The exact lattice oracle is a binary-only, exponential-width comparator.
    # Keep it opt-in on mixed suites.
    names = names or [name for name in FACTORIES if name != "lattice_brier_exact"]
    methods = []
    for name in names:
        method = FACTORIES[name]()
        if only_available and not type(method).available():
            continue
        methods.append(method)
    return methods

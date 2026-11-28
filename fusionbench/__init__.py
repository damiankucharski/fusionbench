"""Generate controlled classification datasets for prediction fusion.

Scenarios produce labeled rows of provider probabilities. Use them to evaluate
fusion methods, compare generator settings, and study generalization under
controlled changes in the data distribution.
"""

from importlib import import_module

from fusionbench._version import __version__
from fusionbench.scenarios import (
    Scenario,
    generate_dataset,
    get_scenario,
    list_scenarios,
    register_scenario,
)
from fusionbench.types import Dataset, Task

__all__ = [
    "Dataset",
    "Scenario",
    "Task",
    "__version__",
    "evaluate_dataset",
    "generate_dataset",
    "get_scenario",
    "list_scenarios",
    "load_dataset",
    "load_reference_dataset",
    "register_scenario",
    "run_suite",
    "save_dataset",
    "summarize_suite",
]


def __getattr__(name):
    """Load filesystem and reference-data helpers on first access."""
    if name in ("load_dataset", "save_dataset"):
        module = "fusionbench.store"
    elif name in ("evaluate_dataset", "run_suite", "summarize_suite"):
        module = "fusionbench.eval.runner"
    elif name == "load_reference_dataset":
        module = "fusionbench.reference"
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module), name)
    globals()[name] = value
    return value

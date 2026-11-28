"""Scenario generators (base-model pathologies)."""

from fusionbench.scenarios.base import Scenario
from fusionbench.scenarios.registry import (
    SCENARIOS,
    default_suite,
    generate_dataset,
    get_scenario,
    list_scenarios,
    register_scenario,
)

__all__ = [
    "SCENARIOS",
    "Scenario",
    "default_suite",
    "generate_dataset",
    "get_scenario",
    "list_scenarios",
    "register_scenario",
]

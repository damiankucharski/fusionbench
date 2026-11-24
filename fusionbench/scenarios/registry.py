"""Discover, configure and extend the controlled prediction-fusion scenarios."""

from __future__ import annotations

import inspect
import json
from threading import Lock
from types import MappingProxyType

import numpy as np

from fusionbench.scenarios.base import Scenario
from fusionbench.scenarios.calibration import Miscalibration, MulticlassCommittee
from fusionbench.scenarios.committee import Committee, DeadWeight, Needle, Saturated
from fusionbench.scenarios.compositional_switch import CompositionalSwitch
from fusionbench.scenarios.correlated import CalibratedSuppressor, CorrelatedRedundancy
from fusionbench.scenarios.gates import (
    Compound,
    HardRescue,
    HardVeto,
    Parity,
    TwoPathways,
    WeakestLink,
)
from fusionbench.scenarios.mixture import GatedExperts
from fusionbench.scenarios.multiclass_provider_panel import MulticlassProviderPanel
from fusionbench.scenarios.native_switch import NativeCloseSwitch, NativeSwitch
from fusionbench.scenarios.polarity import PolarityFault
from fusionbench.scenarios.shared_confusion_panel import SharedConfusionPanel, SimilarClasses38
from fusionbench.types import Dataset

STANDARD_V1_SCENARIOS = MappingProxyType(
    {
        "committee": Committee,
        "deadweight": DeadWeight,
        "needle": Needle,
        "saturated": Saturated,
        "correlated": CorrelatedRedundancy,
        "calibrated_suppressor": CalibratedSuppressor,
        "polarity_fault": PolarityFault,
        "hardveto": HardVeto,
        "hardrescue": HardRescue,
        "weakest_link": WeakestLink,
        "two_pathways": TwoPathways,
        "parity": Parity,
        "compound": Compound,
        "gated_experts": GatedExperts,
        "native_switch": NativeSwitch,
        "compositional_switch": CompositionalSwitch,
        "native_close_switch": NativeCloseSwitch,
        "miscalibration": Miscalibration,
        "multiclass_committee": MulticlassCommittee,
    }
)

STANDARD_SCENARIOS = MappingProxyType({**STANDARD_V1_SCENARIOS, "similar_classes_38": SimilarClasses38})

SCENARIOS: dict[str, type[Scenario]] = {
    **STANDARD_SCENARIOS,
    "multiclass_provider_panel": MulticlassProviderPanel,
    "shared_confusion_panel": SharedConfusionPanel,
}

BUILTIN_IDS = tuple(SCENARIOS)
_BUILTINS = frozenset(SCENARIOS.values())
_registration_lock = Lock()


def list_scenarios() -> tuple[str, ...]:
    """Return registered scenario IDs in insertion order, including local extensions."""
    return tuple(SCENARIOS)


def register_scenario(name: str, scenario_cls: type[Scenario]) -> None:
    """Register a concrete Scenario subclass under an unused ID in this process."""
    if not isinstance(name, str) or not name or name != name.strip():
        raise ValueError("Use a nonempty scenario ID without surrounding whitespace")
    if not isinstance(scenario_cls, type) or not issubclass(scenario_cls, Scenario):
        raise TypeError("scenario_cls must be a Scenario subclass")
    if inspect.isabstract(scenario_cls):
        raise TypeError(
            "Register a concrete scenario implementing sample, model_ids and n_classes"
        )
    with _registration_lock:
        if name in SCENARIOS:
            raise ValueError(f"Scenario ID {name!r} is already registered")
        SCENARIOS[name] = scenario_cls


def _numpy_json(value):
    if isinstance(value, (np.ndarray, np.generic)):
        return value.tolist()
    raise TypeError(
        f"Cannot record scenario parameter of type {type(value).__name__} in JSON"
    )


def _portable_parameters(params):
    """Snapshot settings in the JSON format used by save_dataset."""
    return json.loads(json.dumps(params, default=_numpy_json, allow_nan=False))


def _validated_defaults(scenario_id, params):
    """Check built-in constructor names and return their declared defaults."""
    cls = SCENARIOS[scenario_id]
    if cls not in _BUILTINS:
        # Extensions may implement their own dynamic keyword arguments.
        return None
    declared = {
        key: param
        for parent in reversed(cls.__mro__)
        for key, param in inspect.signature(parent.__init__).parameters.items()
        if key != "self"
        and param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)
    }
    if extra := params.keys() - declared.keys():
        raise ValueError(
            f"Unknown constructor knob for {scenario_id}: {', '.join(sorted(extra))}. "
            f"Available: {', '.join(sorted(declared))}"
        )
    return {key: p.default for key, p in declared.items() if p.default is not p.empty}


def get_scenario(scenario_id: str, **params) -> Scenario:
    """Create a configured scenario whose build() records generation provenance.

    Built-ins accept their declared constructor options, including inherited
    names. Registered extensions receive their own keyword options. Use JSON
    settings for reproducible datasets. NumPy scalars and arrays are recorded
    as JSON numbers and arrays while constructors receive their original values.
    """
    try:
        cls = SCENARIOS[scenario_id]
    except KeyError:
        raise ValueError(
            f"Unknown scenario {scenario_id!r}. Available IDs: {', '.join(SCENARIOS)}"
        ) from None
    defaults = _validated_defaults(scenario_id, params)
    config = {"scenario_id": scenario_id, "parameters": _portable_parameters(params)}
    if defaults is not None:
        config["effective_parameters"] = _portable_parameters({**defaults, **params})
    scenario = cls(**params)
    config["knobs"] = _portable_parameters(scenario.knobs)
    scenario._registry_metadata = config
    return scenario


def generate_dataset(
    scenario_id: str, *, n_per_split: int = 50_000, seed: int = 0, **params
) -> Dataset:
    """Generate a matched fit/test Dataset with a recorded seed and configuration.

    The parameters entry in Dataset.meta contains explicit caller options.
    Built-ins also record effective_parameters including constructor defaults.
    """
    return get_scenario(scenario_id, **params).build(n_per_split=n_per_split, seed=seed)


def default_suite() -> list[Scenario]:
    """Instantiate the current standard regression suite."""
    return [get_scenario(name) for name in STANDARD_SCENARIOS]

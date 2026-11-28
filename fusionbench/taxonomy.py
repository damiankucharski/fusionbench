"""Mechanism and evaluation-purpose assignments for the fixed standard-v6 panel."""

from collections.abc import Mapping
from math import fsum, isfinite
from numbers import Real
from types import MappingProxyType

from fusionbench.scenarios.registry import SCENARIOS, STANDARD_V6_SCENARIOS

TAXONOMY_VERSION = "mechanisms-v1"
TAXONOMY_PRESET = "standard-v6"

MECHANISM_GROUPS = MappingProxyType(
    {
        "evidence_calibration": ("committee", "miscalibration", "multiclass_committee"),
        "pool_selection": ("deadweight", "needle", "redundant_pool", "stratified_pool"),
        "error_dependence": (
            "saturated",
            "correlated",
            "calibrated_suppressor",
            "polarity_fault",
            "similar_classes_38",
        ),
        "structured_rules": (
            "hardveto",
            "hardrescue",
            "weakest_link",
            "two_pathways",
            "parity",
            "compound",
            "native_switch",
            "native_close_switch",
            "compositional_switch",
        ),
        "context_routing": ("gated_experts", "rare_specialist", "selective_fallback"),
        "transfer_shift": (
            "environment_shortcut",
            "cloned_shortcut",
            "rare_shortcut_noise",
        ),
        "grouped_sampling": (
            "grouped_rare_event",
            "shared_error_bloc",
            "rare_scene_specialist",
        ),
    }
)

# Purpose concerns the scientific use of the default generator. Tags can overlap.
PURPOSE_TAGS = MappingProxyType(
    {
        "capability_probe": (
            "committee",
            "multiclass_committee",
            "calibrated_suppressor",
            "gated_experts",
            "hardveto",
            "hardrescue",
            "weakest_link",
            "two_pathways",
            "parity",
            "compound",
            "native_switch",
            "native_close_switch",
            "compositional_switch",
            "rare_specialist",
        ),
        "operational_stress": (
            "deadweight",
            "needle",
            "saturated",
            "correlated",
            "miscalibration",
            "calibrated_suppressor",
            "polarity_fault",
            "similar_classes_38",
            "rare_specialist",
            "redundant_pool",
            "selective_fallback",
            "environment_shortcut",
            "cloned_shortcut",
            "rare_shortcut_noise",
            "stratified_pool",
            "grouped_rare_event",
            "shared_error_bloc",
            "rare_scene_specialist",
        ),
        "reference": (
            "committee",
            "multiclass_committee",
            "redundant_pool",
            "stratified_pool",
        ),
    }
)

# Illustrative priorities over scientific questions. Neither profile is a frequency estimate.
APPLICATION_PROFILES = MappingProxyType(
    {
        "multi_site": MappingProxyType(
            {"transfer_shift": 0.50, "grouped_sampling": 0.30, "context_routing": 0.20}
        ),
        "rule_design": MappingProxyType(
            {"structured_rules": 0.75, "context_routing": 0.25}
        ),
    }
)


def _names(items, catalog, label):
    names = (items,) if isinstance(items, str) else tuple(items)
    if not names or len(set(names)) != len(names):
        raise ValueError(f"Select distinct, nonempty {label}")
    unknown = set(names).difference(catalog)
    if unknown:
        raise ValueError(f"Unknown {label}: {unknown}")
    return names


def _relative(weights, catalog, label):
    if not isinstance(weights, Mapping):
        raise TypeError(f"{label} weights must map names to nonnegative finite numbers")
    names = _names(weights, catalog, label)
    if any(
        isinstance(weights[k], bool)
        or not isinstance(weights[k], Real)
        or not isfinite(weights[k])
        or weights[k] < 0
        for k in names
    ):
        raise ValueError(f"{label} weights must be nonnegative finite numbers")
    maximum = max(weights.values())
    if maximum == 0:
        raise ValueError(f"At least one {label} must have positive weight")
    # Scale before adding to support large finite relative weights.
    return {k: float(weights[k] / maximum) for k in names if weights[k] > 0}


def scenario_weights(*, groups=None, purposes=None, scenarios=None, group_weights=None):
    """Return normalized scenario weights. Keys can also select scenarios for a benchmark.

    The default gives each standard-v6 scenario equal mass. An explicit scenario list or
    mapping may include local registered generators. Each filter is a union of names; filters
    intersect. A scenario mapping supplies relative scenario weights. A group-weight mapping
    limits eligibility to its positive groups and sets their relative total masses. Within
    each represented group, selected scenarios share its mass in proportion to their weights.
    Empty groups drop out before normalization. No scenarios selected raises ValueError.
    """
    if isinstance(scenarios, Mapping):
        weights = _relative(scenarios, SCENARIOS, "scenarios")
    else:
        names = (
            STANDARD_V6_SCENARIOS
            if scenarios is None
            else _names(scenarios, SCENARIOS, "scenarios")
        )
        weights = dict.fromkeys(names, 1.0)
    for items, memberships, label in (
        (groups, MECHANISM_GROUPS, "groups"),
        (purposes, PURPOSE_TAGS, "purposes"),
    ):
        if items is not None:
            selected = _names(items, memberships, label)
            allowed = {s for key in selected for s in memberships[key]}
            weights = {s: value for s, value in weights.items() if s in allowed}
    if group_weights is not None:
        shares = _relative(group_weights, MECHANISM_GROUPS, "groups")
        allocated = {}
        for key, share in shares.items():
            group = {s: weights[s] for s in MECHANISM_GROUPS[key] if s in weights}
            total = fsum(group.values())
            allocated.update({s: share * (value / total) for s, value in group.items()})
        weights = {s: allocated[s] for s in weights if s in allocated}
    if not weights:
        raise ValueError(
            "The requested filters select no scenarios with positive weight"
        )
    total = fsum(weights.values())
    return {s: value / total for s, value in weights.items()}

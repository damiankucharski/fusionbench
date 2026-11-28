"""Mechanism groups stay exhaustive and benchmarking choices have explicit weights."""

from collections import Counter
from math import fsum

import pytest

from fusionbench import register_scenario
from fusionbench.scenarios.registry import SCENARIOS, STANDARD_V6_SCENARIOS
from fusionbench.taxonomy import (
    APPLICATION_PROFILES,
    MECHANISM_GROUPS,
    PURPOSE_TAGS,
    TAXONOMY_PRESET,
    TAXONOMY_VERSION,
    scenario_weights,
    weighted_score,
)


def test_v1_partition_is_complete_and_purpose_tags_overlap():
    flat = [s for members in MECHANISM_GROUPS.values() for s in members]
    assert Counter(flat) == Counter(STANDARD_V6_SCENARIOS.keys())
    assert len(flat) == 30 and len(MECHANISM_GROUPS) == 7
    assert (TAXONOMY_VERSION, TAXONOMY_PRESET) == ("mechanisms-v1", "standard-v6")
    all_tags = [s for members in PURPOSE_TAGS.values() for s in members]
    assert set(all_tags) == set(flat) and len(all_tags) > len(flat)
    assert all(len(set(members)) == len(members) for members in PURPOSE_TAGS.values())
    with pytest.raises(TypeError):
        MECHANISM_GROUPS["structured_rules"] = ()
    with pytest.raises(TypeError):
        APPLICATION_PROFILES["multi_site"]["transfer_shift"] = 0


def test_scenario_and_group_macro_have_different_declared_totals():
    scenario = scenario_weights()
    group = scenario_weights(group_weights=dict.fromkeys(MECHANISM_GROUPS, 1))
    assert list(scenario) == list(STANDARD_V6_SCENARIOS)
    assert set(scenario.values()) == {1 / 30}
    assert all(
        fsum(group[s] for s in members) == pytest.approx(1 / 7)
        for members in MECHANISM_GROUPS.values()
    )
    scores = {s: int(s in MECHANISM_GROUPS["structured_rules"]) for s in scenario}
    assert weighted_score(scores, scenario) == pytest.approx(9 / 30)
    assert weighted_score(scores, group) == pytest.approx(1 / 7)


def test_application_profiles_and_intersection():
    for profile in APPLICATION_PROFILES.values():
        weights = scenario_weights(group_weights=profile)
        mass = {k: fsum(weights[s] for s in MECHANISM_GROUPS[k]) for k in profile}
        assert mass == pytest.approx(dict(profile))
    got = scenario_weights(
        groups=("structured_rules", "transfer_shift"),
        purposes=("capability_probe", "reference"),
    )
    assert set(got) == set(MECHANISM_GROUPS["structured_rules"])
    got = scenario_weights(purposes=("reference", "operational_stress"))
    assert set(got) == set(PURPOSE_TAGS["reference"]) | set(
        PURPOSE_TAGS["operational_stress"]
    )
    assert len(set(got.values())) == 1


def test_weighted_subset_uses_within_group_ratios_and_drops_empty_groups():
    weights = scenario_weights(
        scenarios={
            "committee": 2,
            "miscalibration": 1,
            "gated_experts": 9,
            "parity": 0,
        },
        group_weights={
            "evidence_calibration": 3,
            "context_routing": 1,
            "structured_rules": 7,
        },
    )
    assert weights == pytest.approx(
        {"committee": 0.5, "miscalibration": 0.25, "gated_experts": 0.25}
    )
    assert weighted_score(
        {
            "committee": 0.1,
            "miscalibration": 0.2,
            "gated_experts": 0.4,
            "needle": float("nan"),
        },
        weights,
    ) == pytest.approx(0.2)
    with pytest.raises(ValueError, match="Missing"):
        weighted_score({"committee": 0.1}, weights)
    with pytest.raises(ValueError, match="finite"):
        weighted_score({"committee": float("nan")}, {"committee": 1})


def test_registered_local_case_is_eligible_only_by_explicit_name():
    name = "taxonomy_test_custom"
    register_scenario(name, SCENARIOS["committee"])
    try:
        assert scenario_weights(scenarios={name: 9, "committee": 1}) == pytest.approx(
            {name: 0.9, "committee": 0.1}
        )
        assert name not in scenario_weights()
        assert scenario_weights(
            scenarios=[name, "committee"], groups="evidence_calibration"
        ) == {"committee": 1}
    finally:
        del SCENARIOS[name]


def test_extreme_relative_units_survive_restriction_to_only_small_weights():
    assert scenario_weights(
        scenarios={"committee": 1e300, "needle": 1e-300}, groups="pool_selection"
    ) == {"needle": 1}
    assert scenario_weights(
        scenarios=["needle"],
        group_weights={"evidence_calibration": 1e300, "pool_selection": 1e-300},
    ) == {"needle": 1}
    assert scenario_weights(scenarios={"committee": 1e308, "needle": 1e308}) == {
        "committee": 0.5,
        "needle": 0.5,
    }


@pytest.mark.parametrize(
    "kwargs",
    [
        {"groups": "unknown"},
        {"scenarios": []},
        {"purposes": ["reference", "reference"]},
        {"scenarios": ["parity"], "groups": "transfer_shift"},
        {"scenarios": {"needle": True}},
        {"group_weights": {"pool_selection": -1}},
        {"group_weights": {"not_a_group": 0}},
        {"group_weights": {"pool_selection": float("nan")}},
        {"scenarios": {"needle": float("inf")}},
        {"group_weights": {"pool_selection": 0}},
    ],
)
def test_invalid_or_empty_selections_fail(kwargs):
    with pytest.raises((TypeError, ValueError)):
        scenario_weights(**kwargs)

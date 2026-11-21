"""Scenario registry and the default benchmark suite.

The suite spans the impartial landscape:
- conditionally-independent (logit-stacker near-oracle): Committee, DeadWeight,
  Needle, MulticlassCommittee;
- correlated / redundant (weighted-linear wins): CorrelatedRedundancy, Saturated;
- genuinely nonlinear logical gates (OKAPI min/max & TPOT win, linear fails):
  HardVeto, HardRescue, Compound;
- input-dependent mixture (TPOT wins): GatedExperts;
- calibration (logit recalibration wins): Miscalibration.
"""

from __future__ import annotations

from fusionbench.scenarios.base import Scenario
from fusionbench.scenarios.calibration import Miscalibration, MulticlassCommittee
from fusionbench.scenarios.committee import Committee, DeadWeight, Needle, Saturated
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
from fusionbench.scenarios.polarity import PolarityFault

SCENARIOS: dict[str, type[Scenario]] = {
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
    "miscalibration": Miscalibration,
    "multiclass_committee": MulticlassCommittee,
}


def default_suite() -> list[Scenario]:
    """Instantiate one canonical instance of every scenario."""
    return [cls() for cls in SCENARIOS.values()]

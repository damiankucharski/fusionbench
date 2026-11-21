"""Mixture-of-experts pathology (input-dependent routing; TPOT-favouring).

A latent region ``z`` decides which expert is informative about the label, and a
separate *selector* model reveals ``z``. The Bayes optimum is an input-dependent
convex mixture ``(1-S) * E1 + S * E2`` — neither a fixed weighted average nor a
confidence-based min/max/selection. Tree-based AutoML (which can split on the
selector) is expected to win; OKAPI's fixed-weight and confidence-routing
operators only partially capture it. Included as an honest counterexample where
OKAPI is *not* the right tool.
"""

from __future__ import annotations

import numpy as np

from fusionbench.scenarios import calibrated as cal
from fusionbench.scenarios.base import Scenario


class GatedExperts(Scenario):
    """Generate two regional specialists and an observed noisy selector.

    Jacobs et al. (1991) trained a gate to allocate inputs among experts.
    In this generator, a calibrated selector estimates the probability of the
    active region, giving a known input-dependent mixture for comparison.
    See jacobs1991adaptive in docs/ensemble-study/literature/README.md.
    """

    task = "binary"
    expected_winners = ["okapi", "tpot"]
    oracle_description = "input-dependent mixture: (1-S)*E1 + S*E2"

    def __init__(self, acc_select: float = 0.92, acc_expert: float = 0.90, **kw):
        super().__init__(acc_select=acc_select, acc_expert=acc_expert, **kw)
        self.acc_select, self.acc_expert = acc_select, acc_expert

    @property
    def n_classes(self):
        return 2

    def model_ids(self):
        return ["selector", "expert_a", "expert_b"]

    def _generate(self, n, rng):
        z = rng.integers(0, 2, n)          # region
        y = rng.integers(0, 2, n)          # label (independent of region)
        r1 = rng.integers(0, 2, n)         # distractor target for inactive expert
        r2 = rng.integers(0, 2, n)

        selector = cal.to_binary_proba(cal.binary_calibrated(z, self.acc_select, rng))
        # Expert A informative when z==0, otherwise confidently about a distractor.
        ea_target = np.where(z == 0, y, r1)
        eb_target = np.where(z == 1, y, r2)
        expert_a = cal.to_binary_proba(cal.binary_calibrated(ea_target, self.acc_expert, rng))
        expert_b = cal.to_binary_proba(cal.binary_calibrated(eb_target, self.acc_expert, rng))
        preds = np.stack([selector, expert_a, expert_b], axis=1)
        return preds, y, {}

    def bayes_oracle(self, preds, latent):
        p1 = preds[:, :, 1]
        s, ea, eb = p1[:, 0], p1[:, 1], p1[:, 2]
        return cal.to_binary_proba((1.0 - s) * ea + s * eb)

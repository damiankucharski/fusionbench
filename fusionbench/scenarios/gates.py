"""Logical-gate pathologies (genuinely nonlinear optima).

The label is a *logical function* of independent latent conditions, and each base
model is a calibrated detector of one condition. Because the label is determined
jointly, the detectors are conditionally *dependent* given the label, so the
Bayes optimum is a nonlinear interaction (a product for AND, a noisy-OR for OR)
— **not** representable as a (weighted) sum of logits. Hence even a logit-space
stacker is suboptimal, while gating methods (OKAPI's min/max, tree-based AutoML)
approximate the optimum well.

The generators describe conditions and Boolean logic only; the words
min/max/veto never enter the data-generating code. The veto/rescue reading of
the optimum is an emergent, post-hoc interpretation.
"""

from __future__ import annotations

import numpy as np

from fusionbench.scenarios import calibrated as cal
from fusionbench.scenarios.base import Scenario


def _detector(cond: np.ndarray, accuracy: float, rng) -> np.ndarray:
    """Calibrated [N,2] detector of a balanced Boolean condition."""
    return cal.to_binary_proba(cal.binary_calibrated(cond, accuracy, rng))


def _combine_same_condition(p1_cols: np.ndarray) -> np.ndarray:
    """P(condition=1 | several independent calibrated detectors) = logit-sum."""
    return cal.sigmoid(cal.logit(p1_cols).sum(axis=1))


class HardVeto(Scenario):
    """y = presence AND safe. A precise "safe" detector vetoes false alarms.

    Bayes optimum P(y=1) = P(presence)*P(safe): a product. Equal-weight averaging
    mislabels false alarms (presence high, safe low -> averages near 0.5); a
    min-like gate corrects them.
    """

    task = "binary"
    expected_winners = ["op_min", "okapi", "tpot"]
    oracle_description = "P(presence | D1,D2) * P(safe | V)"

    def __init__(self, acc_detect: float = 0.90, acc_veto: float = 0.93, **kw):
        super().__init__(acc_detect=acc_detect, acc_veto=acc_veto, **kw)
        self.acc_detect, self.acc_veto = acc_detect, acc_veto

    @property
    def n_classes(self):
        return 2

    def model_ids(self):
        return ["detect1", "detect2", "veto"]

    def _generate(self, n, rng):
        c_pres = rng.integers(0, 2, n)
        c_safe = rng.integers(0, 2, n)
        y = (c_pres & c_safe).astype(int)
        preds = np.stack([
            _detector(c_pres, self.acc_detect, rng),
            _detector(c_pres, self.acc_detect, rng),
            _detector(c_safe, self.acc_veto, rng),
        ], axis=1)
        return preds, y, {}

    def bayes_oracle(self, preds, latent):
        p1 = preds[:, :, 1]
        presence = _combine_same_condition(p1[:, 0:2])
        safe = p1[:, 2]
        return cal.to_binary_proba(presence * safe)


class HardRescue(Scenario):
    """y = c1 OR c2. Each detector catches a different way to be positive.

    Bayes optimum is the noisy-OR ``1-(1-D1)(1-D2)``. Averaging dilutes a single
    firing detector below threshold; a max-like gate keeps it.
    """

    task = "binary"
    expected_winners = ["op_max", "okapi", "tpot"]
    oracle_description = "noisy-OR: 1 - (1-D1)(1-D2)"

    def __init__(self, n_detect: int = 3, accuracy: float = 0.90, **kw):
        super().__init__(n_detect=n_detect, accuracy=accuracy, **kw)
        self.n_detect, self.accuracy = n_detect, accuracy

    @property
    def n_classes(self):
        return 2

    def model_ids(self):
        return [f"detect{i}" for i in range(self.n_detect)]

    def _generate(self, n, rng):
        conds = [rng.integers(0, 2, n) for _ in range(self.n_detect)]
        y = np.zeros(n, dtype=int)
        for c in conds:
            y |= c
        preds = np.stack([_detector(c, self.accuracy, rng) for c in conds], axis=1)
        return preds, y, {}

    def bayes_oracle(self, preds, latent):
        p1 = preds[:, :, 1]
        pos = 1.0 - np.prod(1.0 - p1, axis=1)
        return cal.to_binary_proba(pos)


class WeakestLink(Scenario):
    """y = AND of m *distinct* conditions, one detector each.

    The positive region is the high-corner of an m-cube. A single linear boundary
    cannot require *all* detectors to be high simultaneously (it permits trading a
    very-high score off against a low one), so even a logit-space linear stacker
    has irreducible false positives. A product / min-style gate requires every
    detector to agree and is near-optimal — a scenario where OKAPI's min beats all
    linear methods on the primary metric, not just naive averaging.
    """

    task = "binary"
    expected_winners = ["op_min", "okapi", "tpot"]
    oracle_description = "product of m independent calibrated detectors (corner of m-cube)"

    def __init__(self, m: int = 4, accuracy: float = 0.90, **kw):
        super().__init__(m=m, accuracy=accuracy, **kw)
        self.m, self.accuracy = m, accuracy

    @property
    def n_classes(self):
        return 2

    def model_ids(self):
        return [f"link{i}" for i in range(self.m)]

    def _generate(self, n, rng):
        conds = [rng.integers(0, 2, n) for _ in range(self.m)]
        y = np.ones(n, dtype=int)
        for c in conds:
            y &= c
        preds = np.stack([_detector(c, self.accuracy, rng) for c in conds], axis=1)
        return preds, y, {}

    def bayes_oracle(self, preds, latent):
        p1 = preds[:, :, 1]
        return cal.to_binary_proba(np.prod(p1, axis=1))


class TwoPathways(Scenario):
    """y = (A AND B) OR (C AND D): two independent pathways to a positive.

    The decision region is the *union of two disjoint high-corners* in different
    dimensions, which is provably **not linearly separable** in probability or
    logit space (so stacking_lr / stacking_logit cannot represent it), nor by any
    single operator. A composition ``max(min(A,B), min(C,D))`` separates it
    exactly — the canonical case where OKAPI's evolved alternating min/max tree
    (and tree-based AutoML) genuinely beats all linear fusion methods on the
    primary metric.
    """

    task = "binary"
    expected_winners = ["okapi", "tpot"]
    oracle_description = "noisy-OR of two products: 1-(1-A*B)(1-C*D)"

    def __init__(self, accuracy: float = 0.90, **kw):
        super().__init__(accuracy=accuracy, **kw)
        self.accuracy = accuracy

    @property
    def n_classes(self):
        return 2

    def model_ids(self):
        return ["a", "b", "c", "d"]

    def _generate(self, n, rng):
        a, b, c, d = (rng.integers(0, 2, n) for _ in range(4))
        y = (((a & b) | (c & d))).astype(int)
        preds = np.stack([_detector(v, self.accuracy, rng) for v in (a, b, c, d)], axis=1)
        return preds, y, {}

    def bayes_oracle(self, preds, latent):
        p = preds[:, :, 1]
        pab = p[:, 0] * p[:, 1]
        pcd = p[:, 2] * p[:, 3]
        return cal.to_binary_proba(1.0 - (1.0 - pab) * (1.0 - pcd))


class Parity(Scenario):
    """y = c1 XOR c2: a non-monotone interaction.

    The Bayes optimum ``D1(1-D2) + (1-D1)D2`` is neither linearly separable nor
    monotone in the detectors, so it defeats *both* linear fusion (averaging,
    stackers) *and* OKAPI's monotone operators (min/max/far/weighted-mean). Only
    a tree-based meta-learner (TPOT) can represent it. Included as an honest
    limit case marking the boundary of every consensus/operator method.
    """

    task = "binary"
    expected_winners = ["tpot"]
    oracle_description = "XOR posterior: D1(1-D2) + (1-D1)D2"

    def __init__(self, accuracy: float = 0.92, **kw):
        super().__init__(accuracy=accuracy, **kw)
        self.accuracy = accuracy

    @property
    def n_classes(self):
        return 2

    def model_ids(self):
        return ["c1", "c2"]

    def _generate(self, n, rng):
        c1 = rng.integers(0, 2, n)
        c2 = rng.integers(0, 2, n)
        y = (c1 ^ c2).astype(int)
        preds = np.stack([_detector(c1, self.accuracy, rng),
                          _detector(c2, self.accuracy, rng)], axis=1)
        return preds, y, {}

    def bayes_oracle(self, preds, latent):
        d1, d2 = preds[:, 0, 1], preds[:, 1, 1]
        return cal.to_binary_proba(d1 * (1 - d2) + (1 - d1) * d2)


class Compound(Scenario):
    """y = (core OR rescue) AND safe. Consensus, rescue, and veto in one rule.

    Requires a depth>=2 composition: a weak-consensus core, OR'd with a rare
    strong rescue detector, then AND'd (vetoed) by a precise safety detector.
    Only compositional / nonlinear methods reach the optimum.
    """

    task = "binary"
    expected_winners = ["okapi", "tpot"]
    oracle_description = "[1-(1-core)(1-rescue)] * safe; core = logit-sum(D1,D2,D3)"

    def __init__(self, acc_core: float = 0.74, acc_rescue: float = 0.92,
                 acc_safe: float = 0.92, **kw):
        super().__init__(acc_core=acc_core, acc_rescue=acc_rescue, acc_safe=acc_safe, **kw)
        self.acc_core, self.acc_rescue, self.acc_safe = acc_core, acc_rescue, acc_safe

    @property
    def n_classes(self):
        return 2

    def model_ids(self):
        return ["core1", "core2", "core3", "rescue", "safe"]

    def _generate(self, n, rng):
        c_core = rng.integers(0, 2, n)
        c_rescue = rng.integers(0, 2, n)
        c_safe = rng.integers(0, 2, n)
        y = (((c_core | c_rescue) & c_safe)).astype(int)
        preds = np.stack([
            _detector(c_core, self.acc_core, rng),
            _detector(c_core, self.acc_core, rng),
            _detector(c_core, self.acc_core, rng),
            _detector(c_rescue, self.acc_rescue, rng),
            _detector(c_safe, self.acc_safe, rng),
        ], axis=1)
        return preds, y, {}

    def bayes_oracle(self, preds, latent):
        p1 = preds[:, :, 1]
        core = _combine_same_condition(p1[:, 0:3])
        rescue = p1[:, 3]
        safe = p1[:, 4]
        or_part = 1.0 - (1.0 - core) * (1.0 - rescue)
        return cal.to_binary_proba(or_part * safe)

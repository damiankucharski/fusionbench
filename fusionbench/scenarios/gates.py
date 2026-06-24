"""Logical-gate pathologies (genuinely nonlinear optima).

The label is a *logical function* of independent latent conditions, and each base
model is a calibrated detector of one condition. Because the label is determined
jointly, the detectors are conditionally *dependent* given the label, so the
Bayes optimum is a nonlinear interaction (product for AND, noisy-OR for OR, and
disjunction-of-products for DNF) — not representable as a (weighted) sum of
logits. Linear/logit stackers are therefore capped below the oracle on the
non-separable cases (DNF, parity), while OKAPI's evolved min/max/threshold
composition can reach it.

Each condition is detected by ``rep`` redundant calibrated models, so the pool
is realistic (several models per concept) and large enough for OKAPI's
population-based search. The generators describe only conditions and Boolean
logic — the words min/max/veto never enter the data-generating code.
"""

from __future__ import annotations

import numpy as np

from fusionbench.scenarios import calibrated as cal
from fusionbench.scenarios.base import Scenario


def _detector(cond: np.ndarray, accuracy: float, rng) -> np.ndarray:
    """Calibrated [N,2] detector of a balanced Boolean condition."""
    return cal.to_binary_proba(cal.binary_calibrated(cond, accuracy, rng))


def _reps(cond, accuracy, rep, rng):
    return [_detector(cond, accuracy, rng) for _ in range(rep)]


def _combine(p1_cols: np.ndarray) -> np.ndarray:
    """P(condition=1 | several independent calibrated detectors) = logit-sum."""
    return cal.sigmoid(cal.logit(p1_cols).sum(axis=1))


class HardVeto(Scenario):
    """y = presence AND safe. A precise "safe" detector vetoes false alarms.

    Bayes optimum P(y=1) = P(presence)*P(safe): a product. Equal-weight averaging
    mislabels false alarms; a min-like gate corrects them.
    """

    task = "binary"
    expected_winners = ["okapi", "tpot"]
    oracle_description = "P(presence)*P(safe), each a logit-sum of rep detectors"

    def __init__(self, acc_detect: float = 0.85, acc_veto: float = 0.88, rep: int = 3, **kw):
        super().__init__(acc_detect=acc_detect, acc_veto=acc_veto, rep=rep, **kw)
        self.acc_detect, self.acc_veto, self.rep = acc_detect, acc_veto, rep

    @property
    def n_classes(self):
        return 2

    def model_ids(self):
        return [f"detect{i}" for i in range(self.rep)] + [f"veto{i}" for i in range(self.rep)]

    def _generate(self, n, rng):
        c_pres = rng.integers(0, 2, n)
        c_safe = rng.integers(0, 2, n)
        y = (c_pres & c_safe).astype(int)
        cols = _reps(c_pres, self.acc_detect, self.rep, rng) + _reps(c_safe, self.acc_veto, self.rep, rng)
        return np.stack(cols, axis=1), y, {}

    def bayes_oracle(self, preds, latent):
        p1 = preds[:, :, 1]
        presence = _combine(p1[:, : self.rep])
        safe = _combine(p1[:, self.rep:])
        return cal.to_binary_proba(presence * safe)


class HardRescue(Scenario):
    """y = OR of n_detect conditions. Each pathway caught by rep detectors.

    Bayes optimum is the noisy-OR; averaging dilutes a single firing pathway, a
    max-like gate keeps it.
    """

    task = "binary"
    expected_winners = ["okapi", "tpot"]
    oracle_description = "noisy-OR over conditions (each a logit-sum of rep detectors)"

    def __init__(self, n_detect: int = 3, accuracy: float = 0.88, rep: int = 2, **kw):
        super().__init__(n_detect=n_detect, accuracy=accuracy, rep=rep, **kw)
        self.n_detect, self.accuracy, self.rep = n_detect, accuracy, rep

    @property
    def n_classes(self):
        return 2

    def model_ids(self):
        return [f"c{j}_d{i}" for j in range(self.n_detect) for i in range(self.rep)]

    def _generate(self, n, rng):
        conds = [rng.integers(0, 2, n) for _ in range(self.n_detect)]
        y = np.zeros(n, dtype=int)
        for c in conds:
            y |= c
        cols = []
        for c in conds:
            cols += _reps(c, self.accuracy, self.rep, rng)
        return np.stack(cols, axis=1), y, {}

    def bayes_oracle(self, preds, latent):
        p1 = preds[:, :, 1]
        pos = np.ones(p1.shape[0])
        for j in range(self.n_detect):
            pj = _combine(p1[:, j * self.rep:(j + 1) * self.rep])
            pos = pos * (1.0 - pj)
        return cal.to_binary_proba(1.0 - pos)


class WeakestLink(Scenario):
    """y = AND of m conditions (each detected by rep models).

    The positive region is the high-corner of an m-cube. A product / min-style
    gate requires every condition to hold; a single linear boundary permits
    trading a high score against a low one.
    """

    task = "binary"
    expected_winners = ["okapi", "tpot"]
    oracle_description = "product of m conditions (each a logit-sum of rep detectors)"

    def __init__(self, m: int = 4, accuracy: float = 0.85, rep: int = 2, **kw):
        super().__init__(m=m, accuracy=accuracy, rep=rep, **kw)
        self.m, self.accuracy, self.rep = m, accuracy, rep

    @property
    def n_classes(self):
        return 2

    def model_ids(self):
        return [f"link{j}_d{i}" for j in range(self.m) for i in range(self.rep)]

    def _generate(self, n, rng):
        conds = [rng.integers(0, 2, n) for _ in range(self.m)]
        y = np.ones(n, dtype=int)
        for c in conds:
            y &= c
        cols = []
        for c in conds:
            cols += _reps(c, self.accuracy, self.rep, rng)
        return np.stack(cols, axis=1), y, {}

    def bayes_oracle(self, preds, latent):
        p1 = preds[:, :, 1]
        pos = np.ones(p1.shape[0])
        for j in range(self.m):
            pos = pos * _combine(p1[:, j * self.rep:(j + 1) * self.rep])
        return cal.to_binary_proba(pos)


class TwoPathways(Scenario):
    """y = (A AND B) OR (C AND D): two independent pathways to a positive.

    The decision region is the union of two disjoint high-corners in different
    dimensions, provably **not linearly separable** in probability or logit space
    (so both stackers are capped below the oracle), nor reachable by any single
    operator. The composition ``max(min(A,B), min(C,D))`` separates it — the
    canonical case where OKAPI's evolved alternating tree beats every linear
    fusion method. Each variable is detected by ``rep`` models.
    """

    task = "binary"
    expected_winners = ["okapi", "tpot"]
    oracle_description = "noisy-OR of two products; each variable a logit-sum of rep detectors"

    def __init__(self, accuracy: float = 0.9, rep: int = 3, **kw):
        super().__init__(accuracy=accuracy, rep=rep, **kw)
        self.accuracy, self.rep = accuracy, rep

    @property
    def n_classes(self):
        return 2

    def model_ids(self):
        return [f"{v}_d{i}" for v in ("a", "b", "c", "d") for i in range(self.rep)]

    def _generate(self, n, rng):
        a, b, c, d = (rng.integers(0, 2, n) for _ in range(4))
        y = (((a & b) | (c & d))).astype(int)
        cols = []
        for v in (a, b, c, d):
            cols += _reps(v, self.accuracy, self.rep, rng)
        return np.stack(cols, axis=1), y, {}

    def bayes_oracle(self, preds, latent):
        p1 = preds[:, :, 1]
        r = self.rep
        pa, pb, pc, pd = (_combine(p1[:, i * r:(i + 1) * r]) for i in range(4))
        return cal.to_binary_proba(1.0 - (1.0 - pa * pb) * (1.0 - pc * pd))


class Parity(Scenario):
    """y = c1 XOR c2: a non-monotone interaction.

    The Bayes optimum is neither linearly separable nor monotone, so it defeats
    linear fusion *and* OKAPI's monotone operators — only a tree-based meta-learner
    can represent it. An honest limit case for every consensus/operator method.
    """

    task = "binary"
    expected_winners = ["okapi", "tpot"]
    oracle_description = "XOR posterior p1(1-p2)+(1-p1)p2 (each a logit-sum of rep detectors)"

    def __init__(self, accuracy: float = 0.92, rep: int = 2, **kw):
        super().__init__(accuracy=accuracy, rep=rep, **kw)
        self.accuracy, self.rep = accuracy, rep

    @property
    def n_classes(self):
        return 2

    def model_ids(self):
        return [f"c{j}_d{i}" for j in range(2) for i in range(self.rep)]

    def _generate(self, n, rng):
        c1 = rng.integers(0, 2, n)
        c2 = rng.integers(0, 2, n)
        y = (c1 ^ c2).astype(int)
        cols = _reps(c1, self.accuracy, self.rep, rng) + _reps(c2, self.accuracy, self.rep, rng)
        return np.stack(cols, axis=1), y, {}

    def bayes_oracle(self, preds, latent):
        p1 = preds[:, :, 1]
        a = _combine(p1[:, : self.rep])
        b = _combine(p1[:, self.rep:])
        return cal.to_binary_proba(a * (1 - b) + (1 - a) * b)


class Compound(Scenario):
    """y = (core OR rescue) AND safe. Consensus, rescue, and veto in one rule.

    Requires a depth>=2 composition: a consensus core, OR'd with a rescue
    detector, then AND'd (vetoed) by a safety detector. Each concept is detected
    by ``rep`` models. Only compositional / nonlinear methods reach the optimum.
    """

    task = "binary"
    expected_winners = ["okapi", "tpot"]
    oracle_description = "[1-(1-core)(1-rescue)]*safe; each concept a logit-sum of rep detectors"

    def __init__(self, acc_core: float = 0.78, acc_rescue: float = 0.9,
                 acc_safe: float = 0.9, rep: int = 3, **kw):
        super().__init__(acc_core=acc_core, acc_rescue=acc_rescue, acc_safe=acc_safe, rep=rep, **kw)
        self.acc_core, self.acc_rescue, self.acc_safe, self.rep = acc_core, acc_rescue, acc_safe, rep

    @property
    def n_classes(self):
        return 2

    def model_ids(self):
        return ([f"core{i}" for i in range(self.rep)] + [f"rescue{i}" for i in range(self.rep)]
                + [f"safe{i}" for i in range(self.rep)])

    def _generate(self, n, rng):
        c_core = rng.integers(0, 2, n)
        c_rescue = rng.integers(0, 2, n)
        c_safe = rng.integers(0, 2, n)
        y = (((c_core | c_rescue) & c_safe)).astype(int)
        cols = (_reps(c_core, self.acc_core, self.rep, rng)
                + _reps(c_rescue, self.acc_rescue, self.rep, rng)
                + _reps(c_safe, self.acc_safe, self.rep, rng))
        return np.stack(cols, axis=1), y, {}

    def bayes_oracle(self, preds, latent):
        p1 = preds[:, :, 1]
        r = self.rep
        core = _combine(p1[:, :r])
        rescue = _combine(p1[:, r:2 * r])
        safe = _combine(p1[:, 2 * r:])
        return cal.to_binary_proba((1.0 - (1.0 - core) * (1.0 - rescue)) * safe)

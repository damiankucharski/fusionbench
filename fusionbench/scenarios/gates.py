"""Logical-gate pathologies (genuinely nonlinear optima).

The label is a *logical function* of independent latent conditions, and each base
model is a calibrated detector of one condition. Because the label is determined
jointly, the detectors are conditionally *dependent* given the label, so the
Bayes optimum is a nonlinear interaction (product for AND, noisy-OR for OR, and
disjunction-of-products for DNF) — not representable as a (weighted) sum of
logits. Linear/logit stackers are therefore capped below the oracle on the
non-separable cases (DNF, parity), while OKAPI's evolved min/max/threshold
composition can reach it.

Optional case controls reweight the fair-bit reference law. positive_prevalence
is P(y=1); decisive_fraction is P(decisive | eligible label). The eligible
outcome is negative for a veto or single weakest-link failure and positive for
a sole rescue, second-only pathway, or safe rescue-only compound branch.
Within each group, Boolean patterns retain their reference proportions.
The stress posterior uses the changed joint prior and observed detector scores.

Each condition is detected by ``rep`` redundant calibrated models, so the pool
is realistic (several models per concept) and large enough for OKAPI's
population-based search. The generators describe only conditions and Boolean
logic — the words min/max/veto never enter the data-generating code.
"""

from __future__ import annotations

from itertools import product
from typing import ClassVar

import numpy as np
from scipy.special import expit
from scipy.stats import binom

from fusionbench.scenarios import calibrated as cal
from fusionbench.scenarios.base import Scenario
from fusionbench.scenarios.calibration import _detector_log_ratio


def _detector(cond: np.ndarray, accuracy: float, rng) -> np.ndarray:
    """Calibrated [N,2] detector of a balanced Boolean condition."""
    return cal.to_binary_proba(cal.binary_calibrated(cond, accuracy, rng))


def _reps(cond, accuracy, rep, rng):
    return [_detector(cond, accuracy, rng) for _ in range(rep)]


def _combine(p1_cols: np.ndarray) -> np.ndarray:
    """P(condition=1 | several independent calibrated detectors) = logit-sum."""
    return cal.sigmoid(cal.logit(p1_cols).sum(axis=1))


class _GateScenario(Scenario):
    """Reweight decisive, other eligible and opposite-label Boolean patterns.

    Patterns within a group remain equiprobable. Detector likelihoods given
    the bits are unchanged. The posterior reweights the same three observable
    evidence masses by their new prior divided by their original prior.
    """

    def __init__(self, *, positive_prevalence=None, decisive_fraction=None, **kw):
        self._depth = kw.get("m", kw.get("n_detect", self._fixed_depth))
        self._active = positive_prevalence is not None or decisive_fraction is not None
        self._positive_eligible = self._kind in ("rescue", "paths", "compound")
        self._count_gate = self._kind in ("weak", "rescue")
        if self._active:
            for key, value in (
                ("positive_prevalence", positive_prevalence),
                ("decisive_fraction", decisive_fraction),
            ):
                if value is not None:
                    if not 0 <= value <= 1:
                        raise ValueError(f"{key} must lie in [0, 1]")
                    kw[key] = value
            if self._count_gate:
                if not isinstance(self._depth, (int, np.integer)) or self._depth < 1:
                    raise ValueError("Gate depth must be a positive integer")
                counts = np.arange(self._depth + 1)
                # Count successes in OR, failures in AND. Groups 0,1,2 are
                # opposite label, decisive eligible, other eligible.
                self._pattern_groups = np.minimum(2, counts)
                self._pattern_probs = binom.pmf(counts, self._depth, 0.5)
            else:
                self._states = np.array(list(product((0, 1), repeat=self._depth)))
                y, decisive = self._outcomes(self._states.T)
                self._pattern_groups = np.where(
                    y != self._positive_eligible, 0, np.where(decisive, 1, 2)
                )
                self._pattern_probs = np.full(len(y), 1 / len(y))
            base = np.bincount(
                self._pattern_groups, weights=self._pattern_probs, minlength=3
            )
            eligible = base[1:].sum()
            self._positive = positive_prevalence
            if self._positive is None:
                self._positive = eligible if self._positive_eligible else base[0]
            self._fraction = (
                base[1] / eligible if decisive_fraction is None else decisive_fraction
            )
            eligible = self._positive if self._positive_eligible else 1 - self._positive
            opposite = 1 - self._positive if self._positive_eligible else self._positive
            target = np.array(
                [opposite, eligible * self._fraction, eligible * (1 - self._fraction)]
            )
            if np.any((target > 0) & (base == 0)):
                raise ValueError(
                    "Requested decisive fraction assigns mass to an impossible gate pattern"
                )
            with np.errstate(divide="ignore"):
                self._log_weights = np.log(
                    np.divide(target, base, out=np.zeros(3), where=base > 0)
                )
            self.evaluation_group_fields = ("decisive",)
            self.oracle_description = (
                "Clipped Gaussian detector likelihoods integrated over Boolean "
                "patterns with the configured label and decisive-case probabilities."
            )
        super().__init__(**kw)

    def _outcomes(self, bits):
        if self._count_gate:
            count = np.sum(bits, axis=0)
            events = count if self._positive_eligible else self._depth - count
            label = count > 0 if self._positive_eligible else events == 0
            return label, events == 1
        a, b, *tail = bits
        if self._kind == "veto":
            return a & b, (a == 1) & (b == 0)
        if self._kind == "compound":
            (safe,) = tail
            return (a | b) & safe, (a == 0) & (b == 1) & (safe == 1)
        c, d = tail
        return (a & b) | (c & d), ((a & b) == 0) & ((c & d) == 1)

    def _conditions(self, n, rng):
        if not self._active:
            return [rng.integers(0, 2, n) for _ in range(self._depth)]
        y = rng.random(n) < self._positive
        decisive = rng.random(n) < self._fraction
        groups = np.where(y != self._positive_eligible, 0, np.where(decisive, 1, 2))
        u = rng.random(n)
        pattern = np.empty(n, dtype=int)
        for group in range(3):
            rows = groups == group
            if not rows.any():
                continue
            choices = np.flatnonzero(self._pattern_groups == group)
            cdf = np.cumsum(self._pattern_probs[choices])
            pattern[rows] = choices[
                np.searchsorted(cdf / cdf[-1], u[rows], side="right")
            ]
        if not self._count_gate:
            return self._states[pattern].T
        # Given its count, every subset has equal probability. This also
        # handles deep conjunctions without enumerating 2**depth states.
        ranks = rng.random((n, self._depth)).argsort(axis=1).argsort(axis=1)
        selected = ranks < pattern[:, None]
        return (selected if self._positive_eligible else ~selected).astype(int).T

    def _generate(self, n, rng):
        bits = self._conditions(n, rng)
        y, decisive = self._outcomes(bits)
        cols = []
        for condition, accuracy in zip(bits, self._accuracies(), strict=True):
            cols.extend(_reps(condition, accuracy, self.rep, rng))
        return np.stack(cols, axis=1), y.astype(int), {"decisive": decisive}

    def _accuracies(self):
        if self._kind == "veto":
            return (self.acc_detect, self.acc_veto)
        if self._kind == "compound":
            return (self.acc_core, self.acc_rescue, self.acc_safe)
        return [self.accuracy] * self._depth

    def _weighted_oracle(self, preds):
        evidence = np.stack(
            [
                _detector_log_ratio(
                    preds[:, i * self.rep : (i + 1) * self.rep, 1], accuracy
                ).sum(axis=1)
                for i, accuracy in enumerate(self._accuracies())
            ]
        )
        log_one = -np.logaddexp(0, -evidence)
        log_zero = -np.logaddexp(0, evidence)
        if self._count_gate:
            if not self._positive_eligible:
                log_one, log_zero = log_zero, log_one
            none = np.zeros(len(preds))
            one = np.full(len(preds), -np.inf)
            many = one.copy()
            for event, other in zip(log_one, log_zero, strict=True):
                many = np.logaddexp(many, one + event)
                one = np.logaddexp(one + other, none + event)
                none = none + other
            mass = np.stack((none, one, many))
        else:
            mass = np.full((3, len(preds)), -np.inf)
            for state, group in zip(self._states, self._pattern_groups, strict=True):
                joint = np.where(state[:, None], log_one, log_zero).sum(axis=0)
                mass[group] = np.logaddexp(mass[group], joint)
        mass += self._log_weights[:, None]
        eligible_log_odds = np.logaddexp(mass[1], mass[2]) - mass[0]
        return cal.to_binary_proba(
            expit(eligible_log_odds if self._positive_eligible else -eligible_log_odds)
        )


class HardVeto(_GateScenario):
    """y = presence AND safe. A precise "safe" detector vetoes false alarms.

    Bayes optimum P(y=1) = P(presence)*P(safe): a product. Equal-weight averaging
    mislabels false alarms; a min-like gate corrects them.
    """

    _kind = "veto"
    _fixed_depth = 2

    task = "binary"
    expected_winners: ClassVar[list[str]] = ["okapi", "tpot"]
    oracle_description = "P(presence)*P(safe), each a logit-sum of rep detectors"

    def __init__(self, acc_detect: float = 0.85, acc_veto: float = 0.88, rep: int = 3, **kw):
        super().__init__(acc_detect=acc_detect, acc_veto=acc_veto, rep=rep, **kw)
        self.acc_detect, self.acc_veto, self.rep = acc_detect, acc_veto, rep

    @property
    def n_classes(self):
        return 2

    def model_ids(self):
        return [f"detect{i}" for i in range(self.rep)] + [f"veto{i}" for i in range(self.rep)]

    def bayes_oracle(self, preds, latent):
        if self._active:
            return self._weighted_oracle(preds)
        p1 = preds[:, :, 1]
        presence = _combine(p1[:, : self.rep])
        safe = _combine(p1[:, self.rep:])
        return cal.to_binary_proba(presence * safe)


class HardRescue(_GateScenario):
    """y = OR of n_detect conditions. Each pathway caught by rep detectors.

    Bayes optimum is the noisy-OR; averaging dilutes a single firing pathway, a
    max-like gate keeps it.
    """

    _kind = "rescue"
    _fixed_depth = 3

    task = "binary"
    expected_winners: ClassVar[list[str]] = ["okapi", "tpot"]
    oracle_description = "noisy-OR over conditions (each a logit-sum of rep detectors)"

    def __init__(self, n_detect: int = 3, accuracy: float = 0.88, rep: int = 2, **kw):
        super().__init__(n_detect=n_detect, accuracy=accuracy, rep=rep, **kw)
        self.n_detect, self.accuracy, self.rep = n_detect, accuracy, rep

    @property
    def n_classes(self):
        return 2

    def model_ids(self):
        return [f"c{j}_d{i}" for j in range(self.n_detect) for i in range(self.rep)]

    def bayes_oracle(self, preds, latent):
        if self._active:
            return self._weighted_oracle(preds)
        p1 = preds[:, :, 1]
        pos = np.ones(p1.shape[0])
        for j in range(self.n_detect):
            pj = _combine(p1[:, j * self.rep:(j + 1) * self.rep])
            pos = pos * (1.0 - pj)
        return cal.to_binary_proba(1.0 - pos)


class WeakestLink(_GateScenario):
    """y = AND of m conditions (each detected by rep models).

    The positive region is the high-corner of an m-cube. A product / min-style
    gate requires every condition to hold; a single linear boundary permits
    trading a high score against a low one.
    """

    _kind = "weak"
    _fixed_depth = 4

    task = "binary"
    expected_winners: ClassVar[list[str]] = ["okapi", "tpot"]
    oracle_description = "product of m conditions (each a logit-sum of rep detectors)"

    def __init__(self, m: int = 4, accuracy: float = 0.85, rep: int = 2, **kw):
        super().__init__(m=m, accuracy=accuracy, rep=rep, **kw)
        self.m, self.accuracy, self.rep = m, accuracy, rep

    @property
    def n_classes(self):
        return 2

    def model_ids(self):
        return [f"link{j}_d{i}" for j in range(self.m) for i in range(self.rep)]

    def bayes_oracle(self, preds, latent):
        if self._active:
            return self._weighted_oracle(preds)
        p1 = preds[:, :, 1]
        pos = np.ones(p1.shape[0])
        for j in range(self.m):
            pos = pos * _combine(p1[:, j * self.rep:(j + 1) * self.rep])
        return cal.to_binary_proba(pos)


class TwoPathways(_GateScenario):
    """y = (A AND B) OR (C AND D): two independent pathways to a positive.

    The decision region is the union of two disjoint high-corners in different
    dimensions, provably **not linearly separable** in probability or logit space
    (so both stackers are capped below the oracle), nor reachable by any single
    operator. The composition ``max(min(A,B), min(C,D))`` separates it — the
    canonical case where OKAPI's evolved alternating tree beats every linear
    fusion method. Each variable is detected by ``rep`` models.
    """

    _kind = "paths"
    _fixed_depth = 4

    task = "binary"
    expected_winners: ClassVar[list[str]] = ["okapi", "tpot"]
    oracle_description = "noisy-OR of two products; each variable a logit-sum of rep detectors"

    def __init__(self, accuracy: float = 0.9, rep: int = 3, **kw):
        super().__init__(accuracy=accuracy, rep=rep, **kw)
        self.accuracy, self.rep = accuracy, rep

    @property
    def n_classes(self):
        return 2

    def model_ids(self):
        return [f"{v}_d{i}" for v in ("a", "b", "c", "d") for i in range(self.rep)]

    def bayes_oracle(self, preds, latent):
        if self._active:
            return self._weighted_oracle(preds)
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
    expected_winners: ClassVar[list[str]] = ["okapi", "tpot"]
    oracle_description = "XOR posterior p1(1-p2)+(1-p1)p2 (each a logit-sum of rep detectors)"

    def __init__(self, accuracy: float = 0.92, rep: int = 2, second_accuracy=None, **kw):
        if second_accuracy is not None:
            if not 0.5 <= second_accuracy < 1:
                raise ValueError("second_accuracy must lie in [0.5, 1)")
            kw["second_accuracy"] = second_accuracy
        self.second_accuracy = second_accuracy
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
        a = _reps(c1, self.accuracy, self.rep, rng)
        b = _reps(c2, self.accuracy if self.second_accuracy is None else self.second_accuracy, self.rep, rng)
        if self.second_accuracy == 0.5:
            for column in b:
                column.fill(0.5)
        return np.stack(a + b, axis=1), y, {}

    def bayes_oracle(self, preds, latent):
        p1 = preds[:, :, 1]
        if self.second_accuracy is not None:
            a = expit(_detector_log_ratio(p1[:, :self.rep], self.accuracy).sum(axis=1))
            if self.second_accuracy == 0.5:
                return cal.to_binary_proba(np.full(len(a), 0.5))
            b = expit(_detector_log_ratio(p1[:, self.rep:], self.second_accuracy).sum(axis=1))
            return cal.to_binary_proba(a * (1 - b) + (1 - a) * b)
        a = _combine(p1[:, : self.rep])
        b = _combine(p1[:, self.rep:])
        return cal.to_binary_proba(a * (1 - b) + (1 - a) * b)


class Compound(_GateScenario):
    """y = (core OR rescue) AND safe. Consensus, rescue, and veto in one rule.

    Requires a depth>=2 composition: a consensus core, OR'd with a rescue
    detector, then AND'd (vetoed) by a safety detector. Each concept is detected
    by ``rep`` models. Only compositional / nonlinear methods reach the optimum.
    """

    _kind = "compound"
    _fixed_depth = 3

    task = "binary"
    expected_winners: ClassVar[list[str]] = ["okapi", "tpot"]
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

    def bayes_oracle(self, preds, latent):
        if self._active:
            return self._weighted_oracle(preds)
        p1 = preds[:, :, 1]
        r = self.rep
        core = _combine(p1[:, :r])
        rescue = _combine(p1[:, r:2 * r])
        safe = _combine(p1[:, 2 * r:])
        return cal.to_binary_proba((1.0 - (1.0 - core) * (1.0 - rescue)) * safe)

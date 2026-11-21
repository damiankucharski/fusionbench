"""Anonymous provider scores with known native-gate and logistic label laws."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import expit, logit

from fusionbench.scenarios.base import Scenario


def farther(a, b, threshold):
    """Native FAR on two scalar operands, retaining the parent A at ties."""
    return np.where(np.abs(a - threshold) >= np.abs(b - threshold), a, b)


def conditional_probability(a, b, mixing, *, close=False, threshold=27 / 64):
    if not 0 <= mixing <= 1:
        raise ValueError("mixing must lie in [0, 1]")
    if close:
        switch = np.where(np.abs(a - threshold) <= np.abs(b - threshold), a, b)
    else:
        switch = farther(a, b, threshold)
    linear = expit(0.2 + 0.7 * logit(a) + 0.7 * logit(b))
    return (1 - mixing) * switch + mixing * linear


@dataclass
class SwitchDraw:
    preds: np.ndarray
    uniform: np.ndarray
    parent: int
    child: int
    close: bool = False
    threshold: float = 27 / 64

    def probability(self, mixing):
        return conditional_probability(
            self.preds[:, self.parent, 1],
            self.preds[:, self.child, 1],
            mixing,
            close=self.close,
            threshold=self.threshold,
        )

    def labels(self, mixing):
        return (self.uniform < self.probability(mixing)).astype(int)


class NativeSwitch(Scenario):
    """Mix a FAR gate and a logistic control, keeping the input distribution fixed."""

    close = False
    threshold = 27 / 64
    primary_metric = "brier"
    oracle_description = (
        "Mixture of an ordered two-provider FAR gate and an affine logit posterior"
    )

    def __init__(self, mixing=0.0, role_seed=0, **kw):
        if not 0 <= mixing <= 1:
            raise ValueError("mixing must lie in [0, 1]")
        super().__init__(mixing=mixing, role_seed=role_seed, **kw)
        self.mixing = mixing
        self.roles = np.random.default_rng([role_seed, 15]).permutation(6)[:2]

    @property
    def n_classes(self):
        return 2

    def model_ids(self):
        return [f"p{i}" for i in range(6)]

    def draw(self, n, rng):
        scores = rng.integers(1, 64, size=(n, 6)) / 64
        return SwitchDraw(
            np.stack((1 - scores, scores), axis=-1),
            rng.random(n),
            *map(int, self.roles),
            close=self.close,
            threshold=self.threshold,
        )

    def _generate(self, n, rng):
        sample = self.draw(n, rng)
        return sample.preds, sample.labels(self.mixing), {}

    def bayes_oracle(self, preds, latent):
        a, b = self.roles
        score = conditional_probability(
            preds[:, a, 1],
            preds[:, b, 1],
            self.mixing,
            close=self.close,
            threshold=self.threshold,
        )
        return np.stack((1 - score, score), axis=1)


class NativeCloseSwitch(NativeSwitch):
    """Nearest-operand control on exactly the FAR panel's anonymous features.

    The new direction/cutoff forms a preregistered transport test in
    docs/ensemble-study/search-regret/matched-graft-protocol.md.
    """

    close = True
    threshold = 37 / 64
    oracle_description = (
        "Mixture of an ordered two-provider CLOSE gate and an affine logit posterior"
    )


def draw_native_switch(seed, split, n):
    return NativeSwitch(role_seed=seed).draw(n, np.random.default_rng([seed, split]))

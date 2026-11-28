"""An original controlled challenge for identifying compositions of anonymous scores."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import expit, logit

from fusionbench.scenarios.base import Scenario

LAWS = ("composition", "logit")


def _posterior(scores, roles, law):
    a, b, c, d = (scores[:, i] for i in roles)
    if law == "composition":
        return np.maximum(np.minimum(a, b), np.minimum(c, d))
    if law == "logit":
        return expit(0.2 + 0.7 * logit(a) + 0.7 * logit(b))
    raise ValueError(f"law must be one of {LAWS}")


def _probability(value):
    value = float(value)
    if not 0 <= value <= 1:
        raise ValueError("corner_probability must be finite and in [0, 1]")
    return value


@dataclass(frozen=True)
class CompositionalDraw:
    """One anonymous input matrix with a common label-uniform draw under both laws.

    corner_rows records the sampling component. A full-grid draw can also happen
    to lie at a corner. The roles, mask and posterior are evaluator diagnostics.
    """

    preds: np.ndarray
    uniform: np.ndarray
    roles: tuple[int, ...]
    corner_rows: np.ndarray

    @property
    def scores(self):
        return self.preds[:, :, 1]

    def probability(self, law="composition"):
        return _posterior(self.scores, self.roles, law)

    def labels(self, law="composition"):
        return (self.uniform < self.probability(law)).astype(int)


class CompositionalSwitch(Scenario):
    """Six anonymous probabilities with an alternating MIN/MAX or additive logit law."""

    primary_metric = "brier"
    oracle_description = "Known conditional probability given the six anonymous scores."

    def __init__(self, law="composition", role_seed=0, corner_probability=0.5, **kw):
        if law not in LAWS:
            raise ValueError(f"law must be one of {LAWS}")
        self.law = law
        self.corner_probability = _probability(corner_probability)
        super().__init__(
            law=law,
            role_seed=role_seed,
            corner_probability=self.corner_probability,
            **kw,
        )
        rng = np.random.default_rng(np.random.SeedSequence([role_seed, 229]))
        self.roles = tuple(map(int, rng.permutation(6)[:4]))

    @property
    def n_classes(self):
        return 2

    def model_ids(self):
        return [f"p{i}" for i in range(6)]

    def draw(self, n, rng, corner_probability=None):
        """Generate paired assignments; changing the mixture reuses both score panels."""
        p = (
            self.corner_probability
            if corner_probability is None
            else _probability(corner_probability)
        )
        corner_rows = rng.random(n) < p
        bits = rng.integers(0, 2, size=(n, 6))
        interior_ticks = rng.integers(1, 64, size=(n, 6))
        scores = np.where(corner_rows[:, None], 1 + 62 * bits, interior_ticks) / 64
        return CompositionalDraw(
            np.stack((1 - scores, scores), axis=-1),
            rng.random(n),
            self.roles,
            corner_rows,
        )

    def _generate(self, n, rng):
        sample = self.draw(n, rng)
        return sample.preds, sample.labels(self.law), {}

    def bayes_oracle(self, preds, latent):
        q = _posterior(np.asarray(preds)[:, :, 1], self.roles, self.law)
        return np.column_stack((1 - q, q))

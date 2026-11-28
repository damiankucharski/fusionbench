"""Correlated-error pathology.

A large block of models share a correlated noise component (their mistakes
coincide); a few "diverse" models err independently. All have equal marginal
accuracy, so naive *equal-weight* averaging / top-k overcounts the redundant
block. The Bayes-optimal rule is the Gaussian linear discriminant
``2 s^T Sigma^{-1} t``, which down-weights the correlated block — a *weighted
linear* rule, so learned-weight methods (logit stacker, NSGA-III) win here, not
nonlinear gating.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from fusionbench.scenarios import calibrated as cal
from fusionbench.scenarios.base import Scenario


class CorrelatedRedundancy(Scenario):
    task = "binary"
    expected_winners: ClassVar[list[str]] = ["stacking_logit", "nsga3", "okapi"]
    oracle_description = "Gaussian discriminant down-weighting the correlated block"

    def __init__(
        self,
        n_block: int = 8,
        n_diverse: int = 2,
        accuracy: float = 0.75,
        corr: float = 0.85,
        **kw,
    ):
        super().__init__(
            n_block=n_block, n_diverse=n_diverse, accuracy=accuracy, corr=corr, **kw
        )
        self.n_block, self.n_diverse = n_block, n_diverse
        self.accuracy, self.corr = accuracy, corr

    @property
    def n_classes(self) -> int:
        return 2

    def model_ids(self):
        return [f"block{i}" for i in range(self.n_block)] + [
            f"diverse{i}" for i in range(self.n_diverse)
        ]

    def _covariance(self) -> np.ndarray:
        k = self.n_block + self.n_diverse
        sigma = np.eye(k)
        for i in range(self.n_block):
            for j in range(self.n_block):
                if i != j:
                    sigma[i, j] = self.corr
        return sigma

    def _generate(self, n, rng):
        s = cal.acc_to_strength(self.accuracy)
        w = self.corr
        signal = (2.0 * rng.integers(0, 2, n) - 1.0) * s
        y = (signal > 0).astype(int)  # signal sign encodes label
        shared = rng.standard_normal(n)
        t_cols = []
        for _ in range(self.n_block):
            e = rng.standard_normal(n)
            t_cols.append(signal + np.sqrt(w) * shared + np.sqrt(1 - w) * e)
        for _ in range(self.n_diverse):
            t_cols.append(signal + rng.standard_normal(n))
        t = np.stack(t_cols, axis=1)  # [N, K]
        p1 = cal.sigmoid(2.0 * s * t)
        preds = np.stack(
            [cal.to_binary_proba(p1[:, m]) for m in range(t.shape[1])], axis=1
        )
        latent = {"t": t, "s": s}
        return preds, y, latent

    def bayes_oracle(self, preds, latent):
        k = self.n_block + self.n_diverse
        sigma_inv = np.linalg.inv(self._covariance())
        s_vec = np.full(k, latent["s"])
        return cal.binary_gaussian_oracle(latent["t"], s_vec, sigma_inv)


class CalibratedSuppressor(Scenario):
    """Two individually calibrated providers with potentially opposite joint weights.

    The standard deviations of the two Gaussian logits are 2 and 1. Changing
    their class-conditional correlation preserves both marginal calibration curves.
    Kahn (2004) derives the affine aggregate for class-conditional Gaussian
    log odds with shared covariance. Here unequal marginal variances make the
    weaker provider's joint coefficient negative when correlation exceeds 0.5.
    Kahn, "A Generative Bayesian Model for Aggregating Experts' Probabilities"
    (UAI 2004), https://arxiv.org/abs/1207.4144.
    """

    primary_metric = "brier"
    expected_winners: ClassVar[list[str]] = ["stacking_logit"]
    oracle_description = (
        "Gaussian likelihood ratio of two correlated calibrated source logits"
    )

    def __init__(self, corr: float = 0.85, **kw):
        if not -1 < corr < 1:
            raise ValueError("corr must be strictly between -1 and 1")
        super().__init__(corr=corr, **kw)
        self.corr = float(corr)
        self.covariance = np.array([[4.0, 2 * corr], [2 * corr, 1.0]])

    @property
    def n_classes(self):
        return 2

    def model_ids(self):
        return ["source_0", "source_1"]

    def _generate(self, n, rng):
        y = rng.integers(2, size=n)
        class_sign = 2 * y[:, None] - 1
        logits = rng.standard_normal((n, 2)) @ np.linalg.cholesky(self.covariance).T
        logits += class_sign * (0.5 * self.covariance.diagonal())
        preds = np.stack((cal.sigmoid(-logits), cal.sigmoid(logits)), axis=2)
        return preds, y, {}

    def bayes_oracle(self, preds, latent):
        weights = np.linalg.solve(self.covariance, self.covariance.diagonal())
        logits = np.log(preds[:, :, 1]) - np.log(preds[:, :, 0])
        return cal.to_binary_proba(cal.sigmoid(logits @ weights))

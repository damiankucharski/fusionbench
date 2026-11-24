"""One unreliable sensor and a Gaussian block of correlated stable evidence."""

from __future__ import annotations

import numpy as np
from scipy.special import expit

from fusionbench.scenarios.base import Scenario
from fusionbench.scenarios.calibrated import acc_to_strength


class PolarityFault(Scenario):
    primary_metric = "brier"
    oracle_description = (
        "Bounded log odds for a sign-flipping sensor plus the Gaussian discriminant for the stable block"
    )

    def __init__(self, fault_rate=0.1, correlation=0.65, **kw):
        if not 0 <= fault_rate <= 1 or not 0 <= correlation < 1:
            raise ValueError("use fault rate in [0,1] and correlation in [0,1)")
        super().__init__(fault_rate=fault_rate, correlation=correlation, **kw)
        self.fault_rate = fault_rate
        self.correlation = correlation
        self.strengths = np.array(
            [acc_to_strength(a) for a in (0.95, 0.85, 0.82, 0.79, 0.77, 0.73, 0.68)]
        )

    @property
    def n_classes(self):
        return 2

    def model_ids(self):
        return ["unstable", *[f"stable_{i}" for i in range(1, 7)]]

    def _generate(self, n, rng):
        y = rng.integers(2, size=n)
        common = rng.normal(size=n)
        noise = rng.normal(size=(n, len(self.strengths)))
        noise[:, 1:] = (
            np.sqrt(self.correlation) * common[:, None] + np.sqrt(1 - self.correlation) * noise[:, 1:]
        )
        raw = (2 * y - 1)[:, None] * self.strengths + noise
        flipped = rng.random(n) < self.fault_rate
        raw[flipped, 0] *= -1
        logits = 2 * self.strengths * raw
        scores = np.stack([expit(-logits), expit(logits)], axis=-1)
        return scores, y, {"raw": raw, "flipped": flipped}

    def posterior(self, scores, *, fault_rate, correlation):
        # Both output channels are computed directly for numerical stability.
        safe = np.maximum(scores, np.finfo(float).tiny)
        logits = np.log(safe[:, :, 1]) - np.log(safe[:, :, 0])
        if fault_rate == 0:
            unstable = logits[:, 0]
        elif fault_rate == 1:
            unstable = -logits[:, 0]
        else:
            p0, p1 = scores[:, 0, 0], scores[:, 0, 1]
            unstable = np.log((1 - fault_rate) * p1 + fault_rate * p0) - np.log(
                (1 - fault_rate) * p0 + fault_rate * p1
            )
        s = self.strengths[1:]
        cov = np.full((len(s), len(s)), correlation, dtype=float)
        np.fill_diagonal(cov, 1.0)
        # Evidence = 2 s^T Sigma^{-1} raw, expressed using input logits.
        stable = logits[:, 1:] @ (np.linalg.solve(cov, s) / s)
        return expit(unstable + stable)

    def bayes_oracle(self, preds, latent):
        p = self.posterior(preds, fault_rate=self.fault_rate, correlation=self.correlation)
        return np.stack([1 - p, p], axis=1)

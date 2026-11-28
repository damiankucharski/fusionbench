"""Calibration-driven pathologies.

These are judged primarily by log-loss, not accuracy: miscalibration leaves the
argmax decision (and thus accuracy/F1) unchanged but ruins probability quality.
Methods that recalibrate in logit space (logistic stackers) win; plain averaging
and confidence-routing do not. Another honest, non-OKAPI-favouring region.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from scipy.special import expit, log_ndtr, logit

from fusionbench.scenarios import calibrated as cal
from fusionbench.scenarios.base import Scenario


def _detector_log_ratio(p, accuracy):
    """Log likelihood ratio of the recorded, possibly clipped Gaussian detector."""
    s = cal.acc_to_strength(accuracy)
    cutoff = logit(1e-6)
    lower_tail = log_ndtr(cutoff / (2 * s) - s) - log_ndtr(cutoff / (2 * s) + s)
    return np.where(
        p <= 1e-6, lower_tail, np.where(p >= 1 - 1e-6, -lower_tail, logit(p))
    )


class Miscalibration(Scenario):
    """Scale historical logit multipliers k to k**temperature_range (0..1)."""

    task = "binary"
    primary_metric = "log_loss"
    expected_winners: ClassVar[list[str]] = ["stacking_logit", "stacking_lr"]
    oracle_description = "per-model recalibration then logit-sum"

    def __init__(self, k: int = 6, accuracy: float = 0.78, temperature_range=1.0, **kw):
        if not 0 <= temperature_range <= 1:
            raise ValueError("temperature_range must lie in [0, 1]")
        self.temperature_range = temperature_range
        if temperature_range != 1:
            kw["temperature_range"] = temperature_range
        super().__init__(k=k, accuracy=accuracy, **kw)
        self.k, self.accuracy = k, accuracy

    @property
    def n_classes(self):
        return 2

    def model_ids(self):
        return [f"m{i}" for i in range(self.k)]

    def _generate(self, n, rng):
        y = rng.integers(0, 2, n)
        # Distortion factors: a mix of over- (k>1) and under-confident (k<1) models.
        ks = np.linspace(0.5, 2.5, self.k)
        if self.temperature_range != 1:
            ks = ks**self.temperature_range
        cols = []
        for k in ks:
            true_p = cal.binary_calibrated(y, self.accuracy, rng)
            obs = cal.sharpen(true_p, k)
            cols.append(cal.to_binary_proba(obs))
        return np.stack(cols, axis=1), y, {"ks": ks}

    def bayes_oracle(self, preds, latent):
        ks = latent["ks"]
        p1 = preds[:, :, 1]
        if self.temperature_range == 1:
            # The published v1 reference uses its original 1e-6 clipping.
            return cal.to_binary_proba(
                cal.sigmoid((cal.logit(p1) / ks[None, :]).sum(axis=1))
            )
        # Invert the observed temperature before handling mass at the detector
        # clipping limits. Clipping the sharpened scores would lose information.
        raw = expit(logit(p1) / ks[None, :])
        lower = cal.sharpen(np.full(len(ks), 1e-6), ks)
        upper = cal.sharpen(np.full(len(ks), 1 - 1e-6), ks)
        raw = np.where(p1 <= lower, 1e-6, np.where(p1 >= upper, 1 - 1e-6, raw))
        return cal.to_binary_proba(
            expit(_detector_log_ratio(raw, self.accuracy).sum(axis=1))
        )


class MulticlassCommittee(Scenario):
    """K independent calibrated multiclass models (multiclass control)."""

    task = "multiclass"
    expected_winners: ClassVar[list[str]] = ["stacking_lr", "simple_average"]
    oracle_description = "strength-weighted product from independent Gaussian logits"

    def __init__(self, k: int = 8, n_classes: int = 5, accuracy: float = 0.6, **kw):
        super().__init__(k=k, n_classes=n_classes, accuracy=accuracy, **kw)
        self.k, self._nc, self.accuracy = k, n_classes, accuracy

    @property
    def n_classes(self):
        return self._nc

    def model_ids(self):
        return [f"m{i}" for i in range(self.k)]

    def _generate(self, n, rng):
        y = rng.integers(0, self._nc, n)
        cols = [
            cal.multiclass_calibrated(y, self.accuracy, self._nc, rng)
            for _ in range(self.k)
        ]
        return np.stack(cols, axis=1), y, {}

    def bayes_oracle(self, preds, latent):
        return cal.multiclass_gaussian_oracle(preds, self.accuracy)

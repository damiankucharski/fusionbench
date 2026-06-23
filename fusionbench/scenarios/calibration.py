"""Calibration-driven pathologies.

These are judged primarily by log-loss, not accuracy: miscalibration leaves the
argmax decision (and thus accuracy/F1) unchanged but ruins probability quality.
Methods that recalibrate in logit space (logistic stackers) win; plain averaging
and confidence-routing do not. Another honest, non-OKAPI-favouring region.
"""

from __future__ import annotations

import numpy as np

from fusionbench.scenarios import calibrated as cal
from fusionbench.scenarios.base import Scenario


class Miscalibration(Scenario):
    task = "binary"
    primary_metric = "log_loss"
    expected_winners = ["stacking_logit", "stacking_lr"]
    oracle_description = "per-model recalibration then logit-sum"

    def __init__(self, k: int = 6, accuracy: float = 0.78, **kw):
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
        cols = []
        for k in ks:
            true_p = cal.binary_calibrated(y, self.accuracy, rng)
            obs = cal.sharpen(true_p, k)
            cols.append(cal.to_binary_proba(obs))
        return np.stack(cols, axis=1), y, {"ks": ks}

    def bayes_oracle(self, preds, latent):
        ks = latent["ks"]
        p1 = preds[:, :, 1]
        recal_logits = cal.logit(p1) / ks[None, :]  # undo per-model distortion
        return cal.to_binary_proba(cal.sigmoid(recal_logits.sum(axis=1)))


class MulticlassCommittee(Scenario):
    """K independent calibrated multiclass models (multiclass control)."""

    task = "multiclass"
    expected_winners = ["stacking_lr", "simple_average"]
    oracle_description = "product of probabilities (independent multiclass experts)"

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
        cols = [cal.multiclass_calibrated(y, self.accuracy, self._nc, rng)
                for _ in range(self.k)]
        return np.stack(cols, axis=1), y, {}

    def bayes_oracle(self, preds, latent):
        return cal.multiclass_independent_oracle(preds)

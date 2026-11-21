"""Conditionally-independent pathologies.

For calibrated, conditionally-independent models the Bayes-optimal combiner is
the **sum of logits** (binary) / **product of probabilities** (multiclass), not
the mean. A logit-space stacker is therefore near-oracle here, while naive
prob-space averaging is dragged down by weak/garbage members. These scenarios
keep the benchmark honest: they are *not* won by nonlinear gating methods.
"""

from __future__ import annotations

import numpy as np

from fusionbench.scenarios import calibrated as cal
from fusionbench.scenarios.base import Scenario


class Committee(Scenario):
    """K equal-quality calibrated models with independent errors (control)."""

    task = "binary"
    expected_winners = ["stacking_logit", "simple_average"]
    oracle_description = "logit-sum of K independent calibrated experts"

    def __init__(self, k: int = 10, accuracy: float = 0.75, **kw):
        super().__init__(k=k, accuracy=accuracy, **kw)
        self.k, self.accuracy = k, accuracy

    @property
    def n_classes(self) -> int:
        return 2

    def model_ids(self) -> list[str]:
        return [f"m{i}" for i in range(self.k)]

    def _generate(self, n, rng):
        y = rng.integers(0, 2, n)
        cols = [cal.to_binary_proba(cal.binary_calibrated(y, self.accuracy, rng))
                for _ in range(self.k)]
        return np.stack(cols, axis=1), y, {}

    def bayes_oracle(self, preds, latent):
        return cal.binary_independent_oracle(preds)


class DeadWeight(Scenario):
    """Strong models mixed with near-random garbage models.

    Garbage contributes ~zero log-odds, so the Bayes oracle ignores it
    automatically; *prob-space averaging over all models* is the method that
    fails (pulled toward 0.5). Selection / weighting / logit-sum recover.
    """

    task = "binary"
    expected_winners = ["stacking_logit", "top_5", "nsga3", "okapi"]
    oracle_description = "logit-sum; garbage members carry ~0 evidence"

    def __init__(self, n_good: int = 4, n_bad: int = 6, accuracy: float = 0.82, **kw):
        super().__init__(n_good=n_good, n_bad=n_bad, accuracy=accuracy, **kw)
        self.n_good, self.n_bad, self.accuracy = n_good, n_bad, accuracy

    @property
    def n_classes(self) -> int:
        return 2

    def model_ids(self):
        return [f"good{i}" for i in range(self.n_good)] + [f"bad{i}" for i in range(self.n_bad)]

    def _generate(self, n, rng):
        y = rng.integers(0, 2, n)
        cols = [cal.to_binary_proba(cal.binary_calibrated(y, self.accuracy, rng))
                for _ in range(self.n_good)]
        cols += [cal.to_binary_proba(cal.random_binary(y, rng)) for _ in range(self.n_bad)]
        return np.stack(cols, axis=1), y, {}

    def bayes_oracle(self, preds, latent):
        # Garbage models are independent of y, so their log-likelihood ratio is
        # identically 0 (their realised logit is noise, not evidence). The Bayes
        # oracle therefore sums logits over the informative models only.
        good = preds[:, : self.n_good, :]
        return cal.binary_independent_oracle(good)


class Needle(DeadWeight):
    """Needle-in-a-haystack: a few informative models among many noise models.

    Stresses meta-learner robustness to a wide feature space and rewards
    parsimonious selection (compact, auditable, cheap-at-inference rules).
    """

    expected_winners = ["okapi", "stacking_logit", "top_5"]
    oracle_description = "logit-sum over the few informative models"

    def __init__(self, n_good: int = 3, n_bad: int = 97, accuracy: float = 0.80, **kw):
        super().__init__(n_good=n_good, n_bad=n_bad, accuracy=accuracy, **kw)


class Saturated(Scenario):
    """Redundant pool of near-identical, individually-excellent models.

    No fusion headroom: the ceiling is essentially a single competent model.
    All reasonable methods tie on accuracy; the benchmark value is the *no-harm*
    check and the parsimony/cost differentiator (OKAPI should abstain to a tiny
    tree). Multiclass.
    """

    task = "multiclass"
    expected_winners = ["best_single", "simple_average", "okapi"]
    oracle_description = "Gaussian posterior weighted for shared logit noise"

    def __init__(self, k: int = 6, n_classes: int = 4, accuracy: float = 0.95,
                 corr: float = 0.95, **kw):
        super().__init__(k=k, n_classes=n_classes, accuracy=accuracy, corr=corr, **kw)
        self.k, self._nc, self.accuracy, self.corr = k, n_classes, accuracy, corr

    @property
    def n_classes(self) -> int:
        return self._nc

    def model_ids(self):
        return [f"m{i}" for i in range(self.k)]

    def _generate(self, n, rng):
        y = rng.integers(0, self._nc, n)
        shared = rng.standard_normal((n, self._nc))
        cols = [cal.multiclass_calibrated(y, self.accuracy, self._nc, rng,
                                          shared_noise=shared, shared_weight=self.corr)
                for _ in range(self.k)]
        return np.stack(cols, axis=1), y, {}

    def bayes_oracle(self, preds, latent):
        return cal.multiclass_gaussian_oracle(preds, self.accuracy, shared_weight=self.corr)

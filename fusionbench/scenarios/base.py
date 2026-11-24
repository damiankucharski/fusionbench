"""Scenario interface for generating provider predictions and class labels.

Committee-style generators draw labels first, then sample provider outputs
conditional on them. Logical-gate scenarios sample latent conditions that
determine labels and inform noisy detectors. Switch scenarios draw provider
scores first, then sample labels from a conditional law specified on the scores.

The optional clean-domain oracle gives the conditional class probabilities
when known. Scenario.build draws fit and test banks from the same generator.
Finite-sample results vary across draws, and transfer benchmarks can evaluate
new distributions by changing the generator or perturbing its outputs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np

from fusionbench.types import Dataset, Task


class Scenario(ABC):
    task: Task = "binary"
    expected_winners: ClassVar[list[str]] = []
    oracle_description: str = ""
    # Headline metric for this pathology (e.g. calibration scenarios use log_loss).
    primary_metric: str = "macro_f1"

    def __init__(self, name: str | None = None, **knobs):
        self.knobs = knobs
        self.name = name or self.default_name(**knobs)

    def default_name(self, **knobs) -> str:
        cls = type(self).__name__.lower()
        if knobs:
            return f"{cls}_" + "_".join(f"{k}{v}" for k, v in knobs.items())
        return cls

    # --- subclass API ----------------------------------------------------- #
    @property
    @abstractmethod
    def n_classes(self) -> int: ...

    @abstractmethod
    def model_ids(self) -> list[str]: ...

    @abstractmethod
    def _generate(self, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, dict]:
        """Return ``(preds[N, K, C], y[N], latent)`` for one split.

        ``latent`` holds optional generator metadata, such as deferral masks
        used to compute a known clean-domain oracle. An empty dict suffices.
        """

    def bayes_oracle(self, preds: np.ndarray, latent: dict) -> np.ndarray | None:
        """Return the clean-domain posterior ``[N, C]`` when known, or None."""
        return None

    # --- driver ----------------------------------------------------------- #
    def build(self, n_per_split: int = 50_000, seed: int = 0) -> Dataset:
        rng = np.random.default_rng(seed)
        fit_preds, fit_y, fit_latent = self._generate(n_per_split, rng)
        test_preds, test_y, test_latent = self._generate(n_per_split, rng)
        return Dataset(
            name=self.name,
            task=self.task,
            n_classes=self.n_classes,
            model_ids=self.model_ids(),
            fit_preds=fit_preds,
            fit_y=fit_y,
            test_preds=test_preds,
            test_y=test_y,
            fit_oracle=self.bayes_oracle(fit_preds, fit_latent),
            test_oracle=self.bayes_oracle(test_preds, test_latent),
            oracle_description=self.oracle_description,
            expected_winners=list(self.expected_winners),
            meta={"scenario": type(self).__name__, "knobs": self.knobs,
                  "primary_metric": self.primary_metric},
        )

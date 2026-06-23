"""Scenario base class.

A scenario is a parametric generator of one base-model *pathology*. Generation
runs in the natural causal direction:

    1. draw the true label ``y`` from a class prior (independent of any model);
    2. emit each base model's prediction as a calibrated, noisy observation of
       ``y`` with a pathology-specific error process.

The generator never references any fusion method's representation (no
mean/max/min/far over the predictions). The performance ceiling is the
**analytic Bayes posterior** ``P(y | predictions)`` derived from the DGP,
implemented in :meth:`bayes_oracle`. Both splits come from the identical i.i.d.
process, so validation-test reliability holds by construction.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from fusionbench.types import Dataset, Task


class Scenario(ABC):
    task: Task = "binary"
    expected_winners: list[str] = []
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

        ``latent`` carries whatever the analytic oracle needs (e.g. per-model
        signal strengths, deferral masks) so the Bayes posterior is exact.
        """

    @abstractmethod
    def bayes_oracle(self, preds: np.ndarray, latent: dict) -> np.ndarray:
        """Analytic Bayes posterior ``P(y | predictions)`` -> ``[N, C]``.

        Derived from the data-generating process, independent of any fusion
        method. This is the information-theoretic ceiling for the scenario.
        """

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

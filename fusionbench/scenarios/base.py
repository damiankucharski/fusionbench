"""Scenario interface for generating provider predictions and class labels.

Committee-style generators draw labels first, then sample provider outputs
conditional on them. Logical-gate scenarios sample latent conditions that
determine labels and inform noisy detectors. Switch scenarios draw provider
scores first, then sample labels from a conditional law specified on the scores.

The optional domain oracle gives the conditional class probabilities when
known. Scenario.build draws fit and test banks from the same generator.
Scenarios with a domain shift can specialize sample_split for each split.
Finite-sample results vary across draws, and transfer benchmarks can evaluate
new distributions by changing the generator or perturbing its outputs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from operator import index
from typing import ClassVar

import numpy as np

from fusionbench.types import Dataset, Task


class Scenario(ABC):
    task: Task = "binary"
    expected_winners: ClassVar[list[str]] = []
    oracle_description: str = ""
    # Headline metric for this pathology (e.g. calibration scenarios use log_loss).
    primary_metric: str = "macro_f1"
    evaluation_group_fields: ClassVar[tuple[str, ...]] = ()
    _registry_metadata: dict | None = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Give each override both spellings at the class that defines it.
        # Inherited methods and super() then follow Python's ordinary MRO.
        if "sample" in cls.__dict__ and "_generate" not in cls.__dict__:
            cls._generate = cls.sample
        elif "_generate" in cls.__dict__ and "sample" not in cls.__dict__:
            cls.sample = cls._generate

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
    def sample(
        self, n: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """Return ``(preds[N, K, C], y[N], latent)`` for one split.

        Draw all randomness from rng. The class axis follows integer labels
        and the provider axis follows model_ids(). The latent dictionary
        holds optional oracle inputs and may be empty.
        """
        raise NotImplementedError

    _generate = sample

    def sample_split(self, n: int, rng: np.random.Generator, split: str):
        """Draw a named domain. Ordinary scenarios share a law across splits."""
        if split not in ("fit", "test"):
            raise ValueError("split must be 'fit' or 'test'")
        return self.sample(n, rng)

    def bayes_oracle(self, preds: np.ndarray, latent: dict) -> np.ndarray | None:
        """Return the clean-domain posterior ``[N, C]`` when known, or None."""
        return None

    # --- driver ----------------------------------------------------------- #
    def build(self, n_per_split: int = 50_000, seed: int = 0) -> Dataset:
        """Draw fit then test from one private NumPy generator initialized by seed."""
        meta = {
            "scenario": type(self).__name__,
            "knobs": self.knobs,
            "primary_metric": self.primary_metric,
        }
        if self._registry_metadata is not None:
            # Publicly configured datasets record replayable constructor and RNG options.
            from fusionbench import __version__

            if isinstance(n_per_split, (bool, np.bool_)) or index(n_per_split) < 1:
                raise ValueError("n_per_split must be a positive integer")
            if isinstance(seed, (bool, np.bool_)) or index(seed) < 0:
                raise ValueError("seed must be a nonnegative integer")
            meta.update(
                deepcopy(self._registry_metadata),
                seed=index(seed),
                n_per_split=index(n_per_split),
                fusionbench_version=__version__,
            )
        rng = np.random.default_rng(seed)
        fit_preds, fit_y, fit_latent = self.sample_split(n_per_split, rng, "fit")
        test_preds, test_y, test_latent = self.sample_split(n_per_split, rng, "test")
        if self.evaluation_group_fields:
            meta["evaluation_groups"] = {
                split: {key: np.asarray(latent[key]).tolist() for key in self.evaluation_group_fields}
                for split, latent in (("fit", fit_latent), ("test", test_latent))
            }
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
            meta=meta,
        )

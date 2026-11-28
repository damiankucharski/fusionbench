"""Core data types for fusionbench.

A :class:`Dataset` holds stacked base-model predictions for two splits:

- ``fit``  : the data every fusion method is allowed to select / fit on
             (the "Train for OKAPI" / validation pool).
- ``test`` : held-out evaluation data. IID scenarios use the same generator
             parameters on both splits. The transfer suite varies the test
             conditions explicitly. Ranking agreement is measured on held-out rows.

Predictions are stored stacked as ``[N, K, C]`` (samples x base-models x
classes). For binary tasks ``C == 2``; rows over the class axis sum to 1.
Ground truth is integer class indices ``[N]`` (single-label). Multilabel
(label-vector ground truth) is phase 2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

Task = Literal["binary", "multiclass", "multilabel"]


def _numpy_json(value):
    """Represent NumPy knobs and scalar metadata in the public JSON format."""
    if isinstance(value, (np.ndarray, np.generic)):
        return value.tolist()
    raise TypeError(f"Cannot record {type(value).__name__} as JSON metadata")


def _check_probabilities(name: str, values: np.ndarray, shape: tuple[int, ...]) -> None:
    if not isinstance(values, np.ndarray) or values.shape != shape:
        raise ValueError(f"{name} must be an array with shape {shape}")
    # Reductions check large or strided tensors without copying their contents.
    if values.dtype.kind not in "fiu" or not (values.min() >= 0 and values.max() <= 1):
        raise ValueError(f"{name} must contain finite, real probabilities in [0, 1]")
    sums = values.sum(axis=-1)
    if max(abs(float(sums.min()) - 1), abs(float(sums.max()) - 1)) > 1e-5:
        raise ValueError(
            f"{name} probabilities must sum to 1 over classes (tolerance 1e-5)"
        )


@dataclass
class Dataset:
    """A generated benchmark dataset (one scenario instance)."""

    name: str
    task: Task
    n_classes: int
    model_ids: list[str]

    fit_preds: np.ndarray  # [N_fit, K, C] float
    fit_y: np.ndarray  # [N_fit] int
    test_preds: np.ndarray  # [N_test, K, C] float
    test_y: np.ndarray  # [N_test] int

    # The analytic Bayes posterior P(y | predictions) under the data-generating
    # process: the information-theoretic ceiling, derived independently of any
    # fusion method's representation. Stored per split as [N, C].
    fit_oracle: np.ndarray | None = None
    test_oracle: np.ndarray | None = None
    oracle_description: str = ""

    # Methods we *expect* to do well (registry names). Used by the
    # characterization report to check the benchmark discriminates as designed.
    expected_winners: list[str] = field(default_factory=list)

    # Arbitrary scenario parameters (knobs) recorded for provenance.
    meta: dict = field(default_factory=dict)

    @property
    def n_models(self) -> int:
        return len(self.model_ids)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Check shapes, class labels and probability values on both splits."""
        if self.task not in ("binary", "multiclass"):
            raise ValueError(
                "task must be 'binary' or 'multiclass' for single-label datasets"
            )
        if not isinstance(self.n_classes, (int, np.integer)) or self.n_classes < 2:
            raise ValueError("n_classes must be an integer of at least 2")
        if self.task == "binary" and self.n_classes != 2:
            raise ValueError("binary datasets require n_classes=2")
        if (
            not isinstance(self.model_ids, (list, tuple))
            or not self.model_ids
            or not all(isinstance(name, str) and name for name in self.model_ids)
        ):
            raise ValueError("model_ids must be a nonempty sequence of provider names")
        if len(set(self.model_ids)) != self.n_models:
            raise ValueError("model_ids must contain unique provider names")

        for split, labels, preds, oracle in (
            ("fit", self.fit_y, self.fit_preds, self.fit_oracle),
            ("test", self.test_y, self.test_preds, self.test_oracle),
        ):
            if (
                not isinstance(labels, np.ndarray)
                or labels.ndim != 1
                or not labels.size
                or labels.dtype.kind not in "biu"
            ):
                raise ValueError(
                    f"{split}_y must be a nonempty vector of integer class labels"
                )
            if labels.min() < 0 or labels.max() >= self.n_classes:
                raise ValueError(f"{split}_y labels must be in [0, {self.n_classes})")
            _check_probabilities(
                f"{split}_preds", preds, (len(labels), self.n_models, self.n_classes)
            )
            if oracle is not None:
                _check_probabilities(
                    f"{split}_oracle", oracle, (len(labels), self.n_classes)
                )

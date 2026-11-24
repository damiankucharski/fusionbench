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


@dataclass
class Dataset:
    """A generated benchmark dataset (one scenario instance)."""

    name: str
    task: Task
    n_classes: int
    model_ids: list[str]

    fit_preds: np.ndarray  # [N_fit, K, C] float
    fit_y: np.ndarray      # [N_fit] int
    test_preds: np.ndarray  # [N_test, K, C] float
    test_y: np.ndarray      # [N_test] int

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
        assert self.fit_preds.ndim == 3, "fit_preds must be [N, K, C]"
        assert self.test_preds.ndim == 3, "test_preds must be [N, K, C]"
        assert self.fit_preds.shape[1] == self.test_preds.shape[1] == len(self.model_ids)
        assert self.fit_preds.shape[2] == self.test_preds.shape[2] == self.n_classes
        assert self.fit_preds.shape[0] == self.fit_y.shape[0]
        assert self.test_preds.shape[0] == self.test_y.shape[0]
        for orc, n in ((self.fit_oracle, self.fit_y.shape[0]),
                       (self.test_oracle, self.test_y.shape[0])):
            if orc is not None:
                assert orc.shape == (n, self.n_classes), "oracle must be [N, C]"

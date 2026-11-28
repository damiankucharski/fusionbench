"""A calibrated anchor with conditionally redundant candidates and a matched positive control."""

from __future__ import annotations

from numbers import Real

import numpy as np

from fusionbench.scenarios.base import Scenario
from fusionbench.scenarios.calibrated import to_binary_proba


class RedundantPool(Scenario):
    """Anonymous candidate opinions derived from a calibrated anchor and noisy bits.

    In the positive arm precisely one candidate gets a label-informative bit.
    Its score marginal conditional on the anchor is identical in both arms.
    The role seed fixes a column assignment across fit and test.
    """

    n_classes = 2
    primary_metric = "brier"
    oracle_description = "The anchor in the null arm; the conditional posterior of the informed candidate in the positive arm"

    def __init__(
        self,
        n_candidates: int = 15,
        positive_control: bool = False,
        candidate_accuracy: float = 0.8,
        role_seed: int = 0,
        name: str | None = None,
    ):
        for key, value in (("n_candidates", n_candidates), ("role_seed", role_seed)):
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or value < 0
            ):
                raise ValueError(f"{key} must be a nonnegative integer")
        if not isinstance(positive_control, (bool, np.bool_)):
            raise TypeError("positive_control must be boolean")
        if positive_control and n_candidates == 0:
            raise ValueError("The positive control requires at least one candidate")
        if (
            isinstance(candidate_accuracy, (bool, np.bool_))
            or not isinstance(candidate_accuracy, Real)
            or not 0.5 <= candidate_accuracy <= 1
        ):
            raise ValueError("candidate_accuracy must lie in [0.5, 1]")
        super().__init__(
            name=name,
            n_candidates=int(n_candidates),
            positive_control=bool(positive_control),
            candidate_accuracy=float(candidate_accuracy),
            role_seed=int(role_seed),
        )
        self.n_candidates = int(n_candidates)
        self.positive_control = bool(positive_control)
        self.candidate_accuracy = float(candidate_accuracy)
        columns = np.random.default_rng(role_seed).permutation(self.n_candidates + 1)
        self.anchor_index = int(columns[0])
        self.candidate_indices = tuple(map(int, columns[1:]))

    def model_ids(self):
        return [f"model_{i}" for i in range(self.n_candidates + 1)]

    def sample(self, n, rng):
        # Consuming one parent draw keeps the next split paired across pool sizes.
        local = np.random.default_rng(rng.integers(0, 2**63, dtype=np.int64))
        p = local.beta(2, 2, n)
        y = (local.random(n) < p).astype(np.int64)
        scores = np.empty((n, self.n_candidates + 1, 2))
        scores[:, self.anchor_index] = to_binary_proba(p)
        q = self.candidate_accuracy
        marginal = p * q + (1 - p) * (1 - q)
        for j, column in enumerate(self.candidate_indices):
            bit_probability = (
                np.where(y, q, 1 - q) if j == 0 and self.positive_control else marginal
            )
            vote = local.random(n) < bit_probability
            like_one = np.where(vote, q, 1 - q)
            mass_one = p * like_one
            posterior = mass_one / (mass_one + (1 - p) * (1 - like_one))
            scores[:, column] = to_binary_proba(posterior)
        return scores, y, {}

    def bayes_oracle(self, preds, latent):
        column = (
            self.candidate_indices[0] if self.positive_control else self.anchor_index
        )
        return np.asarray(preds)[:, column, :].copy()

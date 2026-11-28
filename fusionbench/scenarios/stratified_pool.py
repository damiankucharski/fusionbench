"""A matched-population fusion task with exact allocation across label and difficulty."""

from numbers import Real

import numpy as np

from fusionbench.scenarios.base import Scenario


class StratifiedPool(Scenario):
    """Interpolate between the arithmetic mean and one more informative provider.

    Conditional on report amplitude, the eight fair report signs are independent.
    Each batch stratifies by amplitude and realized label. Its per-row marginal
    is the same as sampling signs first and then drawing labels from the oracle.
    For a requested size not divisible by four, randomly allocate the remainder
    to distinct strata, so that each count differs from another by at most one.
    """

    n_classes = 2
    primary_metric = "brier"
    evaluation_group_fields = ("high_amplitude",)
    oracle_description = (
        "A known convex combination of the arithmetic mean and provider zero"
    )

    def __init__(self, specialization=0.25, name=None):
        if (
            isinstance(specialization, (bool, np.bool_))
            or not isinstance(specialization, Real)
            or not 0 <= specialization <= 1
        ):
            raise ValueError("specialization must be a probability in [0, 1]")
        super().__init__(name=name, specialization=float(specialization))
        self.specialization = float(specialization)

    def model_ids(self):
        return [f"provider_{i}" for i in range(8)]

    def bayes_oracle(self, preds, latent):
        positive = preds[:, :, 1]
        q = (1 - self.specialization) * positive.mean(
            axis=1
        ) + self.specialization * positive[:, 0]
        return np.column_stack((1 - q, q))

    def support(self):
        """All 512 equiprobable report vectors, their masses and per-row positive probability."""
        signs = 2 * np.indices((2,) * 8).reshape(8, -1).T - 1
        positive = np.vstack([0.5 + a * signs for a in (0.15, 0.35)])
        x = np.stack((1 - positive, positive), axis=-1)
        return x, np.full(len(x), 1 / len(x)), self.bayes_oracle(x, {})[:, 1]

    def sample(self, n, rng):
        x, _, q = self.support()
        count = np.full(4, n // 4)
        count[rng.permutation(4)[: n % 4]] += 1
        chosen, outcomes, groups = [], [], []
        for stratum, size in enumerate(count):
            group, label = divmod(stratum, 2)
            idx = np.arange(group * 256, (group + 1) * 256)
            likelihood = q[idx] if label else 1 - q[idx]
            chosen.append(rng.choice(idx, size=size, p=likelihood / likelihood.sum()))
            outcomes.append(np.full(size, label, dtype=int))
            groups.append(np.full(size, bool(group), dtype=bool))
        order = rng.permutation(n)
        return (
            x[np.concatenate(chosen)[order]],
            np.concatenate(outcomes)[order],
            {"high_amplitude": np.concatenate(groups)[order]},
        )

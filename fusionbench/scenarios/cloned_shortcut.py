"""Two stable votes and a shortcut whose reports can be cloned and reversed."""

from numbers import Real

import numpy as np

from fusionbench.scenarios.base import Scenario


def _probability(value, name):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
        or not 0 <= value <= 1
    ):
        raise ValueError(f"{name} must be a probability in [0, 1]")
    return float(value)


class ClonedShortcut(Scenario):
    """Control exact shortcut copies and shared sign reversal independently by row."""

    n_classes = 2
    primary_metric = "brier"
    evaluation_group_fields = ("shortcut_cloned", "shortcut_reversed")
    oracle_description = (
        "Posterior given the eight votes and the split's known law, marginalizing "
        "over unobserved cloning and reversal events"
    )
    stable_accuracy = 0.75
    shortcut_accuracy = 0.80
    n_stable = 2
    n_shortcut = 6

    def __init__(self, clone_probability=1.0, test_reversal_probability=1.0, **kwargs):
        self.clone_probability = _probability(clone_probability, "clone_probability")
        self.test_reversal_probability = _probability(
            test_reversal_probability, "test_reversal_probability"
        )
        super().__init__(
            clone_probability=self.clone_probability,
            test_reversal_probability=self.test_reversal_probability,
            **kwargs,
        )

    def model_ids(self):
        return [f"stable_{i}" for i in range(self.n_stable)] + [
            f"shortcut_{i}" for i in range(self.n_shortcut)
        ]

    def sample(self, n, rng):
        return self.sample_split(n, rng, "fit")

    def sample_split(self, n, rng, split):
        if split not in ("fit", "test"):
            raise ValueError("split must be 'fit' or 'test'")
        # Draw the same primitives even at 0 or 1 to pair every intervention.
        y = rng.integers(2, size=n)
        stable = y[:, None] ^ (rng.random((n, self.n_stable)) >= self.stable_accuracy)
        shortcut = y[:, None] ^ (
            rng.random((n, self.n_shortcut)) >= self.shortcut_accuracy
        )
        cloned = rng.random(n) < self.clone_probability
        reversal = self.test_reversal_probability if split == "test" else 0.0
        reversed_rows = rng.random(n) < reversal
        shortcut[cloned] = shortcut[cloned, :1]
        shortcut[reversed_rows] ^= 1
        p = np.column_stack(
            (
                np.where(stable, self.stable_accuracy, 1 - self.stable_accuracy),
                np.where(shortcut, self.shortcut_accuracy, 1 - self.shortcut_accuracy),
            )
        )
        latent = {
            "split": split,
            "shortcut_cloned": cloned,
            "shortcut_reversed": reversed_rows,
        }
        return np.stack((1 - p, p), axis=-1), y, latent

    def bayes_oracle(self, preds, latent):
        """Integrate both hidden shared events for the observed ordered vote vector."""
        votes = np.asarray(preds)[:, :, 1] > 0.5
        counts = votes[:, : self.n_stable].sum(axis=1)
        positives = votes[:, self.n_stable :].sum(axis=1)
        stable, shortcut = self.stable_accuracy, self.shortcut_accuracy
        clone = self.clone_probability
        reversal = self.test_reversal_probability if latent["split"] == "test" else 0.0
        # Columns give the number of matching votes for label 0 and label 1.
        matches = np.column_stack((self.n_stable - counts, counts))
        hits = np.column_stack((self.n_shortcut - positives, positives))
        stable_likelihood = stable**matches * (1 - stable) ** (self.n_stable - matches)
        independent = shortcut**hits * (1 - shortcut) ** (self.n_shortcut - hits)
        copied = shortcut * (hits == self.n_shortcut) + (1 - shortcut) * (hits == 0)
        aligned = (1 - clone) * independent + clone * copied
        # Under reversal the two classes exchange their shortcut likelihoods.
        mass = stable_likelihood * (
            (1 - reversal) * aligned + reversal * aligned[:, ::-1]
        )
        total = mass.sum(axis=1, keepdims=True)
        if np.any(total == 0):
            raise ValueError(
                "Observed votes have zero probability under this clone law"
            )
        return mass / total

"""Stable and environment-dependent binary provider signals."""

import numpy as np

from fusionbench.scenarios.base import Scenario


def _mixture(value):
    mix = np.asarray(value, dtype=float)
    if mix.shape != (3,) or not np.all((mix >= 0) & (mix <= 1)) or not np.isclose(mix.sum(), 1, atol=1e-10, rtol=0):
        raise ValueError("Environment proportions must be three probabilities summing to one")
    return tuple(float(v) for v in mix / mix.sum())


class EnvironmentShortcut(Scenario):
    """Change the mixture of aligned, independent and reversed shortcut environments."""

    primary_metric = "brier"
    evaluation_group_fields = ("environment",)
    oracle_description = "Posterior conditional on the two reported votes, marginalized over the split's environment mixture"

    def __init__(self, fit_mix=(0.8, 0.1, 0.1), test_mix=(0.1, 0.1, 0.8),
                 stable_accuracy=0.75, shortcut_accuracy=0.90, stable_only=False, **kw):
        self.fit_mix, self.test_mix = _mixture(fit_mix), _mixture(test_mix)
        if not 0.5 <= stable_accuracy <= 1 or not 0.5 <= shortcut_accuracy <= 1:
            raise ValueError("Stable and aligned shortcut accuracies must be in [0.5, 1]")
        if not isinstance(stable_only, (bool, np.bool_)):
            raise TypeError("stable_only must be boolean")
        self.stable_accuracy, self.shortcut_accuracy, self.stable_only = float(stable_accuracy), float(shortcut_accuracy), bool(stable_only)
        super().__init__(fit_mix=self.fit_mix, test_mix=self.test_mix, stable_accuracy=self.stable_accuracy,
                         shortcut_accuracy=self.shortcut_accuracy, stable_only=self.stable_only, **kw)

    @property
    def n_classes(self):
        return 2

    def model_ids(self):
        return ["stable", "shortcut"]

    def sample(self, n, rng):
        return self.sample_split(n, rng, "fit")

    def sample_split(self, n, rng, split):
        if split not in ("fit", "test"):
            raise ValueError("split must be 'fit' or 'test'")
        mix = self.fit_mix if split == "fit" else self.test_mix
        y = rng.integers(2, size=n)
        env = np.searchsorted(np.cumsum(mix)[:2], rng.random(n), side="right")
        accuracy = np.array([self.shortcut_accuracy, 0.5, 1 - self.shortcut_accuracy])
        stable = np.where(rng.random(n) < self.stable_accuracy, y, 1 - y)
        shortcut = np.where(rng.random(n) < accuracy[env], y, 1 - y)
        scores = np.column_stack((np.where(stable, self.stable_accuracy, 1 - self.stable_accuracy),
                                  np.where(shortcut, self.shortcut_accuracy, 1 - self.shortcut_accuracy)))
        if self.stable_only:
            scores[:, 1] = 0.5
        return np.stack((1 - scores, scores), axis=-1), y, {"environment": env, "split": split}

    def bayes_oracle(self, preds, latent):
        # At equal prior the stable score is its own posterior. The split's
        # marginalized shortcut accuracy supplies an independent likelihood.
        stable = preds[:, 0, :]
        if self.stable_only:
            return stable.copy()
        mix = self.test_mix if latent.get("split", "fit") == "test" else self.fit_mix
        q = mix[0] * self.shortcut_accuracy + mix[1] * 0.5 + mix[2] * (1 - self.shortcut_accuracy)
        votes = preds[:, 1, :].argmax(axis=1)
        mass = stable * np.where(votes[:, None] == np.arange(2), q, 1 - q)
        total = mass.sum(axis=1, keepdims=True)
        if np.any(total <= 0):
            raise ValueError("Observed votes have zero probability under this environment mixture")
        return mass / total

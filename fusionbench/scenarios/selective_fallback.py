"""A specialist can lose resolution while a reference provider stays available."""

import numpy as np

from fusionbench.scenarios.base import Scenario


class SelectiveFallback(Scenario):
    """Censor a fixed quota of reports, randomly or in order of latent difficulty."""

    primary_metric = "brier"
    evaluation_group_fields = ("fallback", "hard")
    oracle_description = "Row posterior marginalized over hidden difficulty and availability, conditional on all reported scores"
    _reference_accuracy = np.array([0.82, 0.60])
    _fallible_accuracy = np.array([0.94, 0.72])

    def __init__(
        self,
        fallback_rate=0.25,
        mechanism="difficulty",
        expose_flag=False,
        hard_fraction=0.30,
        neutral_rate=0.20,
        **kw,
    ):
        if any(not 0 <= s <= 1 for s in (fallback_rate, hard_fraction, neutral_rate)):
            raise ValueError(
                "fallback_rate, hard_fraction and neutral_rate must be probabilities"
            )
        if mechanism not in ("difficulty", "random"):
            raise ValueError("mechanism must be 'difficulty' or 'random'")
        if not isinstance(expose_flag, (bool, np.bool_)):
            raise TypeError("expose_flag must be boolean")
        self.fallback_rate, self.mechanism, self.expose_flag = (
            float(fallback_rate),
            mechanism,
            bool(expose_flag),
        )
        self.hard_fraction, self.neutral_rate = (
            float(hard_fraction),
            float(neutral_rate),
        )
        super().__init__(
            fallback_rate=self.fallback_rate,
            mechanism=mechanism,
            expose_flag=self.expose_flag,
            hard_fraction=self.hard_fraction,
            neutral_rate=self.neutral_rate,
            **kw,
        )

    @property
    def n_classes(self):
        return 2

    def model_ids(self):
        return ["reference", "fallible"] + (["available"] if self.expose_flag else [])

    def _quotas(self, n):
        return int(np.floor(n * self.hard_fraction + 0.5)), int(
            np.floor(n * self.fallback_rate + 0.5)
        )

    def sample(self, n, rng):
        h, f = self._quotas(n)
        y = rng.integers(2, size=n)
        difficulty_order = rng.permutation(n)
        random_order = rng.permutation(n)
        hard, fallback = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
        hard[difficulty_order[:h]] = True
        fallback[
            (difficulty_order if self.mechanism == "difficulty" else random_order)[:f]
        ] = True
        ref = np.where(
            rng.random(n) < self._reference_accuracy[hard.astype(int)], y, 1 - y
        )
        expert = np.where(
            rng.random(n) < self._fallible_accuracy[hard.astype(int)], y, 1 - y
        )
        neutral = rng.random(n) < self.neutral_rate
        p = np.column_stack(
            (
                np.where(ref, 0.75, 0.25),
                np.where(fallback | neutral, 0.5, np.where(expert, 0.85, 0.15)),
            )
        )
        if self.expose_flag:
            p = np.column_stack((p, ~fallback))
        return (
            np.stack((1 - p, p), axis=-1),
            y,
            {"fallback": fallback, "hard": hard, "bank_size": n},
        )

    def bayes_oracle(self, preds, latent):
        # A random permutation gives each row an equal chance at each rank.
        # Only the known quota design enters here, never the realized masks.
        n = latent.get("bank_size", len(preds))
        h, f = self._quotas(n)
        if self.mechanism == "difficulty":
            both = min(h, f)
            joint = np.array([[n - max(h, f), f - both], [h - both, both]]) / n
        else:
            joint = np.outer([n - h, h], [n - f, f]) / n**2
        scores = preds[:, :, 1]
        same_ref = (scores[:, 0, None] > 0.5) == np.arange(2)
        same_expert = (scores[:, 1, None] > 0.5) == np.arange(2)
        neutral = scores[:, 1, None] == 0.5
        mass = np.zeros((len(preds), 2))
        for hard, (active, censored) in enumerate(joint):
            ref = self._reference_accuracy[hard]
            expert = self._fallible_accuracy[hard]
            available_likelihood = np.where(
                neutral,
                self.neutral_rate,
                (1 - self.neutral_rate) * np.where(same_expert, expert, 1 - expert),
            )
            if self.expose_flag:
                observed_active = scores[:, 2, None] == 1
                active, censored = active * observed_active, censored * ~observed_active
            mass += np.where(same_ref, ref, 1 - ref) * (
                active * available_likelihood + censored * neutral
            )
        total = mass.sum(axis=1, keepdims=True)
        if np.any(total <= 0):
            raise ValueError(
                "Observed report has zero probability under this fallback mechanism"
            )
        return mass / total

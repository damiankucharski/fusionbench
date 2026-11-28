"""Binary expertise restricted to a rare region with a noisy observable cue."""

from __future__ import annotations

from numbers import Real

import numpy as np

from fusionbench.scenarios.base import Scenario
from fusionbench.scenarios.calibrated import to_binary_proba


class RareSpecialist(Scenario):
    """A balanced label, a reliable generalist vote and a conditionally useful vote.

    The third column encodes a noisy measurement of region membership. Its
    meaning is fixed across examples. The private region is for slice metrics;
    the oracle marginalizes it out given the observed cue.
    """

    n_classes = 2
    primary_metric = "brier"
    evaluation_group_fields = ("region",)
    oracle_description = "Binary vote likelihoods, marginalizing the private region conditional on the observed routing cue"

    def __init__(
        self,
        region_prevalence: float = 0.1,
        specialist_gain: float = 0.4,
        cue_accuracy: float = 0.85,
        generalist_accuracy: float = 0.75,
        name: str | None = None,
    ):
        values = {
            "region_prevalence": region_prevalence,
            "specialist_gain": specialist_gain,
            "cue_accuracy": cue_accuracy,
            "generalist_accuracy": generalist_accuracy,
        }
        limits = {
            "region_prevalence": (0, 1),
            "specialist_gain": (0, 0.5),
            "cue_accuracy": (0.5, 1),
            "generalist_accuracy": (0.5, 1),
        }
        for key, value in values.items():
            low, high = limits[key]
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, Real)
                or not low <= value <= high
            ):
                raise ValueError(f"{key} must lie in [{low}, {high}]")
        values = {key: float(value) for key, value in values.items()}
        super().__init__(name=name, **values)
        self.region_prevalence = values["region_prevalence"]
        self.specialist_gain = values["specialist_gain"]
        self.cue_accuracy = values["cue_accuracy"]
        self.generalist_accuracy = values["generalist_accuracy"]

    def model_ids(self):
        return ["generalist", "specialist", "routing_cue"]

    def sample(self, n, rng):
        y = rng.integers(0, 2, size=n)
        region = rng.random(n) < self.region_prevalence
        generalist = np.where(rng.random(n) < self.generalist_accuracy, y, 1 - y)
        specialist = np.where(
            rng.random(n) < 0.5 + region * self.specialist_gain, y, 1 - y
        )
        cue = np.where(rng.random(n) < self.cue_accuracy, region, ~region)
        scores = np.column_stack(
            (
                np.where(
                    generalist, self.generalist_accuracy, 1 - self.generalist_accuracy
                ),
                np.where(specialist, 0.75, 0.25),
                np.where(cue, 0.75, 0.25),
            )
        )
        return np.stack((1 - scores, scores), axis=-1), y, {"region": region}

    def bayes_oracle(self, preds, latent):
        generalist, specialist, cue = (np.asarray(preds)[:, :, 1] > 0.5).T
        cue_if_region = np.where(cue, self.cue_accuracy, 1 - self.cue_accuracy)
        mass_in = self.region_prevalence * cue_if_region
        mass_out = (1 - self.region_prevalence) * (1 - cue_if_region)
        # P(cue, specialist vote | Y). Outside the region the specialist flips a fair coin.
        specialty_given_one = np.where(
            specialist, 0.5 + self.specialist_gain, 0.5 - self.specialist_gain
        )
        specialist_and_cue_one = mass_in * specialty_given_one + 0.5 * mass_out
        specialist_and_cue_zero = mass_in * (1 - specialty_given_one) + 0.5 * mass_out
        generalist_given_one = np.where(
            generalist, self.generalist_accuracy, 1 - self.generalist_accuracy
        )
        one = generalist_given_one * specialist_and_cue_one
        zero = (1 - generalist_given_one) * specialist_and_cue_zero
        if np.any(one + zero == 0):
            raise ValueError(
                "Observed votes have zero probability under the configured scenario"
            )
        return to_binary_proba(one / (one + zero))

"""Native seeded trees, with fit-only objectives for rounds two through four."""

import numpy as np
from engine import BASE_OPTIONS, fit_model
from okapi.paired_fitness import PairedGainFitness, arithmetic_mean_tree
from okapi.prediction_pull import PredictionPullBrier

from fusionbench.methods.okapi import OKAPIMethod, _brier_objective


def _row_brier(prediction, labels):
    p = np.asarray(prediction)
    y = np.asarray(labels).reshape(-1).astype(int)
    if p.shape[1] == 1:
        return 2 * np.square(p[:, 0] - y)
    return np.square(p - np.eye(p.shape[1])[y]).sum(1)


class SeededNativeGP(OKAPIMethod):
    """Fit ordinary OKAPI with one additional initial slot and a fixed objective."""

    def __init__(self, policy, seed=0, groups=None):
        if policy not in {"sham", "seeded_mean", "pull_mean", "pull_zero", "paired_se", "paired_zero"}:
            raise ValueError(f"Unknown native candidate {policy!r}")
        super().__init__(**BASE_OPTIONS, seed=seed)
        self.policy = policy
        self.groups = groups

    def _seed_population(self, okapi, rng):
        super()._seed_population(okapi, rng)
        labels = okapi.gt_tensor
        seed = arithmetic_mean_tree(okapi.train_tensors)
        if self.policy == "sham":
            seed = okapi.population[0].copy()
        if not okapi._within_caps(seed):
            raise ValueError("The additional starting tree exceeds the study's fixed structural budget")
        self.seed_tree_ = seed.copy()
        if self.policy.startswith("pull"):
            strength = np.sqrt(8 / len(labels))
            if self.policy == "pull_mean":
                objective = PredictionPullBrier(seed.predict(), labels, strength=strength)
            else:
                # Keep the scale (hence rounding resolution) of the penalized arm.
                def objective(p, y):
                    return 1 - (1 - _brier_objective(p, y)) / (1 + strength)

            okapi.objective_functions = (objective,)
        elif self.policy.startswith("paired"):
            z = 1.645 if self.policy == "paired_se" else 0.
            objective = PairedGainFitness(seed.predict(), labels, _row_brier, z=z, groups=self.groups)
            okapi.paired_fitness_ = objective
            okapi.objective_functions = (objective,)

        # Native parent selection, survivor selection and terminal Pareto choice all
        # use the same rounded score. No checkpoint overrides a lost mean candidate.
        okapi.population.append(seed)
        okapi.population_size += 1
        okapi.fitnesses = None
        self.initial_population_size_ = len(okapi.population)


def fit_candidate(name, x, y, seed=0, groups=None):
    """Train on every supplied label. Only the selected native tree is deployed."""
    return fit_model(SeededNativeGP(name, seed=seed, groups=groups), x, y)

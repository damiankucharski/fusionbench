"""Choose a single fitted native tree using perturbed source-validation views."""

import numpy as np

from fusionbench.methods.base import Method
from fusionbench.methods.native_pattern import NativePatternOKAPI, _brier_skill


def validation_probes(x, seed):
    """Build two paired input perturbations using only the source validation rows."""
    rows, providers, _ = x.shape
    rng = np.random.default_rng(np.random.SeedSequence([seed, 841]))
    order = rng.permutation(rows)
    donor = np.empty_like(order)
    donor[order] = np.roll(order, 1)
    faulty = rng.integers(providers, size=rows)
    affected = np.flatnonzero(rng.random(rows) < 0.5)
    swapped = x.copy()
    swapped[affected, faulty[affected]] = x[donor[affected], faulty[affected]]
    exponents = rng.choice([0.5, 2.0], size=(1, providers, 1))
    warped = x**exponents
    return swapped, warped / warped.sum(axis=2, keepdims=True)


class TransferPatternOKAPI(Method):
    """Select one native program by worst validation fitness within a clean allowance.

    clean_tolerance uses the units of the supplied higher-is-better objective.
    With default fitness, 0.01 allows 0.02 extra full-class Brier loss.
    """

    name = "okapi_transfer_pattern"

    def __init__(self, seed=0, fitness=None, clean_tolerance=0.01):
        self.seed = seed
        self.fitness = fitness
        self.clean_tolerance = clean_tolerance

    @classmethod
    def available(cls):
        try:
            from okapi.transfer_selection import shrink_subtree_weights  # noqa: F401
        except ImportError:
            return False
        return NativePatternOKAPI.available()

    def fit(self, fit_preds, fit_y, *, task, n_classes):
        from okapi.transfer_selection import (
            select_transfer_candidate,
            shrink_subtree_weights,
        )
        from fusionbench.eval.suite_point_gp import _full, predict_tree

        base = NativePatternOKAPI(seed=self.seed, fitness=self.fitness)
        base.fit(fit_preds, fit_y, task=task, n_classes=n_classes)
        indices = base.validation_indices_
        x = _full(fit_preds)[indices]
        y = np.asarray(fit_y).ravel()[indices]
        discovery = dict(base.discovery_trees_)
        discovery["half"] = shrink_subtree_weights(
            discovery["augmented"], base.n_providers_
        )
        objective = _brier_skill if self.fitness is None else self.fitness
        clean_scores = dict(base.counts_["validation_scores"])
        clean_scores["half"] = objective(predict_tree(discovery["half"], x), y)
        probes = validation_probes(x, self.seed)
        scores = {
            key: (
                clean_scores[key],
                *(objective(predict_tree(tree, data), y) for data in probes),
            )
            for key, tree in discovery.items()
        }
        chosen = select_transfer_candidate(scores, clean_tolerance=self.clean_tolerance)
        refitted = dict(base.trees_)
        refitted["half"] = shrink_subtree_weights(
            refitted["augmented"], base.n_providers_
        )

        super().fit(fit_preds, fit_y, task=task, n_classes=n_classes)
        self.base_ = base
        self.chosen_ = chosen
        self.clean_scores_ = clean_scores
        self.scores_ = scores
        self.discovery_trees_ = discovery
        self.trees_ = refitted
        additional = 1 + len(probes) * len(discovery)
        self.counts_ = {
            "validation_rows": len(y),
            "additional_score_calls": additional,
            "base_score_calls": base.counts_["total_score_calls"],
            "total_score_calls": base.counts_["total_score_calls"] + additional,
        }
        return self

    def predict(self, preds):
        from fusionbench.eval.suite_point_gp import predict_tree

        return predict_tree(self.trees_[self.chosen_], self.base_._query(preds))

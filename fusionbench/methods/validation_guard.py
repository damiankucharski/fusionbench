"""An outer validation comparison of unchanged native fusion trees against averaging."""

import numpy as np

from fusionbench.methods.base import Method
from fusionbench.methods.native_pattern import RegularizedNativePatternOKAPI
from fusionbench.methods.search_guard import row_brier


class GuardedNativeOKAPI(Method):
    """Fit the incumbent on inner rows and guard deployment on separate outer rows."""

    name = "okapi_terminal_guard"

    def __init__(self, seed=0, *, split_seed=None, gate_fraction=0.3):
        if not 0 < gate_fraction < 1:
            raise ValueError("gate_fraction must be between zero and one")
        self.seed = seed
        self.split_seed = seed if split_seed is None else split_seed
        self.gate_fraction = gate_fraction

    @classmethod
    def available(cls):
        if not RegularizedNativePatternOKAPI.available():
            return False
        try:
            from okapi.mean_guard import select_against_mean  # noqa: F401
        except ImportError:
            return False
        return True

    def fit(self, fit_preds, fit_y, *, task, n_classes):
        from okapi.mean_guard import select_against_mean
        from okapi.paired_fitness import arithmetic_mean_tree

        from fusionbench.eval.suite_point_gp import _bind, _full

        x, y = _full(fit_preds), np.asarray(fit_y)
        n = len(y)
        n_outer = int(np.ceil(n * self.gate_fraction))
        if len(x) != n or n - n_outer < 2:
            raise ValueError(
                "Supply aligned rows with at least two inner-fit observations"
            )
        order = np.random.default_rng([self.split_seed, 159]).permutation(n)
        self.outer_indices_, self.inner_indices_ = np.split(order, [n_outer])
        inner, outer = self.inner_indices_, self.outer_indices_
        self.estimator_ = RegularizedNativePatternOKAPI(
            seed=self.seed, split_seed=self.split_seed
        ).fit(x[inner], y[inner], task=task, n_classes=n_classes)
        self.trees_ = {
            "mean": arithmetic_mean_tree(_bind(x[inner])),
            **self.estimator_.trees_,
        }
        predictions = self.candidate_predictions(x[outer])
        names = list(self.estimator_.trees_)
        reference = row_brier(predictions["mean"], y[outer])
        losses = [row_brier(predictions[k], y[outer]) for k in names]
        self.guard_decision_ = select_against_mean(
            reference, np.stack(losses), alpha=0.05, draws=4095, seed=0, min_units=8
        )
        # Use the identical paired gain and tie policy in the unguarded comparator.
        gains = self.guard_decision_.gains
        nominee = self.guard_decision_.best_index
        self.raw_choice_ = (
            names[nominee] if nominee is not None and gains[nominee] > 0 else "mean"
        )
        winner = self.guard_decision_.index
        self.chosen_ = "mean" if winner is None else names[winner]
        self.tree_ = self.trees_[self.chosen_]
        super().fit(fit_preds, fit_y, task=task, n_classes=n_classes)
        return self

    def candidate_predictions(self, preds):
        from fusionbench.eval.suite_point_gp import predict_tree

        x = self.estimator_._query(preds)
        return {name: predict_tree(tree, x) for name, tree in self.trees_.items()}

    def predict(self, preds):
        from fusionbench.eval.suite_point_gp import predict_tree

        return predict_tree(self.tree_, self.estimator_._query(preds))

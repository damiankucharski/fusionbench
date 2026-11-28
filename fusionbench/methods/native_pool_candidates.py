"""Offer two native arithmetic pools on the regularized pattern search's held-out rows."""

from time import perf_counter

from fusionbench.methods.averaging import TopN
from fusionbench.methods.native_pattern import (
    RegularizedNativePatternOKAPI,
    _brier_skill,
)


def _pool_tree(inputs, indices):
    from okapi.paired_fitness import arithmetic_mean_tree

    names = (f"m{i}.npy" for i in indices)
    return arithmetic_mean_tree({name: inputs[name] for name in names})


def _top_n(x, y, task):
    return (
        TopN(metric="brier")
        .fit(x, y, task=task, n_classes=x.shape[-1])
        .selected_indices_
    )


class PoolCandidateNativePatternOKAPI(RegularizedNativePatternOKAPI):
    """Select a native arithmetic mean or ranked-prefix mean alongside existing trees.

    Reuses the parent's discovery programs, raw held-out fitness and tie ordering.
    Top-N ranks providers by discovery Brier and searches their equal-weight
    prefixes. Only after it wins does it repeat the ranking and prefix search on
    all source rows. The incumbent retains its original deployment/refit semantics.
    The historical row split can place different rows from a scene in both subsets.
    """

    name = "okapi_native_pattern_l2_pools"

    def __init__(self, seed=0, *, split_seed=0, gp_validation_top_k=8):
        super().__init__(
            seed=seed, split_seed=split_seed, gp_validation_top_k=gp_validation_top_k
        )

    @classmethod
    def available(cls):
        try:
            from okapi.paired_fitness import arithmetic_mean_tree  # noqa: F401
        except ImportError:
            return False
        return super().available()

    def fit(self, fit_preds, fit_y, *, task, n_classes, on_gp_candidate=None):
        from fusionbench.eval.suite_point_gp import _bind, _targets, predict_tree

        super().fit(
            fit_preds,
            fit_y,
            task=task,
            n_classes=n_classes,
            on_gp_candidate=on_gp_candidate,
        )
        self.old_chosen_ = self.chosen_
        x = self._query(fit_preds)
        y, _ = _targets(fit_y, len(x), n_classes)
        discovery, held = self.discovery_indices_, self.validation_indices_
        start = perf_counter()
        d, dy = x[discovery], y[discovery]
        m = x.shape[1]
        indices = _top_n(d, dy, task)
        inputs = _bind(d)
        pools = {
            "mean": _pool_tree(inputs, range(m)),
            "top_n": _pool_tree(inputs, indices),
        }
        elapsed = self.counts_["seconds"]
        elapsed["pool_discovery"] = perf_counter() - start
        start = perf_counter()
        v, vy = x[held], y[held]
        scores = self.counts_["validation_scores"]
        scores.update(
            {
                name: _brier_skill(predict_tree(tree, v), vy)
                for name, tree in pools.items()
            }
        )
        elapsed["pool_validation"] = perf_counter() - start
        # Insertion order preserves historical precedence on an exact skill tie.
        self.chosen_ = max(scores, key=scores.get)
        self.action_ = self.chosen_ if self.chosen_ in pools else "native"
        self.discovery_trees_.update(pools)
        start = perf_counter()
        refit_indices = None
        if self.chosen_ == "top_n":
            refit_indices = _top_n(x, y, task)
            self.trees_["top_n"] = _pool_tree(_bind(x), refit_indices)
        elif self.chosen_ == "mean":
            self.trees_["mean"] = _pool_tree(_bind(x), range(m))
        self.best_tree = self.trees_[self.chosen_]
        elapsed["pool_refit"] = perf_counter() - start
        calls = self.counts_["score_calls"]
        calls["family_validation"] += len(pools)
        # One Brier per individual provider and one per candidate prefix.
        calls["top_n_discovery"] = 2 * m
        calls["top_n_refit"] = 2 * m if refit_indices is not None else 0
        self.counts_["total_score_calls"] = sum(calls.values())
        self.counts_["pool_candidates"] = {
            "incumbent": self.old_chosen_,
            "action": self.action_,
            "discovery_top_n_indices": indices.tolist(),
            "refit_top_n_indices": None
            if refit_indices is None
            else refit_indices.tolist(),
            "extra_score_calls": 2 * m + len(pools) + calls["top_n_refit"],
        }
        return self

    def predict_incumbent(self, preds):
        """Replay the original winner without rerunning the matched evolution."""
        from fusionbench.eval.suite_point_gp import predict_tree

        return predict_tree(self.trees_[self.old_chosen_], self._query(preds))

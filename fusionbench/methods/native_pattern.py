"""Select and deploy one OKAPI tree using a scalar predictive fitness."""

from time import perf_counter

import numpy as np

from fusionbench.eval.metrics import normalize_proba
from fusionbench.methods.base import Method
from fusionbench.methods.okapi import _task_postprocessing


def _brier_skill(p, labels):
    return 1.0 - 0.5 * np.mean(np.sum((p - np.eye(p.shape[1])[labels]) ** 2, axis=1))


def _full_probability(p):
    if p.shape[1] == 1:
        positive = np.clip(p[:, 0], 0.0, 1.0)
        return np.column_stack((1.0 - positive, positive))
    return normalize_proba(p)


def _head(inputs, labels, bases, objective, multiclass):
    """Bind independent copies of the frozen feature programs to this fit split."""
    from okapi.node import ValueNode
    from okapi.pattern_logit import fit_pattern_tree

    roots = [ValueNode(None, tensor, name) for name, tensor in inputs.items()]
    for tree in bases:
        rebound = tree.copy()
        for node in rebound.value_nodes:
            node.value = inputs[node.id]
        roots.append(rebound.root)
    return fit_pattern_tree(roots, labels, objective=objective, multiclass=multiclass)


class NativePatternOKAPI(Method):
    """Discover complete GP bases, fit native heads and select one tree on held-out rows.

    fitness(probabilities, labels) receives canonical full-class probabilities
    and returns a finite higher-is-better scalar. The default, one minus half
    the full-class Brier score, uses OKAPI's historical unit-interval fitness.
    """

    name = "okapi_native_pattern"

    def __init__(self, seed=0, fitness=None):
        if fitness is not None and not callable(fitness):
            raise TypeError("fitness must be callable")
        self.seed = seed
        self.fitness = fitness

    @classmethod
    def available(cls):
        try:
            from okapi.pattern_features import select_pattern_features  # noqa: F401
            from okapi.pattern_logit import fit_pattern_tree  # noqa: F401
        except ImportError:
            return False
        return True

    def fit(self, fit_preds, fit_y, *, task, n_classes):
        from okapi.pattern_features import select_pattern_features

        from fusionbench.eval.full_suite_point import fingerprint
        from fusionbench.eval.seed_geometry_standard import fit_seed_geometry
        from fusionbench.eval.suite_point_gp import _bind, _full, _targets, predict_tree

        x = _full(fit_preds)
        n, providers, classes = x.shape
        expected_task = "binary" if classes == 2 else "multiclass"
        if n < 2 or providers < 2 or classes != n_classes or task != expected_task:
            raise ValueError(
                "Supply aligned fit rows, at least two providers and a matching classification task"
            )
        if not np.all((x >= 0.0) & (x <= 1.0)) or not np.allclose(
            x.sum(2), 1.0, rtol=0, atol=1e-6
        ):
            raise ValueError(
                "Each provider must supply finite class probabilities summing to one"
            )
        labels, _ = _targets(fit_y, n, classes)
        order = np.random.default_rng(
            np.random.SeedSequence([self.seed, 51])
        ).permutation(n)
        discovery, validation = np.split(order, [7 * n // 10])
        d, v = x[discovery], x[validation]
        dy, vy = labels[discovery], labels[validation]
        multiclass = classes > 2
        fitness = _brier_skill if self.fitness is None else self.fitness

        def full_score(p, y):
            value = np.asarray(fitness(p, y))
            if value.ndim != 0 or not np.isfinite(value):
                raise ValueError("fitness must return a finite scalar")
            return float(value)

        def native_score(p, y):
            return full_score(_full_probability(p), y)

        archive = []

        def collect(tree, prediction):
            archive.append((tree.copy(), prediction))

        start = perf_counter()
        gp = fit_seed_geometry(
            d,
            dy,
            v,
            vy,
            seed=self.seed,
            shape="mixed_branch",
            budgets=(600,),
            fitness=None if self.fitness is None else native_score,
            on_candidate=collect,
        )[600]
        gp_seconds = perf_counter() - start
        inputs = _bind(d)
        start = perf_counter()
        flat_d, flat_d_info = _head(inputs, dy, (), native_score, multiclass)
        flat_seconds = perf_counter() - start
        start = perf_counter()
        with _task_postprocessing(task):
            flat_prediction = flat_d.predict()
        bases, screen = select_pattern_features(
            archive,
            inputs.values(),
            flat_prediction,
            dy,
            objective=native_score,
        )
        screen_seconds = perf_counter() - start
        start = perf_counter()
        aug_d, aug_d_info = _head(inputs, dy, bases, native_score, multiclass)
        aug_seconds = perf_counter() - start
        discovery_trees = {"flat": flat_d, "augmented": aug_d, "gp": gp.val_tree}
        validation_scores = {
            name: full_score(predict_tree(tree, v), vy)
            for name, tree in discovery_trees.items()
        }
        chosen = max(validation_scores, key=validation_scores.get)

        inputs = _bind(x)
        start = perf_counter()
        flat, flat_info = _head(inputs, labels, (), native_score, multiclass)
        flat_refit_seconds = perf_counter() - start
        start = perf_counter()
        augmented, aug_info = _head(inputs, labels, bases, native_score, multiclass)
        aug_refit_seconds = perf_counter() - start
        score_calls = {
            "gp_discovery": gp.calls,
            "gp_validation": gp.n_trees,
            "flat_discovery": flat_d_info["evaluations"],
            "feature_screen": screen["evaluations"],
            "augmented_discovery": aug_d_info["evaluations"],
            "family_validation": len(validation_scores),
            "flat_refit": flat_info["evaluations"],
            "augmented_refit": aug_info["evaluations"],
        }
        super().fit(fit_preds, fit_y, task=task, n_classes=n_classes)
        self.n_providers_ = providers
        self.chosen_ = chosen
        self.discovery_trees_ = discovery_trees
        self.trees_ = {"flat": flat, "augmented": augmented, "gp": gp.val_tree}
        self.counts_ = {
            "discovery_rows": len(d),
            "validation_rows": len(v),
            "refit_rows": n,
            "discovery_sha256": fingerprint(discovery),
            "validation_sha256": fingerprint(validation),
            "fitness": "one_minus_half_full_brier"
            if self.fitness is None
            else "custom",
            "gp_selection_scale": "unit_brier_3_decimals"
            if self.fitness is None
            else "population_ranks",
            "gp_path_digest": gp.path_digest,
            "gp_unique_programs": gp.n_trees,
            "screen": screen,
            "validation_scores": validation_scores,
            "score_calls": score_calls,
            "total_score_calls": sum(score_calls.values()),
            "seconds": {
                "gp": gp_seconds,
                "flat_discovery": flat_seconds,
                "feature_screen": screen_seconds,
                "augmented_discovery": aug_seconds,
                "flat_refit": flat_refit_seconds,
                "augmented_refit": aug_refit_seconds,
            },
        }
        return self

    def _query(self, preds):
        from fusionbench.eval.suite_point_gp import _full

        x = _full(preds)
        if x.shape[1:] != (self.n_providers_, self.n_classes):
            raise ValueError("Query provider and class counts must match training")
        return x

    def component_predictions(self, preds):
        from fusionbench.eval.suite_point_gp import predict_tree

        x = self._query(preds)
        result = {name: predict_tree(tree, x) for name, tree in self.trees_.items()}
        result["native"] = result[self.chosen_]
        return result

    def predict(self, preds):
        from fusionbench.eval.suite_point_gp import predict_tree

        return predict_tree(self.trees_[self.chosen_], self._query(preds))

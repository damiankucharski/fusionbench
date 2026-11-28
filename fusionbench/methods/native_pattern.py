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


def _raw_roots(inputs):
    from okapi.node import ValueNode

    return [ValueNode(None, tensor, name) for name, tensor in inputs.items()]


def _validate_class_head(class_head, max_evaluations):
    if class_head not in (None, "diagonal", "full"):
        raise ValueError("class_head must be 'diagonal' or 'full'")
    if max_evaluations is not None:
        if class_head is None:
            raise ValueError("class_head_max_evaluations requires an explicit class_head")
        if isinstance(max_evaluations, (bool, np.bool_)) or not isinstance(max_evaluations, (int, np.integer)) or max_evaluations < 1:
            raise ValueError("Class head evaluation budget must be a positive integer")


def _head(inputs, labels, bases, objective, multiclass, l2_strength, class_head=None, max_evaluations=None):
    """Bind independent copies of the frozen feature programs to this fit split."""
    roots = _raw_roots(inputs)
    for tree in bases:
        rebound = tree.copy()
        for node in rebound.value_nodes:
            node.value = inputs[node.id]
        roots.append(rebound.root)
    options = {"l2_strength": l2_strength} if l2_strength else {}
    if class_head is not None:
        from okapi.multiclass_affine import fit_multiclass_affine_tree

        if max_evaluations is not None:
            options["max_evaluations"] = int(max_evaluations)
        return fit_multiclass_affine_tree(roots, labels, objective=objective, mode=class_head, **options)
    from okapi.pattern_logit import fit_pattern_tree

    return fit_pattern_tree(
        roots, labels, objective=objective, multiclass=multiclass, **options
    )


class NativePatternOKAPI(Method):
    """Discover complete GP bases, fit native heads and select one tree on held-out rows.

    fitness(probabilities, labels) receives canonical full-class probabilities
    and returns a finite higher-is-better scalar. The default, one minus half
    the full-class Brier score, uses OKAPI's historical unit-interval fitness.
    pair_basis appends native two-provider minima to the GP feature archive
    before the same discovery-only functional deduplication and screening.
    Set split_seed to hold the discovery/validation partition fixed while
    varying seed for the GP search. The partition uses seed by default.
    Positive l2_strength regularizes the native head weights on each fit
    split. Discovery screening and held-out selection use raw fitness.
    On multiclass tasks, class_head="diagonal" or "full" fits a native
    probability-affine head on each ordered set of providers and GP bases.
    class_head_max_evaluations limits objective calls per head fit.
    gp_validation_top_k limits the GP candidates eligible on validation to
    the k best programs on discovery, with ties kept in first-visit order.
    Default discovery ranking minimizes raw Brier. With a supplied fitness
    or discovery objective, ranking maximizes its unrounded discovery score.
    The complete GP archive remains available for screening head features.
    """

    name = "okapi_native_pattern"

    def __init__(
        self,
        seed=0,
        fitness=None,
        *,
        pair_basis=False,
        split_seed=None,
        l2_strength=0.0,
        class_head=None,
        class_head_max_evaluations=None,
        gp_validation_top_k=None,
    ):
        if fitness is not None and not callable(fitness):
            raise TypeError("fitness must be callable")
        if not isinstance(pair_basis, (bool, np.bool_)):
            raise TypeError("pair_basis must be boolean")
        strength = np.asarray(l2_strength)
        if (
            strength.ndim != 0
            or strength.dtype.kind not in "iuf"
            or not np.isfinite(strength)
            or strength < 0
        ):
            raise ValueError("l2_strength must be a finite nonnegative scalar")
        self.seed = seed
        self.fitness = fitness
        self.pair_basis = bool(pair_basis)
        self.split_seed = split_seed
        self.l2_strength = float(strength)
        _validate_class_head(class_head, class_head_max_evaluations)
        self.class_head = class_head
        self.class_head_max_evaluations = class_head_max_evaluations
        if gp_validation_top_k is not None:
            from fusionbench.eval.seed_geometry_standard import validate_gp_validation_top_k

            gp_validation_top_k = validate_gp_validation_top_k(gp_validation_top_k)
        self.gp_validation_top_k = gp_validation_top_k

    @classmethod
    def available(cls):
        try:
            from okapi.pattern_features import select_pattern_features  # noqa: F401
            from okapi.pattern_logit import fit_pattern_tree  # noqa: F401
        except ImportError:
            return False
        return True

    def _training_fitness(self, probabilities, labels):
        """Optional objective bound to this precise training split."""
        return None

    def fit(self, fit_preds, fit_y, *, task, n_classes, on_gp_candidate=None):
        """Fit the model and optionally observe each distinct GP discovery evaluation.

        on_gp_candidate(tree, prediction) receives the live program and its
        raw discovery output in first-visit order. Treat both as read-only
        and copy any state that must outlive the callback.
        """
        from okapi.pattern_features import select_pattern_features

        from fusionbench.eval.full_suite_point import fingerprint
        from fusionbench.eval.seed_geometry_standard import fit_seed_geometry
        from fusionbench.eval.suite_point_gp import _bind, _full, _targets, predict_tree

        if on_gp_candidate is not None and not callable(on_gp_candidate):
            raise TypeError("on_gp_candidate must be callable")
        x = _full(fit_preds)
        n, providers, classes = x.shape
        expected_task = "binary" if classes == 2 else "multiclass"
        if self.class_head is not None:
            if task != "multiclass" or classes < 3:
                raise ValueError("class_head requires a multiclass bank with at least three classes")
            # Check the optional native operator before starting the GP search.
            from okapi.multiclass_affine import fit_multiclass_affine_tree  # noqa: F401
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
        split_seed = self.seed if self.split_seed is None else self.split_seed
        order = np.random.default_rng(
            np.random.SeedSequence([split_seed, 51])
        ).permutation(n)
        discovery, validation = np.split(order, [7 * n // 10])
        self.validation_indices_ = validation.copy()
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

        discovery_objective = self._training_fitness(d, dy)
        refit_objective = self._training_fitness(x, labels)
        discovery_score = native_score if discovery_objective is None else discovery_objective
        refit_score = native_score if refit_objective is None else refit_objective

        def fit_head(inputs, fit_labels, bases, objective):
            return _head(inputs, fit_labels, bases, objective, multiclass, self.l2_strength,
                         self.class_head, self.class_head_max_evaluations)

        archive = []

        def collect(tree, prediction):
            archive.append((tree.copy(), prediction))
            if on_gp_candidate is not None:
                on_gp_candidate(tree, prediction)

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
            **({"discovery_fitness": discovery_objective} if discovery_objective is not None else {}),
            **({"gp_validation_top_k": self.gp_validation_top_k} if self.gp_validation_top_k is not None else {}),
        )[600]
        gp_seconds = perf_counter() - start
        inputs = _bind(d)
        gp_archive_size = len(archive)
        if self.pair_basis:
            from okapi.pair_basis import build_min_pair_basis

            start = perf_counter()
            pairs, pair_info = build_min_pair_basis(
                _raw_roots(inputs), dy, objective=discovery_score, multiclass=multiclass
            )
            archive.extend(pairs)
            pair_seconds = perf_counter() - start
        start = perf_counter()
        flat_d, flat_d_info = fit_head(inputs, dy, (), discovery_score)
        flat_seconds = perf_counter() - start
        start = perf_counter()
        with _task_postprocessing(task):
            flat_prediction = flat_d.predict()
        bases, screen = select_pattern_features(
            archive,
            inputs.values(),
            flat_prediction,
            dy,
            objective=discovery_score,
            **({"report_indices": True} if self.pair_basis else {}),
        )
        screen_seconds = perf_counter() - start
        start = perf_counter()
        aug_d, aug_d_info = fit_head(inputs, dy, bases, discovery_score)
        aug_seconds = perf_counter() - start
        discovery_trees = {"flat": flat_d, "augmented": aug_d, "gp": gp.val_tree}
        validation_scores = {
            name: full_score(predict_tree(tree, v), vy)
            for name, tree in discovery_trees.items()
        }
        chosen = max(validation_scores, key=validation_scores.get)

        inputs = _bind(x)
        start = perf_counter()
        flat, flat_info = fit_head(inputs, labels, (), refit_score)
        flat_refit_seconds = perf_counter() - start
        start = perf_counter()
        augmented, aug_info = fit_head(inputs, labels, bases, refit_score)
        aug_refit_seconds = perf_counter() - start
        score_calls = {
            "gp_discovery": gp.calls,
            "gp_validation": gp.validation_calls,
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
        if self.class_head is not None:
            self.counts_["class_head"] = self.class_head
            self.counts_["class_head_max_evaluations"] = flat_info["max_evaluations"]
        if self.gp_validation_top_k is not None:
            self.counts_["gp_validation_top_k"] = self.gp_validation_top_k
            self.counts_["gp_final_eligible"] = min(self.gp_validation_top_k, gp.n_trees)
        if self.l2_strength or self.class_head is not None:
            self.counts_["l2_strength"] = self.l2_strength
            self.counts_["head_fit_scores"] = {
                stage: {
                    "raw_score": info["score"],
                    "penalty": info["penalty"],
                    "penalized_score": info["penalized_score"],
                }
                for stage, info in (
                    ("flat_discovery", flat_d_info),
                    ("augmented_discovery", aug_d_info),
                    ("flat_refit", flat_info),
                    ("augmented_refit", aug_info),
                )
            }
        if self.pair_basis:
            # The common screen gives precedence to raw providers and earlier
            # GP programs. Only genuinely novel minima can incur trial calls.
            trials = screen.pop("evaluations_by_archive_index")
            pair_trials = [calls for i, calls in trials.items() if i >= gp_archive_size]
            selected = [
                item["archive_index"]
                for item in screen["selected"]
                if item["archive_index"] >= gp_archive_size
            ]
            self.counts_["pair_basis"] = {
                **pair_info,
                "gp_archive_size": gp_archive_size,
                "unique_novel_features": len(pair_trials),
                "screen_calls": sum(pair_trials),
                "additional_discovery_calls": pair_info["evaluations"]
                + sum(pair_trials),
                "selected_pair_count": len(selected),
                "selected_pairs": [
                    pair_info["pairs"][i - gp_archive_size] for i in selected
                ],
                "selected_archive_indices": selected,
            }
            score_calls["pair_ranking"] = pair_info["evaluations"]
            self.counts_["total_score_calls"] = sum(score_calls.values())
            self.counts_["seconds"]["pair_basis"] = pair_seconds
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


class RegularizedNativePatternOKAPI(NativePatternOKAPI):
    """Native pattern search with the fixed variance-scaled L2 strength of 0.25."""

    name = "okapi_native_pattern_l2"

    def __init__(self, seed=0, *, split_seed=None, gp_validation_top_k=None):
        super().__init__(seed=seed, split_seed=split_seed, l2_strength=0.25,
                         gp_validation_top_k=gp_validation_top_k)

    @classmethod
    def available(cls):
        from inspect import signature

        if not super().available():
            return False
        from okapi.pattern_logit import fit_pattern_tree

        return "l2_strength" in signature(fit_pattern_tree).parameters

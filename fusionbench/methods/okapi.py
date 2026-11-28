"""Adapt the optional OKAPI tree search to FusionBench probability banks.

Training reads one NumPy file per provider. Prediction evaluates the selected
tree directly on a query bank with the same provider and class order.
"""

from __future__ import annotations

import tempfile
from contextlib import ExitStack, contextmanager
from pathlib import Path
from threading import RLock

import numpy as np

from fusionbench.eval.metrics import normalize_proba
from fusionbench.methods.base import Method


def _identity(x):
    return x


_okapi_lock = RLock()


@contextmanager
def _okapi_context(n_classes, *, training=False):
    from okapi import globals as state
    from okapi.backend.numpy_backend import NumpyBackend
    from okapi.functions import scale_vector_to_sum_1

    # The upstream API uses process-wide backend, postprocessing and training RNG.
    # Reapply the fitted task at prediction and preserve the caller's settings.
    with _okapi_lock, ExitStack() as stack:
        if training:
            stack.callback(np.random.set_state, np.random.get_state())
        if state.get_backend() is not NumpyBackend:
            state.set_backend("numpy")
            stack.callback(state.set_backend, "pytorch")
        function = scale_vector_to_sum_1 if n_classes > 2 else _identity
        scoped = getattr(state, "use_postprocessing_function", None)
        if scoped is not None:
            stack.enter_context(scoped(function))
        else:
            wrapper = state.postprocessing_function
            stack.callback(
                wrapper.set_postprocessing_function, wrapper._postprocessing_function
            )
            wrapper.set_postprocessing_function(function)
        yield


def _macro_f1_objective(prediction, gt):
    """macro-F1 of an OKAPI tree prediction.

    Binary predictions are single-channel ``[N, 1]`` positive probabilities
    (thresholded at 0.5); multiclass are ``[N, C]`` (argmax).
    """
    from sklearn.metrics import f1_score

    p = np.asarray(prediction)
    y = np.asarray(gt).ravel().astype(int)
    yhat = (p[:, 0] > 0.5).astype(int) if p.shape[1] == 1 else p.argmax(axis=1)
    return float(f1_score(y, yhat, average="macro", zero_division=0))


def _brier_objective(prediction, gt):
    """Brier skill on OKAPI's required [0, 1] fitness scale."""
    p = np.asarray(prediction)
    y = np.asarray(gt).ravel().astype(int)
    if p.shape[1] == 1:
        return 1.0 - float(np.mean((p[:, 0] - y) ** 2))
    return 1.0 - 0.5 * float(np.mean(np.sum((p - np.eye(p.shape[1])[y]) ** 2, axis=1)))


class OKAPIMethod(Method):
    name = "okapi"

    def __init__(
        self,
        population_size: int = 30,
        generations: int = 150,
        population_multiplier: int = 3,
        tournament_size: int = 5,
        seed: int = 0,
        minimize_node_count: bool = True,
        n_restarts: int = 2,
        n_seeds: int = 30,
        fitness: str = "macro_f1",
        max_depth: int | None = None,
        max_nodes: int | None = None,
    ):
        if fitness not in {"macro_f1", "brier"}:
            raise ValueError("fitness must be macro_f1 or brier")
        if any(cap is not None and cap < 1 for cap in (max_depth, max_nodes)):
            raise ValueError("tree caps must be positive")
        if (
            min(
                population_size,
                generations,
                population_multiplier,
                tournament_size,
                n_restarts,
            )
            < 1
            or n_seeds < 0
        ):
            raise ValueError("Search sizes must be positive and n_seeds nonnegative")
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self.fitness = fitness
        self.population_size = population_size
        self.generations = generations
        self.population_multiplier = population_multiplier
        self.tournament_size = tournament_size
        self.seed = seed
        self.minimize_node_count = minimize_node_count
        self.n_restarts = n_restarts
        self.n_seeds = n_seeds

    @classmethod
    def available(cls) -> bool:
        try:
            from okapi.okapi import Okapi  # noqa: F401

            return True
        except ImportError:
            return False

    def _to_okapi(self, preds: np.ndarray) -> np.ndarray:
        """Binary -> single positive-probability channel so min/max/threshold
        operators are coherent; multiclass kept as full ``[N, C]``."""
        return preds[:, :, 1:2] if preds.shape[2] == 2 else preds

    def _seed_population(self, okapi, rng):
        """Inject diverse depth-1 compositions (min/max/mean over random model
        subsets) into the initial population. Standard GP diverse-init: gives
        crossover the building blocks to assemble deep structures reliably,
        without supplying any specific target rule."""
        from okapi.node import MaxNode, MeanNode, MinNode, ValueNode
        from okapi.tree import Tree

        ids, models = okapi.ids, okapi.models
        k = len(ids)
        max_inputs = (
            min(k, 4) if self.max_nodes is None else min(k, 4, self.max_nodes - 1)
        )
        if max_inputs < 2 or (self.max_depth is not None and self.max_depth < 3):
            return
        ops = [MinNode, MaxNode, MeanNode]
        seeds = []
        for _ in range(self.n_seeds):
            size = int(rng.integers(2, max_inputs + 1))
            pick = rng.choice(k, size=size, replace=False)
            root = ValueNode(None, models[pick[0]], ids[pick[0]])
            children = [ValueNode(None, models[j], ids[j]) for j in pick[1:]]
            op_cls = ops[int(rng.integers(len(ops)))]
            root.add_child(op_cls(children))
            seeds.append(Tree.create_tree_from_root(root))
        okapi.population += seeds
        okapi.population_size = min(40, okapi.population_size + len(seeds))
        okapi.fitnesses = None

    def fit(self, fit_preds, fit_y, *, task, n_classes):
        super().fit(fit_preds, fit_y, task=task, n_classes=n_classes)
        self.n_providers_ = fit_preds.shape[1]
        data = self._to_okapi(np.asarray(fit_preds, dtype=np.float64))
        with (
            _okapi_context(n_classes, training=True),
            tempfile.TemporaryDirectory(prefix="okapi_") as temp,
        ):
            if self.n_providers_ == 1:
                from okapi.node import ValueNode
                from okapi.tree import Tree

                self.okapi = None
                self.best_tree = Tree.create_tree_from_root(
                    ValueNode(None, data[:, 0], "m0.npy")
                )
                self.best_nodes = self.best_tree.nodes_count
                return self
            directory = Path(temp)
            fit_dir = directory / "fit"
            fit_dir.mkdir()
            for i in range(self.n_providers_):
                np.save(fit_dir / f"m{i}.npy", data[:, i])
            gt_path = directory / "gt.npy"
            np.save(gt_path, np.asarray(fit_y, dtype=np.int64).ravel())
            return self._fit_tree(fit_dir, gt_path)

    def _fit_tree(self, fit_dir, gt_path):
        from okapi.okapi import Okapi
        from okapi.operators import (
            CLOSE_THRESHOLD,
            FAR_THRESHOLD,
            MAX,
            MEAN,
            MIN,
            WEIGHTED_MEAN,
        )
        from okapi.pareto import maximize

        objective = (
            _macro_f1_objective if self.fitness == "macro_f1" else _brier_objective
        )

        # OKAPI seeds one single-model tree per base model, so the population
        # cannot exceed the number of models (complexity grows via evolution).
        pop = max(2, min(self.population_size, self.n_providers_))
        tournament = max(2, min(self.tournament_size, pop))
        ops = (MEAN, MIN, MAX, WEIGHTED_MEAN, FAR_THRESHOLD, CLOSE_THRESHOLD)

        best_tree, best_fit = None, -np.inf
        # Older OKAPI versions can still run an unbounded search.
        caps = {
            k: v
            for k, v in (("max_nodes", self.max_nodes), ("max_depth", self.max_depth))
            if v is not None
        }
        for r in range(self.n_restarts):
            okapi = Okapi(
                preds_source=fit_dir,
                gt_path=gt_path,
                population_size=pop,
                population_multiplier=self.population_multiplier,
                tournament_size=tournament,
                minimize_node_count=self.minimize_node_count,
                objective_functions=(objective,),
                objectives=(maximize,),
                allowed_ops=ops,
                backend=None,
                seed=self.seed + r,
                postprocessing_function=None,
                **caps,
            )
            self._seed_population(okapi, np.random.default_rng(self.seed + r))
            okapi.train(self.generations)
            trees = okapi.pareto_trees
            fits = np.asarray(okapi.pareto_fitnesses)
            i = int(np.argmax(fits[:, 0]))
            if fits[i, 0] > best_fit:
                best_fit, best_tree, self.okapi = fits[i, 0], trees[i], okapi

        self.best_tree = best_tree
        self.best_nodes = self.best_tree.nodes_count
        return self

    def predict(self, preds):
        if preds.shape[1:] != (self.n_providers_, self.n_classes):
            raise ValueError("Prediction provider and class axes must match fitting")
        data = self._to_okapi(np.asarray(preds, dtype=np.float64))
        providers = {f"m{i}.npy": data[:, i, :] for i in range(data.shape[1])}
        with _okapi_context(self.n_classes):
            out = self.best_tree.do_pred_on_another_tensors(current_tensors=providers)
        arr = np.asarray(out)
        if arr.ndim == 1:
            arr = arr[:, None]
        if self.n_classes == 2:
            pos = np.clip(arr[:, 0], 0.0, 1.0)
            return np.stack([1.0 - pos, pos], axis=1)
        return normalize_proba(arr)

    def tree_repr(self) -> str:
        try:
            return repr(self.best_tree)
        except Exception:
            return "<okapi tree>"

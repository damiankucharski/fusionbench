"""OKAPI adapter (genetic-programming fusion-tree search).

Wraps the real OKAPI library (dev branch). OKAPI evolves an alternating tree of
operators (mean / min / max / weighted-mean / close- & far-threshold) over the
base-model prediction tensors, with a parsimony (node-count) Pareto objective.

Implementation notes (verified against the installed package):
- numpy backend by default; predictions are materialised as one ``.npy`` per
  model in a temp dir (``B.load`` -> ``np.load``); ground truth as ``gt.npy``.
- objective functions are pluggable ``(prediction, gt) -> float``; we pass a
  numpy/sklearn macro-F1 so no torch metrics are needed (torch itself is still
  imported by ``okapi.fitness`` at module load).
- the evolved tree is evaluated on a fresh split via
  ``tree.do_pred_on_another_tensors(preds_directory=...)``.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np

from fusionbench.eval.metrics import normalize_proba
from fusionbench.methods.base import Method


def _identity(x):
    return x


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

    def __init__(self, population_size: int = 30, generations: int = 150,
                 population_multiplier: int = 3, tournament_size: int = 5, seed: int = 0,
                 minimize_node_count: bool = True, n_restarts: int = 2, n_seeds: int = 30,
                 fitness: str = "macro_f1", max_depth: int | None = None,
                 max_nodes: int | None = None):
        if fitness not in {"macro_f1", "brier"}:
            raise ValueError("fitness must be macro_f1 or brier")
        if any(cap is not None and cap < 1 for cap in (max_depth, max_nodes)):
            raise ValueError("tree caps must be positive")
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
        self._tmp = tempfile.TemporaryDirectory(prefix="okapi_")

    @classmethod
    def available(cls) -> bool:
        try:
            import okapi  # noqa: F401
            import torch  # noqa: F401
            return True
        except Exception:
            return False

    def _to_okapi(self, preds: np.ndarray) -> np.ndarray:
        """Binary -> single positive-probability channel so min/max/threshold
        operators are coherent; multiclass kept as full ``[N, C]``."""
        return preds[:, :, 1:2] if preds.shape[2] == 2 else preds

    def _write_split(self, preds: np.ndarray, split: str) -> Path:
        arr = self._to_okapi(preds)
        d = Path(self._tmp.name) / split
        # Clear any stale files: the instance is reused across scenarios with
        # differing model counts, and OKAPI globs the directory.
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        for k in range(arr.shape[1]):
            np.save(d / f"m{k}.npy", arr[:, k, :].astype(np.float64))
        return d

    def _seed_population(self, okapi, rng):
        """Inject diverse depth-1 compositions (min/max/mean over random model
        subsets) into the initial population. Standard GP diverse-init: gives
        crossover the building blocks to assemble deep structures reliably,
        without supplying any specific target rule."""
        from okapi.node import MaxNode, MeanNode, MinNode, ValueNode
        from okapi.tree import Tree

        ids, models = okapi.ids, okapi.models
        k = len(ids)
        max_inputs = min(k, 4) if self.max_nodes is None else min(k, 4, self.max_nodes - 1)
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
        from okapi.functions import scale_vector_to_sum_1
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

        objective = _macro_f1_objective if self.fitness == "macro_f1" else _brier_objective
        fit_dir = self._write_split(fit_preds, "fit")
        gt_path = Path(self._tmp.name) / "gt.npy"
        np.save(gt_path, fit_y.ravel().astype(np.int64))

        # OKAPI seeds one single-model tree per base model, so the population
        # cannot exceed the number of models (complexity grows via evolution).
        pop = max(2, min(self.population_size, fit_preds.shape[1]))
        tournament = max(2, min(self.tournament_size, pop))
        # OKAPI's postprocessing is a global; always set it explicitly so a prior
        # multiclass scenario cannot leave softmax-normalisation active for a
        # later binary one (which would normalise the single channel to 1.0).
        post = scale_vector_to_sum_1 if task == "multiclass" else _identity
        ops = (MEAN, MIN, MAX, WEIGHTED_MEAN, FAR_THRESHOLD, CLOSE_THRESHOLD)

        best_tree, best_fit = None, -np.inf
        # Older OKAPI versions can still run an unbounded search.
        caps = {k: v for k, v in (("max_nodes", self.max_nodes), ("max_depth", self.max_depth)) if v is not None}
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
                backend="numpy",
                seed=self.seed + r,
                postprocessing_function=post,
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
        q_dir = self._write_split(preds, "query")
        out = self.best_tree.do_pred_on_another_tensors(preds_directory=str(q_dir))
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

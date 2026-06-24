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


class OKAPIMethod(Method):
    name = "okapi"

    def __init__(self, population_size: int = 30, generations: int = 100,
                 population_multiplier: int = 2, tournament_size: int = 5, seed: int = 0):
        self.population_size = population_size
        self.generations = generations
        self.population_multiplier = population_multiplier
        self.tournament_size = tournament_size
        self.seed = seed
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
        self.okapi = Okapi(
            preds_source=fit_dir,
            gt_path=gt_path,
            population_size=pop,
            population_multiplier=self.population_multiplier,
            tournament_size=tournament,
            minimize_node_count=True,
            objective_functions=(_macro_f1_objective,),
            objectives=(maximize,),
            allowed_ops=(MEAN, MIN, MAX, WEIGHTED_MEAN, FAR_THRESHOLD, CLOSE_THRESHOLD),
            backend="numpy",
            seed=self.seed,
            postprocessing_function=post,
        )
        self.okapi.train(self.generations)

        trees = self.okapi.pareto_trees
        fits = np.asarray(self.okapi.pareto_fitnesses)
        best = int(np.argmax(fits[:, 0]))  # best macro-F1 on the Pareto front
        self.best_tree = trees[best]
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

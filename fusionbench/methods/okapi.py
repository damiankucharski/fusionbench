"""OKAPI adapter (genetic-programming fusion-tree search).

Materialises base predictions as per-model tensors in OKAPI's value-node format,
runs the evolutionary search on the fit split, and evaluates the selected tree on
a query split. OKAPI + torch are optional; locally the method reports itself
unavailable and is skipped. The API calls below follow the OKAPI system
reference and are verified against the installed package on the GPU host.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from fusionbench.eval.metrics import normalize_proba
from fusionbench.methods.base import Method


class OKAPIMethod(Method):
    name = "okapi"

    def __init__(self, population_size: int = 20, generations: int = 100,
                 objective: str = "prf", seed: int = 0):
        self.population_size = population_size
        self.generations = generations
        self.objective = objective
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

    # --- tensor IO -------------------------------------------------------- #
    def _write_split(self, preds: np.ndarray, split: str) -> Path:
        import torch
        d = Path(self._tmp.name) / split
        d.mkdir(parents=True, exist_ok=True)
        for k, mid in enumerate(self.model_ids):
            torch.save(torch.tensor(preds[:, k, :], dtype=torch.float32), d / mid)
        return d

    def _fitness(self):
        from okapi import fitness as F
        task = self.task
        if self.objective == "rocpr":
            return [getattr(F, f"roc_auc_{task}"), getattr(F, f"average_precision_{task}")]
        # default prf-style: accuracy as the scalar performance driver
        return [getattr(F, f"accuracy_{task}")]

    def fit(self, fit_preds, fit_y, *, task, n_classes):
        super().fit(fit_preds, fit_y, task=task, n_classes=n_classes)
        import torch
        from okapi import operators as ops
        from okapi.functions import set_multiclass_postprocessing
        from okapi.okapi import Okapi

        self.model_ids = [f"m{i}" for i in range(fit_preds.shape[1])]
        if task == "multiclass":
            set_multiclass_postprocessing()

        fit_dir = self._write_split(fit_preds, "fit")
        tensor_paths = [str(fit_dir / m) for m in self.model_ids]
        y = torch.tensor(fit_y.ravel(), dtype=torch.long)

        operators = [ops.MEAN, ops.MAX, ops.MIN, ops.WEIGHTED_MEAN,
                     ops.CLOSE_THRESHOLD, ops.FAR_THRESHOLD]
        self.okapi = Okapi(
            tensor_paths=tensor_paths,
            y=y,
            fitnesses=self._fitness(),
            operators=operators,
            population_size=self.population_size,
            generations=self.generations,
            seed=self.seed,
        )
        self.okapi.fit()
        # Pick the Pareto member with best primary performance objective.
        self.best = self.okapi.best_individual()
        return self

    def predict(self, preds):
        import torch
        q_dir = self._write_split(preds, "query")
        tree = self.best
        tree.load_tensors_from_path(str(q_dir))
        out = tree.calculate()
        arr = out.detach().cpu().numpy() if hasattr(out, "detach") else np.asarray(out)
        if arr.ndim == 1:  # binary single-column -> [N, 2]
            arr = np.stack([1 - arr, arr], axis=1)
        return normalize_proba(arr)

    def tree_repr(self) -> str:
        try:
            return str(self.best)
        except Exception:
            return "<okapi tree>"

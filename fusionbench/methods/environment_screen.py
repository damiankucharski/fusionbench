"""Train a native fusion tree after screening providers across source environments."""

import numpy as np

from fusionbench.eval.suite_point_gp import _full
from fusionbench.methods.auto_configuration import _inputs, _task
from fusionbench.methods.averaging import SimpleAverage
from fusionbench.methods.base import Method
from fusionbench.methods.native_pattern import RegularizedNativePatternOKAPI


def _diagnostic_indices(indices, rows):
    if indices is None:
        return np.arange(rows)
    indices = np.asarray(indices)
    if indices.dtype.kind == "b":
        if indices.shape != (rows,):
            raise ValueError("A screen mask must have one entry per source row")
        indices = np.flatnonzero(indices)
    if (
        indices.ndim != 1
        or indices.dtype.kind not in "iu"
        or len(indices) == 0
        or np.any(indices < 0)
        or np.any(indices >= rows)
        or len(np.unique(indices)) != len(indices)
    ):
        raise ValueError("Supply nonempty, unique source row indices for screening")
    return indices.astype(np.intp, copy=False)


def _take_ids(values, indices, rows, name):
    if values is None:
        return None
    values = np.asarray(values, dtype=object)
    if values.shape != (rows,):
        raise ValueError(f"Supply one {name} ID per source row")
    return values[indices]


class EnvironmentScreenedOKAPI(Method):
    """Screen on observed source sites, then fit fusion on every source row.

    Adjusted mode uses pooled source context frequencies when comparing sites.
    Supply aligned environment/context vectors for the full source bank, even
    when screen_indices selects only a diagnostic subset. After fitting,
    kept_indices_ gives the retained providers in their original order.
    """

    name = "environment_screened_okapi"

    def __init__(
        self,
        mode="adjusted",
        threshold=0.10,
        z=1.96,
        min_cell=8,
        seed=0,
        split_seed=0,
        gp_validation_top_k=8,
    ):
        self.mode = mode
        self.threshold = threshold
        self.z = z
        self.min_cell = min_cell
        self.seed = seed
        self.split_seed = split_seed
        self.gp_validation_top_k = gp_validation_top_k

    @classmethod
    def available(cls):
        try:
            from okapi.environment_stability import environment_screen  # noqa: F401
        except ImportError:
            return False
        return RegularizedNativePatternOKAPI.available()

    def fit(
        self,
        x,
        y,
        *,
        task,
        n_classes,
        environments=None,
        contexts=None,
        screen_indices=None,
    ):
        from okapi.environment_stability import environment_screen

        x, y, _ = _inputs(x, y, None)
        if task != _task(x) or n_classes != x.shape[2]:
            raise ValueError("Class axis must match the supplied classification task")
        indices = _diagnostic_indices(screen_indices, len(y))
        info = environment_screen(
            x[indices],
            y[indices],
            _take_ids(environments, indices, len(y), "environment"),
            contexts=_take_ids(contexts, indices, len(y), "context")
            if self.mode == "adjusted"
            else None,
            mode=self.mode,
            threshold=self.threshold,
            z=self.z,
            min_cell=self.min_cell,
        )
        drop = info["drop"]
        kept = np.asarray([i for i in range(x.shape[1]) if i != drop])
        selected = x if drop is None else np.ascontiguousarray(x[:, kept, :])
        if len(kept) == 1:
            model = SimpleAverage()
        else:
            model = RegularizedNativePatternOKAPI(
                seed=self.seed,
                split_seed=self.split_seed,
                gp_validation_top_k=self.gp_validation_top_k,
            )
        model.fit(selected, y, task=task, n_classes=n_classes)
        super().fit(x, y, task=task, n_classes=n_classes)
        self.model_, self.drop_, self.kept_indices_ = model, drop, kept
        self.screen_info_, self.screen_indices_ = info, indices.copy()
        self.n_providers_ = x.shape[1]
        return self

    def predict(self, preds):
        """Predict from the original aligned provider axis, as supplied to fit."""
        if not hasattr(self, "model_"):
            raise ValueError("Fit EnvironmentScreenedOKAPI before prediction")
        x = _full(preds)
        if x.shape[1:] != (self.n_providers_, self.n_classes):
            raise ValueError("Expected the original provider and class axes")
        retained = (
            x
            if self.drop_ is None
            else np.ascontiguousarray(x[:, self.kept_indices_, :])
        )
        return self.model_.predict(retained)

    def predict_retained(self, preds):
        """Predict from only the provider columns in kept_indices_, in that order."""
        if not hasattr(self, "model_"):
            raise ValueError("Fit EnvironmentScreenedOKAPI before prediction")
        x = _full(preds)
        if x.shape[1:] != (len(self.kept_indices_), self.n_classes):
            raise ValueError(
                "Expected only the retained providers and the original classes"
            )
        return self.model_.predict(x if self.drop_ is None else np.ascontiguousarray(x))

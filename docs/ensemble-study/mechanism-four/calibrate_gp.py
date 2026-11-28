"""Recalibrate an already selected tree, with exactly the same underlying providers."""
from copy import copy

import numpy as np
from okapi.paired_fitness import arithmetic_mean_tree
from okapi.temperature import fit_brier_temperature

from fusionbench.methods.averaging import SimpleAverage
from fusionbench.methods.okapi import OKAPIMethod


def fit_calibrated(base, x, y, groups=None, penalized=True):
    """Use only fit-row forecasts, labels and optional scene identifiers."""
    if isinstance(base, SimpleAverage):
        adapter = OKAPIMethod()
        adapter.n_classes = x.shape[-1]
        adapter.task = "binary" if adapter.n_classes == 2 else "multiclass"
        inputs = adapter._to_okapi(np.asarray(x, dtype=float))
        tree = arithmetic_mean_tree({f"m{i}.npy": inputs[:, i] for i in range(x.shape[1])})
    else:
        adapter = copy(base)
        tree = base.best_tree
    fitted, info = fit_brier_temperature(tree, base.predict(x), y, groups=groups, penalized=penalized)
    if info["beta"] == 1 and isinstance(base, SimpleAverage):
        # Preserve the reference's exact summation order and probability flooring.
        adapter = copy(base)
    adapter.best_tree = fitted
    adapter.best_nodes = fitted.nodes_count
    adapter.calibration_ = info
    return adapter

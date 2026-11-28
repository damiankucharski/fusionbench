"""Compare a separately trained regularized OKAPI candidate with arithmetic averaging."""

import numpy as np

from fusionbench.methods.averaging import SimpleAverage
from fusionbench.methods.native_pattern import RegularizedNativePatternOKAPI


def gate_choices(x, y):
    """Fit exclusively on source inputs. True deploys a full-source OKAPI refit."""
    labels = np.asarray(y).reshape(-1)
    classes = x.shape[-1]
    task = "binary" if classes == 2 else "multiclass"
    rng = np.random.default_rng(7301)
    held = []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        if len(indices) >= 2:
            held.extend(rng.permutation(indices)[: max(1, len(indices) // 4)])
    held = np.array(sorted(held), dtype=int)
    if len(held) < 2:
        return {"gain": 0.0, "se": 0.0, "plain": False, "guard": False}
    remaining = np.ones(len(labels), dtype=bool)
    remaining[held] = False
    challenger = RegularizedNativePatternOKAPI(seed=0, split_seed=0)
    challenger.fit(x[remaining], labels[remaining], task=task, n_classes=classes)
    candidate = challenger.predict(x[held])
    avg = SimpleAverage().fit(
        x[remaining], labels[remaining], task=task, n_classes=classes
    )
    midpoint = avg.predict(x[held])
    target = np.eye(classes)[labels[held]]
    diff = np.sum((midpoint - target) ** 2 - (candidate - target) ** 2, axis=1)
    gain = float(np.mean(diff))
    se = float(np.std(diff, ddof=1) / np.sqrt(len(diff)))
    return {"gain": gain, "se": se, "plain": gain > 0, "guard": gain > se}

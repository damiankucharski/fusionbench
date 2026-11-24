"""Test-time paired provider outage, with the affected pair chosen on source rows."""

import numpy as np


def coherent_pair(x_dev, x_clean_test, seed):
    dev = np.asarray(x_dev, dtype=float)
    test = np.asarray(x_clean_test)
    if dev.ndim != 3 or test.ndim != 3 or dev.shape[1:] != test.shape[1:]:
        raise ValueError("Expected matching source and target provider/class axes")
    if min(len(dev), len(test), dev.shape[1]) < 2:
        raise ValueError("Two providers and two rows per split are required")
    centered = (dev - dev.mean(axis=0)).transpose(1, 0, 2).reshape(dev.shape[1], -1)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    unit = np.divide(centered, norms, out=np.zeros_like(centered), where=norms > 0)
    correlations = unit @ unit.T
    pairs = np.triu_indices(len(unit), 1)
    best = np.argmax(correlations[pairs])
    i, j = int(pairs[0][best]), int(pairs[1][best])
    rng = np.random.default_rng([seed, 903])
    order = rng.permutation(len(test))
    donor = np.empty_like(order)
    donor[order] = np.roll(order, 1)
    rows = np.flatnonzero(rng.random(len(test)) < 0.5)
    target = test.copy()
    for col in (i, j):
        target[rows, col] = test[donor[rows], col]
    return target, (i, j)

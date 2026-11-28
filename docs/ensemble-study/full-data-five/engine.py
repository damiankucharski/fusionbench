"""Full-data native GP and matched discovery/selection controls for round one."""

from copy import copy

import numpy as np

from fusionbench.methods.okapi import OKAPIMethod

# Run the ordinary OKAPI entry point with a bounded search budget on every bank.
BASE_OPTIONS = {
    "generations": 8,
    "n_restarts": 1,
    "n_seeds": 0,
    "population_size": 8,
    "population_multiplier": 2,
    "tournament_size": 3,
    "max_nodes": 31,
    "max_depth": 7,
    "fitness": "brier",
}


def fit_model(model, x, y):
    classes = x.shape[-1]
    if isinstance(model, OKAPIMethod):
        # The all-provider mean needs M leaves and one operator, including on needle.
        model.max_nodes = max(BASE_OPTIONS["max_nodes"], x.shape[1] + 1)
    return model.fit(x, y, task="binary" if classes == 2 else "multiclass", n_classes=classes)


def fit_full(x, y, seed=0):
    """Evolve on every supplied row, returning the ordinary native Pareto champion."""
    return fit_model(OKAPIMethod(seed=seed, **BASE_OPTIONS), x, y)


def partition(n, groups=None):
    """Reserve 30% rounded up, sampling complete scenes when group IDs exist."""
    if n < 2:
        raise ValueError("The split control requires at least two fit rows")
    rng = np.random.default_rng([0, 51])
    if groups is None:
        return np.split(rng.permutation(n), [7 * n // 10])
    groups = np.asarray(groups)
    if groups.shape != (n,):
        raise ValueError("Supply one scene ID per fit row")
    ids = np.unique(groups)
    if len(ids) < 2:
        raise ValueError("The grouped split requires at least two scenes")
    first, second = np.split(rng.permutation(ids), [7 * len(ids) // 10])
    return np.flatnonzero(np.isin(groups, first)), np.flatnonzero(np.isin(groups, second))


def select_on_heldout(reduced, x, y):
    """Select a terminal tree using only this fixed set of held-out fit examples.

    The ordinary discovery champion is first and wins equal-loss ties. Remaining
    programs follow the native final population order after structural deduplication.
    All predictions use the normal adapter's binary/multiclass postprocessing.
    """
    unique = {}
    for tree in (reduced.best_tree, *reduced.okapi.population):
        unique.setdefault(tree.structural_key(), tree)

    selected = copy(reduced)
    truth = np.eye(reduced.n_classes)[y]
    candidates = list(unique.values())
    losses = []
    for tree in candidates:
        selected.best_tree = tree
        p = selected.predict(x)
        losses.append(float(np.square(p - truth).sum(axis=1).mean()))

    winner = int(np.argmin(losses))
    selected.best_tree = candidates[winner]
    selected.best_nodes = selected.best_tree.nodes_count
    selected.selection_ = {
        "eligible": len(candidates),
        "winner_index": winner,
        "champion_heldout_brier": losses[0],
        "winner_heldout_brier": losses[winner],
        "n_selection_rows": len(y),
    }
    return selected


def fit_round1(x, y, seed=0, groups=None):
    """Full data and two choices from one shared reduced-data evolution."""
    discovery, heldout = partition(len(y), groups)
    full = fit_full(x, y, seed)
    reduced = fit_full(x[discovery], y[discovery], seed)
    split = select_on_heldout(reduced, x[heldout], y[heldout])
    return {"full": full, "reduced": reduced, "split": split}

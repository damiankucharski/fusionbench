"""Native seeds with fewer operands and the historical full-data training objective."""

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from okapi.paired_fitness import arithmetic_mean_tree
from okapi.prediction_pull import PredictionPullBrier
from okapi.sparse_mean import sparse_mean_size, sparse_mean_tree

from fusionbench.eval.suite_point_gp import _bind, predict_tree
from fusionbench.methods.okapi import OKAPIMethod

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "full-data-five"))
from engine import BASE_OPTIONS, fit_model


def _seed(tensors, random_subset):
    # The legacy file-backed adapter loads m10 before m2. Ranking ties and
    # constructing both sparse policies use the input tensor's column order.
    keys = [f"m{i}.npy" for i in range(len(tensors))]
    ordered = {key: tensors[key] for key in keys}
    if random_subset:
        k = sparse_mean_size(len(keys))
        indices = np.sort(np.random.default_rng(421).choice(len(keys), size=k, replace=False))
        return arithmetic_mean_tree({keys[i]: ordered[keys[i]] for i in indices}), tuple(indices.tolist())
    tree, selected = sparse_mean_tree(ordered)
    selected_set = set(selected)
    return tree, tuple(i for i, key in enumerate(keys) if key in selected_set)


class SparseSeedGP(OKAPIMethod):
    """Change only the extra initial tree in the registered pull-mean pipeline."""

    def __init__(self, name, seed=0):
        if name not in {"sparse_gp", "random_gp"}:
            raise ValueError(f"Unknown policy {name}")
        super().__init__(**BASE_OPTIONS, seed=seed)
        self.random_subset = name == "random_gp"

    def _seed_population(self, okapi, rng):
        super()._seed_population(okapi, rng)
        tree, self.selected_indices_ = _seed(okapi.train_tensors, self.random_subset)
        if not okapi._within_caps(tree):
            raise ValueError("The starting tree exceeds the structural budget")
        labels = okapi.gt_tensor
        full_mean = arithmetic_mean_tree(okapi.train_tensors)
        objective = PredictionPullBrier(full_mean.predict(), labels, strength=np.sqrt(8 / len(labels)))
        okapi.objective_functions = (objective,)
        self.seed_tree_ = tree.copy()
        okapi.population.append(tree)
        okapi.population_size += 1
        okapi.fitnesses = None


@dataclass
class SparsePool:
    best_tree: object
    selected_indices_: tuple

    def predict(self, x):
        return predict_tree(self.best_tree, x)


def fit_sparse_pool(x, *, random_subset=False):
    """Select and deploy a native mean tree solely from aligned fit forecasts."""
    return SparsePool(*_seed(_bind(x), random_subset))


def fit_sparse_gp(name, x, y, seed=0):
    return fit_model(SparseSeedGP(name, seed=seed), x, y)


def fit_sparse(x, y, seed=0):
    """Matched discovery policies and the two seeds before supervised search."""
    models = {}
    for name in ("sparse", "random"):
        gp = fit_sparse_gp(f"{name}_gp", x, y, seed)
        models[f"{name}_gp"] = gp
        models[f"{name}_pool"] = SparsePool(gp.seed_tree_, gp.selected_indices_)
    return models

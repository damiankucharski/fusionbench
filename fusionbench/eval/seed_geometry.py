"""Paired label-free native tree initializations using four anonymous providers."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np


@dataclass(frozen=True)
class SeedPlan:
    provider_ids: tuple
    inner: str


def make_seed_plans(provider_ids, seed, count=30):
    """Draw four ordered provider IDs and a fair min/max orientation per plan.

    Without enough distinct providers every provider is used once, and extra
    positions are drawn with replacement before shuffling all four positions.
    The private NumPy stream does not consume the native optimizer's RNG.
    """
    names = tuple(provider_ids)
    if not names or len(set(names)) != len(names):
        raise ValueError("Provide nonempty, distinct provider identifiers")
    if isinstance(count, bool) or not isinstance(count, Integral) or count < 0:
        raise ValueError("count must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    plans = []
    m = len(names)
    for _ in range(count):
        if m >= 4:
            idx = rng.choice(m, size=4, replace=False)
        else:
            idx = np.concatenate(
                (np.arange(m), rng.choice(m, size=4 - m, replace=True))
            )
            idx = idx[rng.permutation(4)]
        plans.append(
            SeedPlan(tuple(names[i] for i in idx), ("min", "max")[rng.integers(2)])
        )
    return tuple(plans)


def geometry_trees(providers, plans, shape):
    """Materialize fresh native trees in plan order against an ID-to-tensor map."""
    from okapi.node import MaxNode, MinNode, ValueNode
    from okapi.tree import Tree

    if shape not in {"branch", "chain"}:
        raise ValueError("Expected shape 'branch' or 'chain'")
    trees = []
    for plan in plans:
        if plan.inner not in {"min", "max"} or len(plan.provider_ids) != 4:
            raise ValueError("Each plan requires four IDs and a min/max orientation")
        inner, outer = (MinNode, MaxNode) if plan.inner == "min" else (MaxNode, MinNode)
        a, b, c, d = [
            ValueNode(None, providers[name], name) for name in plan.provider_ids
        ]
        a.add_child(inner([b]))
        if shape == "branch":
            c.add_child(inner([d]))
            a.add_child(outer([c]))
        else:
            a.add_child(outer([c]))
            a.add_child(inner([d]))
        trees.append(Tree.create_tree_from_root(a))
    return trees


def append_geometry_seeds(engine, plans, shape):
    """Add the native seeds with the public adapter's usual population-size cap."""
    providers = dict(zip(engine.ids, engine.models, strict=True))
    seeds = geometry_trees(providers, plans, shape)
    engine.population += seeds
    engine.population_size = min(40, engine.population_size + len(seeds))
    engine.fitnesses = None


def replace_deep_seeds(engine, seed, shape, singletons):
    """Replace the final six shallow starters, preserving native RNG and population size."""
    from okapi.node import MaxNode, MinNode, ValueNode
    from okapi.tree import Tree

    if shape not in ("mixed_branch", "mixed_chain"):
        raise ValueError("Expected an eight-input branch or chain")
    if len(engine.population) - singletons != 30:
        raise ValueError("The matched study expects exactly 30 native seeds")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 8301]))
    n = len(engine.ids)
    for slot in range(len(engine.population) - 6, len(engine.population)):
        if n >= 8:
            indices = rng.choice(n, 8, replace=False)
        else:
            indices = rng.permutation(np.r_[np.arange(n), rng.choice(n, 8 - n)])
        leaves = [ValueNode(None, engine.models[i], engine.ids[i]) for i in indices]
        inner, middle = (MinNode, MaxNode) if rng.integers(2) else (MaxNode, MinNode)
        if shape == "mixed_branch":
            level = leaves
            for operator in (inner, middle, inner):
                for left, right in zip(level[::2], level[1::2], strict=True):
                    left.add_child(operator([right]))
                level = level[::2]
        else:
            inorder = (inner, middle, inner, inner, inner, middle, inner)
            for operator, right in zip(inorder, leaves[1:], strict=True):
                leaves[0].add_child(operator([right]))
        engine.population[slot] = Tree.create_tree_from_root(leaves[0])

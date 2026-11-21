"""Test topology-controlled native seed trees and the stock adapter initialization."""

import hashlib
from collections import Counter
from dataclasses import FrozenInstanceError
from itertools import product

import numpy as np
import pytest

from fusionbench.eval.suite_point_gp import fit_suite_point, predict_tree


def inputs(classes, providers=7):
    rng = np.random.default_rng([84, classes, providers])
    train = rng.dirichlet(np.ones(classes), size=(32, providers))
    validation = rng.dirichlet(np.ones(classes), size=(15, providers))
    y = rng.permutation(np.arange(len(train)) % classes)
    vy = rng.permutation(np.arange(len(validation)) % (classes - 1))
    return train, y, validation, vy


def test_seed_plans_are_shared_immutable_and_handle_small_provider_counts():
    from fusionbench.eval.seed_geometry import make_seed_plans

    for n in (1, 2, 3, 4, 7):
        ids = tuple(f"m{i}.npy" for i in range(n))
        plans = make_seed_plans(ids, 71)
        assert len(plans) == 30
        assert plans == make_seed_plans(ids, 71)
        assert len(make_seed_plans(ids, 71, count=5)) == 5
        assert {plan.inner for plan in plans} == {"min", "max"}
        for plan in plans:
            assert isinstance(plan.provider_ids, tuple)
            assert len(plan.provider_ids) == 4
            assert set(plan.provider_ids) <= set(ids)
            assert len(set(plan.provider_ids)) == min(4, n)
        with pytest.raises((FrozenInstanceError, AttributeError)):
            plans[0].inner = "other"
        if n >= 2:
            assert plans != make_seed_plans(ids, 72)


def canonical_probability(x):
    mass = x.sum(axis=1, keepdims=True)
    empty = mass == 0
    return (x + empty) / np.where(empty, x.shape[1], mass)


def expected(providers, plan, shape, classes):
    a, b, c, d = (providers[i] for i in plan.provider_ids)
    inner, outer = (
        (np.minimum, np.maximum) if plan.inner == "min" else (np.maximum, np.minimum)
    )

    def update(op, left, right):
        x = op(left, right)
        return x if classes == 2 else canonical_probability(x)

    left = update(inner, a, b)
    if shape == "branch":
        return update(outer, left, update(inner, c, d))
    return update(inner, update(outer, left, c), d)


def assert_topology(tree, plan, shape):
    a, b, c, d = plan.provider_ids
    inside = {"min": "MinNode", "max": "MaxNode"}[plan.inner]
    outside = {"MinNode": "MaxNode", "MaxNode": "MinNode"}[inside]
    root = tree.root
    assert tree.nodes_count == 7
    assert root.id == a

    def operand(operator, name):
        assert type(operator).__name__ == name
        assert len(operator.children) == 1
        return operator.children[0]

    assert operand(root.children[0], inside).id == b
    center = operand(root.children[1], outside)
    assert center.id == c
    if shape == "branch":
        assert len(root.children) == 2
        assert operand(center.children[0], inside).id == d
    else:
        assert len(root.children) == 3 and not center.children
        assert operand(root.children[2], inside).id == d

    def walk(node):
        yield node
        for kid in node.children:
            yield from walk(kid)

    nodes = list(walk(root))
    assert Counter(n.id for n in nodes if hasattr(n, "id")) == Counter(
        plan.provider_ids
    )
    assert Counter(type(n).__name__ for n in nodes if not hasattr(n, "id")) == Counter(
        {inside: 2, outside: 1}
    )


@pytest.mark.parametrize("classes", [2, 4])
def test_normalized_native_trees_realize_both_geometries(classes):
    pytest.importorskip("okapi")
    from okapi.functions import scale_vector_to_sum_1
    from okapi.globals import set_backend, use_postprocessing_function

    from fusionbench.eval.seed_geometry import geometry_trees, make_seed_plans

    if classes == 2:
        x = np.array(list(product((0.1, 0.9), repeat=4)), dtype=float)[:, :, None]
    else:
        x, *_ = inputs(classes, providers=4)
        x[0] = np.eye(classes)  # disjoint supports force zero-mass MIN updates
    providers = {f"m{i}.npy": x[:, i, :] for i in range(4)}
    plans = make_seed_plans(tuple(providers), 71)
    predictions = {}
    set_backend("numpy")
    transform = np.asarray if classes == 2 else scale_vector_to_sum_1
    with use_postprocessing_function(transform):
        for shape in ("branch", "chain"):
            trees = geometry_trees(providers, plans, shape)
            assert len(trees) == len(plans)
            results = []
            for tree, plan in zip(trees, plans, strict=True):
                assert_topology(tree, plan, shape)
                pred = tree.predict()
                np.testing.assert_allclose(
                    pred, expected(providers, plan, shape, classes), rtol=0, atol=3e-15
                )
                results.append(pred)
            predictions[shape] = np.stack(results)
    for sign in ("min", "max"):
        subset = [i for i, p in enumerate(plans) if p.inner == sign]
        assert (
            np.max(np.abs(predictions["branch"][subset] - predictions["chain"][subset]))
            > 0.05
        )


def archive(cp):
    return (
        cp.calls,
        cp.generation,
        cp.n_trees,
        cp.path_digest,
        cp.fit_brier,
        cp.validation_brier,
        cp.fit_tree.structural_key(),
        cp.validation_tree.structural_key(),
    )


@pytest.mark.parametrize("classes", (2, 4))
def test_stock_archive_and_initial_diversity_match_native_reference(
    classes, monkeypatch
):
    sdk = pytest.importorskip("okapi")
    from fusionbench.eval.seed_geometry_standard import fit_seed_geometry

    x, y, v, vy = inputs(classes)
    seed = 157
    budgets = (5, 36, 37, 38, 197)
    original = sdk.Okapi._calculate_fitnesses
    genotypes, hashes = [], []

    def read_initial(engine, trees):
        if genotypes:
            return original(engine, trees)
        genotypes.extend(t.structural_key() for t in trees)
        objectives = engine.objective_functions

        def observe(p, labels):
            hashes.append(
                hashlib.sha256(np.asarray(p, dtype="<f8").tobytes()).hexdigest()
            )
            return objectives[0](p, labels)

        engine.objective_functions = (observe,)
        try:
            return original(engine, trees)
        finally:
            engine.objective_functions = objectives

    with monkeypatch.context() as patch:
        patch.setattr(sdk.Okapi, "_calculate_fitnesses", read_initial)
        reference = fit_suite_point(
            x, y, v, vy, seed=seed, point_mutation=False, budgets=budgets
        )
    assert len(genotypes) == len(hashes) == 37
    result = fit_seed_geometry(x, y, v, vy, seed=seed, shape="stock", budgets=budgets)
    assert set(result) == {*budgets, "initial"}
    assert archive(result["initial"]) == archive(result[37])
    for cost in budgets:
        cp, old = result[cost], reference[cost]
        assert archive(cp) == archive(old)
        prefix = hashes[:cost]
        assert tuple(cp.diversity["initial_prediction_hashes"]) == tuple(prefix)
        assert cp.diversity["initial_trees"] == len(prefix)
        assert cp.diversity["initial_unique_genotypes"] == len(set(genotypes[:cost]))
        assert cp.diversity["seed_unique_fit_predictions"] == len(set(prefix[7:]))
        for field in ("fit_tree", "validation_tree"):
            np.testing.assert_array_equal(
                predict_tree(getattr(cp, field), v),
                predict_tree(getattr(old, field), v),
            )
    diversity = result["initial"].diversity
    assert diversity["initial_trees"] == 37
    assert diversity["initial_singletons"] == 7
    assert diversity["initial_seed_trees"] == 30
    assert diversity["initial_unique_genotypes"] == len(set(genotypes))
    assert diversity["initial_unique_fit_predictions"] == len(set(hashes))
    assert diversity["seed_unique_fit_predictions"] == len(set(hashes[7:]))
    assert tuple(diversity["initial_prediction_hashes"]) == tuple(hashes)

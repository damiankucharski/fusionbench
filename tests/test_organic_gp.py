import hashlib
import struct

import numpy as np
import pytest

from fusionbench.eval.native_switch import tensors
from fusionbench.eval.structural_key_audit import training_files


@pytest.fixture(autouse=True)
def isolated_native_globals(monkeypatch):
    pytest.importorskip("okapi")
    from okapi.backend.backend import Backend
    from okapi.globals import postprocessing_function

    monkeypatch.setattr(Backend, "_current_backend", Backend._current_backend)
    monkeypatch.setattr(
        postprocessing_function,
        "_postprocessing_function",
        postprocessing_function._postprocessing_function,
    )


def examples(seed=813):
    rng = np.random.default_rng(seed)
    x = rng.uniform(size=(48, 6))
    val = rng.uniform(size=(31, 6))
    y = np.maximum(np.minimum(x[:, 1], x[:, 2]), np.minimum(x[:, 0], x[:, 4])) > 0.36
    val_y = (
        np.maximum(np.minimum(val[:, 1], val[:, 2]), np.minimum(val[:, 0], val[:, 4]))
        > 0.36
    )
    return x, y, tensors(val), val_y


def native_checkpoints(paths, gt, validation, validation_y, seed, budgets, generations):
    from okapi import Okapi
    from okapi.operators import (
        CLOSE_THRESHOLD,
        FAR_THRESHOLD,
        MAX,
        MEAN,
        MIN,
        WEIGHTED_MEAN,
    )

    results, archive, raw = {}, {}, []
    path = hashlib.sha256()
    calls = generation = 0

    def objective(p, y):
        loss = float(np.mean(np.square(p.ravel() - y.ravel())))
        raw.append(loss)
        return 1.0 - loss

    def snapshot(name):
        fit_key = min(archive, key=lambda key: archive[key][0])
        val_key = min(archive, key=lambda key: (archive[key][1], archive[key][0]))
        results[name] = (
            calls,
            generation,
            path.hexdigest(),
            len(archive),
            fit_key,
            val_key,
            archive[fit_key][0],
            archive[val_key][1],
        )

    class Batched(Okapi):
        def _calculate_fitnesses(self, trees=None):
            nonlocal calls
            trees = self.population if trees is None else trees
            raw.clear()
            scores = super()._calculate_fitnesses(trees)
            for tree, loss in zip(trees, raw, strict=True):
                key = tree.structural_key()
                calls += 1
                path.update((repr(key) + "\n").encode())
                path.update(struct.pack("<d", loss))
                if key not in archive:
                    prediction = tree.do_pred_on_another_tensors(
                        current_tensors=validation
                    ).ravel()
                    archive[key] = (
                        loss,
                        float(np.mean(np.square(prediction - validation_y))),
                    )
                if calls in budgets:
                    snapshot(f"calls_{calls}")
            return scores

    gp = Batched(
        paths,
        gt,
        len(paths),
        2,
        2,
        backend="numpy",
        seed=seed,
        allowed_ops=(MEAN, MIN, MAX, WEIGHTED_MEAN, FAR_THRESHOLD, CLOSE_THRESHOLD),
        objective_functions=(objective,),
        postprocessing_function=np.asarray,
        reuse_survivor_fitness=False,
    )
    while generation < generations or calls < max(budgets):
        generation += 1
        gp.run_iteration()
        if generation == generations:
            snapshot(f"generation_{generation}")
    return results


@pytest.mark.parametrize(
    ("gp_seed", "budgets", "generations"),
    [(19, (3, 38, 113), 3), (73, (18, 34, 59), 11)],
)
def test_stock_matches_unmodified_native_batches(gp_seed, budgets, generations):
    from fusionbench.eval.organic_gp import Edit, organic_gp

    x, y, validation, val_y = examples()

    def trace_only(tree, op_path, child_index, rng, providers, arm):
        # Drawing many random values on the event stream must not move the GP stream.
        rng.random(13)
        assert arm == "stock"
        op = tree.root
        for step in op_path:
            op = op.children[step]
        assert 0 <= child_index < len(op.children)
        partner = next(k for k in providers if k != op.children[child_index].id)
        return Edit(
            tree, "MaxNode" if type(op).__name__ != "MaxNode" else "MinNode", partner
        )

    with training_files(x, y) as (paths, gt):
        expected = native_checkpoints(
            paths, gt, validation, val_y, gp_seed, budgets, generations
        )
        result = organic_gp(
            paths,
            gt,
            validation,
            val_y,
            gp_seed,
            "stock",
            event_seed=546,
            transform=trace_only,
            budgets=budgets,
            generation_checkpoint=generations,
        )
    assert any(e["operator_path"] is not None for e in result.events)
    assert result.checkpoints.keys() == expected.keys()
    for name, cp in result.checkpoints.items():
        assert (
            cp.calls,
            cp.generation,
            cp.path_digest,
            cp.n_trees,
            cp.fit_tree.structural_key(),
            cp.validation_tree.structural_key(),
            cp.fit_brier,
            cp.validation_brier,
        ) == expected[name]


@pytest.mark.parametrize(
    ("arm", "duplicate_parent"),
    [("both", False), ("both", True), ("type_only", False), ("provider_only", False)],
)
def test_exact_offspring_evaluation_and_survival_are_separate_from_an_equivalent_parent(
    monkeypatch, arm, duplicate_parent
):
    from okapi import Okapi
    from okapi.node import MaxNode, MinNode, ValueNode
    from okapi.tree import Tree

    from fusionbench.eval.organic_gp import Edit, organic_gp

    def binary(sources, op, name):
        root = ValueNode(None, sources["p0.npy"], "p0.npy")
        child = ValueNode(None, sources[name], name)
        root.add_child(op([child]))
        return Tree(root)

    initialize = Okapi._initialize_population

    def initial(engine):
        parents = initialize(engine)
        if duplicate_parent:
            parents[-1] = binary(engine.train_tensors, MinNode, "p2.npy")
        return parents

    type_changed = arm in ("both", "type_only")
    provider_changed = arm in ("both", "provider_only")
    before_type = MaxNode if type_changed else MinNode
    before_provider = "p1.npy" if provider_changed else "p2.npy"
    origin = "mutation" if arm == "provider_only" else "crossover"

    def one_child(engine, fitnesses):
        tree = binary(engine.train_tensors, before_type, before_provider)
        engine.additional_population = [
            tree if origin == "crossover" else engine.population[0].copy()
        ]
        return 1

    def one_mutant(engine):
        if origin == "mutation":
            engine.additional_population.append(
                binary(engine.train_tensors, before_type, before_provider)
            )
            return 1
        return 0

    monkeypatch.setattr(Okapi, "_initialize_population", initial)
    monkeypatch.setattr(Okapi, "_perform_crossovers", one_child)
    monkeypatch.setattr(Okapi, "_mutate_additional_population", one_mutant)

    transformed = []

    def apply_edit(tree, op_path, child_index, rng, providers, requested_arm):
        assert requested_arm == arm
        assert tree.root.children[0].children[0].id == before_provider
        assert type(tree.root.children[0]) is before_type
        assert tuple(op_path) == (0,) and child_index == 0
        after = binary(providers, MinNode, "p2.npy")
        transformed.append(after)
        # Both proposals are generated even when only one factor is applied.
        return Edit(
            after,
            "MinNode" if type_changed else "MaxNode",
            "p2.npy" if provider_changed else "p1.npy",
        )

    x = ((np.arange(64)[:, None] >> np.arange(6)) & 1).astype(float)
    y = np.minimum(x[:, 0], x[:, 2])
    with training_files(x, y) as (paths, gt):
        run = organic_gp(
            paths,
            gt,
            tensors(x),
            y,
            177,
            arm,
            event_seed=23,
            transform=apply_edit,
            budgets=(2, 9),
            generation_checkpoint=1,
        )

    assert len(transformed) == len(run.events) == 1
    event = run.events[0]
    assert (event["before_type"], event["after_type"]) == (
        before_type.__name__,
        "MinNode",
    )
    assert (event["before_provider"], event["after_provider"]) == (
        before_provider,
        "p2.npy",
    )
    assert event["changed"] and event["origin"] == origin
    assert bool(event["passed_dedup"]) == (not duplicate_parent)
    assert bool(event["evaluated"]) == (not duplicate_parent)
    assert bool(event["survived"]) == (not duplicate_parent)
    if duplicate_parent:
        assert event["equivalent_survived"]
        assert event["evaluation_call"] is None and event["fit_brier"] is None
    else:
        assert event["evaluation_call"] == 13 and event["fit_brier"] == 0.0

    finished = run.checkpoints["generation_1"]
    assert finished.calls == (12 if duplicate_parent else 13)
    assert finished.fit_tree.structural_key() == transformed[0].structural_key()
    assert finished.fit_brier == 0
    count = finished.ledger
    assert count["changed"] == 1
    assert count["type_changed"] == int(type_changed)
    assert count["provider_changed"] == int(provider_changed)
    assert count["opportunities"] == count["generations_completed"] == 1
    assert (count["evaluated"], count["survived"], count["filtered_duplicate"]) == (
        int(not duplicate_parent),
        int(not duplicate_parent),
        int(duplicate_parent),
    )
    # An interrupted batch has yet to score the final child or select survivors.
    partial = run.checkpoints["calls_9"].ledger
    assert (
        partial["survived"]
        == partial["evaluated"]
        == partial["generations_completed"]
        == 0
    )


def test_no_candidate_with_an_operator_leaves_the_generation_unchanged(monkeypatch):
    from okapi import Okapi

    from fusionbench.eval.organic_gp import organic_gp

    def singleton_offspring(engine, fitnesses):
        engine.additional_population = [parent.copy() for parent in engine.population]
        return len(engine.additional_population)

    monkeypatch.setattr(Okapi, "_perform_crossovers", singleton_offspring)
    monkeypatch.setattr(Okapi, "_mutate_additional_population", lambda engine: 0)

    def unexpected(*args):
        pytest.fail("A singleton with no operator cannot be edited")

    x, y, val, val_y = examples()
    with training_files(x, y) as (paths, gt):
        run = organic_gp(
            paths,
            gt,
            val,
            val_y,
            131,
            "both",
            event_seed=19,
            transform=unexpected,
            budgets=(2, 10),
            generation_checkpoint=1,
        )
    [event] = run.events
    assert event["eligible_children"] == 0
    assert event["operator_path"] is None
    ledger = run.checkpoints["generation_1"].ledger
    assert ledger["generations_completed"] == ledger["no_eligible"] == 1
    assert ledger["opportunities"] == ledger["changed"] == ledger["survived"] == 0

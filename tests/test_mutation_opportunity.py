"""Match both clocks to ordinary native GP and the original point-mutation benchmark."""

from pathlib import Path

import numpy as np
import pytest

from fusionbench.eval.suite_point_gp import fit_suite_point, predict_tree
from fusionbench.methods.okapi import OKAPIMethod

pytest.importorskip("okapi")
pytest.importorskip(
    "experiments.point_ablation.moves",
    reason="requires the optional OKAPI research controls",
)


@pytest.fixture(autouse=True)
def restore_settings(monkeypatch):
    from okapi.backend.backend import Backend
    from okapi.globals import postprocessing_function

    monkeypatch.setattr(Backend, "_current_backend", Backend._current_backend)
    monkeypatch.setattr(
        postprocessing_function,
        "_postprocessing_function",
        postprocessing_function._postprocessing_function,
    )


def inputs(classes):
    rng = np.random.default_rng([1375, classes])
    fit = rng.dirichlet(np.ones(classes), size=(26, 6))
    val = rng.dirichlet(np.ones(classes), size=(13, 6))
    fit_y = rng.permutation(np.arange(len(fit)) % classes)
    val_y = rng.permutation(np.arange(len(val)) % (classes - 1))
    return fit, fit_y, val, val_y


def native_boundary(fit, fit_y, seed, point):
    """Count true objective calls as the uninstrumented engine completes generation 2."""
    from okapi import Okapi
    from okapi.functions import scale_vector_to_sum_1
    from okapi.operators import (
        CLOSE_THRESHOLD,
        FAR_THRESHOLD,
        MAX,
        MEAN,
        MIN,
        WEIGHTED_MEAN,
    )

    calls = 0

    def objective(p, truth):
        nonlocal calls
        calls += 1
        y = truth.ravel()
        if p.shape[1] == 1:
            return 1 - float(np.mean(np.square(p[:, 0] - y)))
        return 1 - 0.5 * float(
            np.mean(np.sum(np.square(p - np.eye(p.shape[1])[y]), axis=1))
        )

    adapter = OKAPIMethod()
    try:
        folder = adapter._write_split(fit, "fit")
        target = Path(adapter._tmp.name) / "labels.npy"
        np.save(target, fit_y)
        n_providers = fit.shape[1]
        engine = Okapi(
            folder,
            target,
            n_providers,
            3,
            5,
            backend="numpy",
            seed=seed,
            operator_point_mutation=point,
            allowed_ops=(MEAN, MIN, MAX, WEIGHTED_MEAN, FAR_THRESHOLD, CLOSE_THRESHOLD),
            objective_functions=(objective,),
            minimize_node_count=True,
            reuse_survivor_fitness=False,
            postprocessing_function=np.asarray
            if fit.shape[-1] == 2
            else scale_vector_to_sum_1,
        )
        adapter._seed_population(engine, np.random.default_rng(seed))
        engine.run_iteration()
        first = calls
        engine.run_iteration()
        assert 0 < first < calls - 1
        return calls
    finally:
        adapter._tmp.cleanup()


def archive(checkpoint):
    return (
        checkpoint.calls,
        checkpoint.generation,
        checkpoint.n_trees,
        checkpoint.path_digest,
        checkpoint.fit_tree.structural_key(),
        checkpoint.validation_tree.structural_key(),
        checkpoint.fit_brier,
        checkpoint.validation_brier,
    )


@pytest.mark.parametrize("classes", [2, 4])
@pytest.mark.parametrize(("arm", "point"), [("stock", False), ("retype", True)])
def test_generation_and_exact_call_clocks_follow_original_trajectory(
    classes, arm, point, monkeypatch
):
    from fusionbench.eval.mutation_opportunity import fit_mutation_opportunity

    fit, fit_y, val, val_y = inputs(classes)
    seed = 471
    boundary = native_boundary(fit, fit_y, seed, point)
    budgets = (boundary - 1, boundary, boundary + 1)
    reference = fit_suite_point(
        fit, fit_y, val, val_y, seed=seed, point_mutation=point, budgets=budgets
    )
    assert [reference[b].generation for b in budgets] == [2, 2, 3]
    from okapi import Okapi

    original = Okapi._calculate_fitnesses
    scored = []

    def count_batch(engine, trees=None):
        batch = engine.population if trees is None else trees
        result = original(engine, batch)
        scored.append(len(batch))
        return result

    monkeypatch.setattr(Okapi, "_calculate_fitnesses", count_batch)
    for budget in budgets:
        scored.clear()
        observed = fit_mutation_opportunity(
            fit, fit_y, val, val_y, seed=seed, arm=arm, budget=budget, generations=2
        )
        assert sum(scored) == max(budget, boundary)
        assert set(observed) == {"calls", "generation"}
        for clock, calls in (("calls", budget), ("generation", boundary)):
            checkpoint = observed[clock]
            assert archive(checkpoint) == archive(reference[calls])
            assert isinstance(checkpoint.counts, dict)
            for name in ("fit_tree", "validation_tree"):
                np.testing.assert_array_equal(
                    predict_tree(getattr(checkpoint, name), val),
                    predict_tree(getattr(reference[calls], name), val),
                )
        if budget == boundary:
            before = observed["calls"].counts
            after = observed["generation"].counts
            # The last fitness call precedes the second survival decision.
            assert before["completed_generations"] == 1
            assert after == {**before, "completed_generations": 2}


@pytest.mark.parametrize("arm", ["stock", "identity", "refresh", "retype"])
def test_mutation_and_fitness_counts_follow_the_evaluated_archive(arm):
    from fusionbench.eval.mutation_opportunity import fit_mutation_opportunity

    checkpoints = fit_mutation_opportunity(
        *inputs(2), seed=319, arm=arm, budget=9, generations=3
    )
    initial = checkpoints["calls"]
    initial_reference = fit_suite_point(
        *inputs(2), seed=319, point_mutation=False, budgets=(9,)
    )[9]
    assert archive(initial) == archive(initial_reference)
    assert initial.counts["point_opportunities"] == 0
    assert initial.counts["point_program_changes"] == 0
    assert initial.counts["point_family_changes"] == 0

    final = checkpoints["generation"]
    assert final.generation == 3
    assert final.calls > initial.calls
    for cp in (initial, final):
        c = cp.counts
        assert c["fit_calls"] == cp.calls
        assert c["validation_calls"] == cp.n_trees
        assert c["dedup_removed"] >= 0
        assert sum(cp.family_transitions.values()) == c["point_opportunities"]
        assert (
            c["point_opportunities"]
            >= c["point_program_changes"]
            >= c["point_family_changes"]
            >= 0
        )
    last = final.counts
    if arm == "stock":
        assert last["point_opportunities"] == 0
    else:
        assert last["point_opportunities"] > 0
    if arm in ("stock", "identity", "refresh"):
        assert last["point_family_changes"] == 0
    else:
        assert last["point_family_changes"] == last["point_opportunities"]
    if arm in ("stock", "identity"):
        assert last["point_program_changes"] == 0
    if arm == "refresh":
        assert last["point_program_changes"] <= last["point_original_parametric"]

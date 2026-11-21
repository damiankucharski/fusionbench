import numpy as np
import pytest

from fusionbench.eval.gate_proposal_gp import (
    ProposalPlan,
    probe_and_continue,
    replay_insertion,
)
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


def cube():
    rows = np.arange(64)[:, None]
    bits = ((rows >> np.arange(6)) & 1).astype(float)
    truth = (bits[:, 0] * bits[:, 1] + bits[:, 2] * bits[:, 3]) > 0
    return bits, truth.astype(float)


def pathway_tree(x):
    from okapi.node import MaxNode, MinNode, ValueNode
    from okapi.tree import Tree

    a, b, c, d = [ValueNode(None, x[:, i : i + 1], f"p{i}.npy") for i in range(4)]
    a.add_child(MinNode([b]))
    c.add_child(MinNode([d]))
    a.add_child(MaxNode([c]))
    return Tree(a)


def comparisons(actual, expected):
    assert actual.calls == expected.calls
    assert actual.generation == expected.generation
    assert actual.path_digest == expected.path_digest
    assert actual.n_trees == expected.n_trees
    assert actual.fit_brier == expected.fit_brier
    assert actual.validation_brier == expected.validation_brier
    assert actual.fit_tree.structural_key() == expected.fit_tree.structural_key()
    assert (
        actual.validation_tree.structural_key()
        == expected.validation_tree.structural_key()
    )


def test_completed_boundary_and_noop_are_exactly_the_plain_trajectory(monkeypatch):
    from okapi import Okapi

    x, y = cube()
    total, boundaries, callbacks = [], [], []
    calc, iterate = Okapi._calculate_fitnesses, Okapi.run_iteration

    def observe_calls(engine, trees):
        total.extend(trees)
        return calc(engine, trees)

    def observe_iterations(engine):
        result = iterate(engine)
        boundaries.append(len(total))
        return result

    monkeypatch.setattr(Okapi, "_calculate_fitnesses", observe_calls)
    monkeypatch.setattr(Okapi, "run_iteration", observe_iterations)

    def noop(fit, val, base):
        callbacks.append((base.calls, val.structural_key()))
        assert fit.structural_key() == base.fit_tree.structural_key()
        assert len(total) == base.calls == boundaries[-1]
        assert boundaries[-2] < 1200 <= base.calls
        return ProposalPlan(fit, base.fit_brier, [0, np.int64(1), 17, 1, 300])

    with training_files(x, y) as (paths, gt):
        args = paths, gt, tensors(x), y, 277
        probe = probe_and_continue(*args, noop)
        c = probe.base.calls
        assert len(total) == c + 300
        assert set(probe.plain) == {c, c + 1, c + 17, c + 300}
        comparisons(probe.plain[c + 300], probe.sham)
        total.clear()
        boundaries.clear()
        result = replay_insertion(*args, probe)
        assert len(total) == c + 300
        assert not result.inserted and result.status == "unchanged"
        comparisons(result.checkpoint, probe.sham)
        assert len(callbacks) == 1


def test_fresh_perfect_candidate_enters_ordinary_survival_and_both_archives():
    x, y = cube()

    def submit(_fit, _val, base):
        assert base.fit_brier > 0
        proposal = pathway_tree(x)
        assert np.array_equal(proposal.predict().ravel(), y)
        return ProposalPlan(proposal, 0.0, [0])

    with training_files(x, y) as (paths, gt):
        args = paths, gt, tensors(x), y, 101
        probe = probe_and_continue(*args, submit, initial_calls=1)
        inserted = replay_insertion(*args, probe)
    assert inserted.inserted and inserted.survived and inserted.status == "admitted"
    assert inserted.checkpoint.calls == probe.sham.calls == probe.base.calls + 300
    assert inserted.checkpoint.fit_brier == probe.sham.fit_brier == 0.0
    assert inserted.checkpoint.validation_brier == probe.sham.validation_brier == 0.0
    assert inserted.checkpoint.path_digest != probe.sham.path_digest


def test_rejection_preserves_parent_order_rng_and_sham_archive():
    from okapi.node import MinNode, ValueNode
    from okapi.tree import Tree

    _, y = cube()
    x = np.column_stack([y] * 5 + [1 - y])

    def inferior(_fit, _val, _base):
        root = ValueNode(None, x[:, 5:6], "p5.npy")
        for _ in range(4):
            root.add_child(MinNode([ValueNode(None, x[:, 5:6], "p5.npy")]))
        return ProposalPlan(Tree(root), 1.0, ())

    with training_files(x, y) as (paths, gt):
        args = paths, gt, tensors(x), y, 101
        probe = probe_and_continue(*args, inferior, initial_calls=1, followup_calls=64)
        result = replay_insertion(*args, probe)
    assert not result.inserted and not result.survived and result.status == "rejected"
    comparisons(result.checkpoint, probe.sham)
    assert probe.sham.n_trees == probe.plain[probe.sham.calls].n_trees + 1


def test_duplicate_parent_does_not_reorder_or_rescore_the_population():
    from okapi.node import ValueNode
    from okapi.tree import Tree

    _, y = cube()
    x = np.column_stack([y] * 6)

    def another_parent(_fit, _val, base):
        name = "p0.npy" if base.fit_tree.root.id != "p0.npy" else "p1.npy"
        return ProposalPlan(Tree(ValueNode(None, x[:, :1], name)), 0.0, [])

    with training_files(x, y) as (paths, gt):
        args = paths, gt, tensors(x), y, 101
        probe = probe_and_continue(
            *args, another_parent, initial_calls=1, followup_calls=32
        )
        result = replay_insertion(*args, probe)
    assert (
        result.survived and not result.inserted and result.status == "duplicate_parent"
    )
    comparisons(result.checkpoint, probe.sham)

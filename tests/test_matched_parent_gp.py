from types import SimpleNamespace

import numpy as np
import pytest

from fusionbench.eval.gate_proposal_gp import ProposalPlan, _copy, probe_and_continue
from fusionbench.eval.matched_parent_gp import (
    ARMS,
    OFFERS,
    Admission,
    MatchedPlan,
    _install,
    _rng_digest,
    matched_parent_gp,
)
from fusionbench.eval.native_switch import tensors
from fusionbench.eval.structural_key_audit import training_files


@pytest.fixture(autouse=True)
def native(monkeypatch):
    pytest.importorskip("okapi")
    from okapi.backend.backend import Backend
    from okapi.globals import postprocessing_function

    monkeypatch.setattr(Backend, "_current_backend", Backend._current_backend)
    monkeypatch.setattr(
        postprocessing_function,
        "_postprocessing_function",
        postprocessing_function._postprocessing_function,
    )


def problem():
    x = ((np.arange(64)[:, None] >> np.arange(6)) & 1).astype(float)
    y = ((x[:, 0] * x[:, 1] + x[:, 2] * x[:, 3]) > 0).astype(float)
    return x, y


def pathway(x):
    from okapi.node import MaxNode, MinNode, ValueNode
    from okapi.tree import Tree

    a, b, c, d = [ValueNode(None, x[:, i : i + 1], f"p{i}.npy") for i in range(4)]
    a.add_child(MinNode([b]))
    c.add_child(MinNode([d]))
    a.add_child(MaxNode([c]))
    return Tree(a)


def equal_checkpoints(a, b):
    assert (a.calls, a.generation, a.path_digest) == (
        b.calls,
        b.generation,
        b.path_digest,
    )
    assert (a.n_trees, a.fit_brier, a.validation_brier) == (
        b.n_trees,
        b.fit_brier,
        b.validation_brier,
    )
    assert a.fit_tree.structural_key() == b.fit_tree.structural_key()
    assert a.validation_tree.structural_key() == b.validation_tree.structural_key()


def test_shared_archive_no_admission_matches_independent_stock_path():
    x, y = problem()
    perfect = pathway(x)
    callbacks = []

    def propose(fit, _val, base):
        callbacks.append(base.calls)
        assert base.fit_brier > 0
        unchanged = (fit, base.fit_brier)
        offers = dict.fromkeys(OFFERS, unchanged)
        # This independently admissible arm may enter the shared archive only.
        offers[OFFERS[1]] = (perfect, 0.0)
        return MatchedPlan(offers, [0, 3, 300])

    def noop(fit, _, base):
        return ProposalPlan(fit, base.fit_brier, [3, 300])

    with training_files(x, y) as (paths, gt):
        args = paths, gt, tensors(x), y, 101
        result = matched_parent_gp(*args, propose, initial_calls=1)
        untouched = probe_and_continue(*args, noop, initial_calls=1)
    c = result.base.calls
    assert callbacks == [c]
    for k in (c, c + 3, c + 300):
        equal_checkpoints(result.plain[k], untouched.plain[k])
    assert result.admission.status == "unchanged"
    assert result.admission.marginal[OFFERS[1]] == "admitted"
    assert result.admission.parent_order == tuple(range(6))
    assert result.admission.slot is None and result.admission.displaced is None
    snapshots = [arm[c + 300] for arm in result.arms.values()]
    for arm, cp in zip(ARMS, snapshots, strict=True):
        equal_checkpoints(cp, snapshots[0])
        assert cp.path_digest == result.plain[c + 300].path_digest
        assert cp.fit_brier == result.arms[arm][c].fit_brier == 0
        assert result.states[arm] == result.states["archive_only"]
        assert result.lineage[arm].marked_evaluations == 0


def test_admitted_matched_copies_and_marker_leave_selection_invariant():
    x, y = problem()
    perfect = pathway(x)

    def tracked(tree):
        count = sum(hasattr(n, "test_origin") for n in tree.op_nodes)
        return count, count

    def mark(tree, arm):
        clone = _copy(tree)
        clone.op_nodes[0].test_origin = arm
        return clone

    def occurrences(tree):
        from okapi.interventions import operator_paths

        return tuple(
            SimpleNamespace(path=path)
            for path, n in operator_paths(tree)
            if hasattr(n, "test_origin")
        )

    def plan(fit, _, base, with_marker=False):
        offers = dict.fromkeys(OFFERS[:2], (perfect, 0.0))
        offers.update(dict.fromkeys(OFFERS[2:], (fit, base.fit_brier)))
        return MatchedPlan(
            offers,
            [],
            mark if with_marker else None,
            tracked if with_marker else None,
            occurrences if with_marker else None,
        )

    with training_files(x, y) as (paths, gt):
        args = paths, gt, tensors(x), y, 101
        tagged = matched_parent_gp(
            *args,
            lambda *a: plan(*a, with_marker=True),
            initial_calls=1,
            followup_calls=45,
        )
        control = matched_parent_gp(*args, plan, initial_calls=1, followup_calls=45)
    target = tagged.base.calls + 45
    assert tagged.admission.admitted and tagged.admission.slot is not None
    assert tagged.admission.parent_order[tagged.admission.slot] == -1
    assert set(tagged.admission.parent_order) == {-1, *range(6)} - {
        tagged.admission.displaced
    }
    assert [tagged.admission.marginal[k] for k in OFFERS[:2]] == [
        "admitted",
        "admitted",
    ]
    assert tagged.states[OFFERS[0]].slot_fitness == 1.0
    for arm in ARMS:
        equal_checkpoints(tagged.arms[arm][target], control.arms[arm][target])
        if arm in OFFERS:
            trace = tagged.lineage[arm]
            if arm in OFFERS[:2]:
                assert trace.marked_evaluations > 0
            # A structurally duplicated slot can be removed at the next ordinary
            # deduplication, before the marked parent is evaluated.
            assert (
                trace.marked_occurrences
                >= trace.marked_evaluations
                >= trace.marked_unique
            )
            assert (
                not trace.winner_first_marked
            )  # perfect fit was already in the common archive
        else:
            assert tagged.lineage[arm].marked_evaluations == 0
    equal_checkpoints(tagged.arms[OFFERS[0]][target], tagged.arms[OFFERS[1]][target])
    assert tagged.lineage[OFFERS[0]].winner_ever_marked
    assert tagged.lineage[OFFERS[0]].winner_ever_exact
    assert tagged.lineage[OFFERS[0]].winner_ever_paths == ((0,),)
    assert len(tagged.states["reorder"].parent_fitness) == 6
    assert len(tagged.states["reorder"].parent_mutation_chances) == 6
    assert set(tagged.states["reorder"].parent_keys) == set(
        tagged.states["archive_only"].parent_keys
    )


def test_slot_keeps_duplicate_controls_and_restores_displaced_parent_score_and_rng():
    from okapi.node import ValueNode
    from okapi.tree import Tree

    x, _ = problem()
    parents = [
        Tree(
            ValueNode(None, x[:, i : i + 1], f"p{i}.npy"),
            mutation_chance=0.1 + 0.01 * i,
        )
        for i in range(6)
    ]
    scores = np.arange(1, 7).reshape(-1, 1) / 10
    order = (3, -1, 1, 2, 5, 4)
    admission = Admission("admitted", True, 1, 0, order, {}, "", "")
    after = np.random.RandomState(826).get_state()
    source = _copy(parents[3])
    prepared = {OFFERS[-1]: (source, source.structural_key(), (0.654321, 0.0, source))}
    for arm in ("reorder", OFFERS[-1]):
        gp = SimpleNamespace(
            population=parents.copy(),
            fitnesses=scores.copy(),
            rng=np.random.RandomState(2),
        )
        row = _install(SimpleNamespace(gp=gp), arm, admission, after, prepared, None)
        assert _rng_digest(gp.rng.get_state()) == _rng_digest(after)
        for i, parent in enumerate(order):
            if parent >= 0:
                assert gp.population[i] is parents[parent]
                assert gp.fitnesses[i, 0] == scores[parent, 0]
        if arm == "reorder":
            assert gp.population[1] is parents[0] and row == scores[0, 0]
        else:
            assert gp.population[1] is not parents[3]
            assert (
                gp.population[1].structural_key() == gp.population[0].structural_key()
            )
            assert gp.population[1].mutation_chance == parents[3].mutation_chance
            assert row == gp.fitnesses[1, 0] == 0.346


def test_fork_preserves_native_parent_node_index_order():
    from fusionbench.eval.matched_parent_gp import _parent_copy, _population_digest

    x, _ = problem()
    source = pathway(x)
    source.mutation_chance = 0.37
    # append/replace may leave valid index lists in a non-breadth-first order.
    source.nodes["op_nodes"].reverse()
    source.nodes["value_nodes"].reverse()
    clone = _parent_copy(source)

    def digest(tree):
        gp = SimpleNamespace(population=[tree], fitnesses=np.array([[0.7]]))
        return _population_digest(gp)

    assert source.structural_key() == clone.structural_key()
    assert digest(source) == digest(clone)
    assert digest(source) != digest(_copy(source))
    assert all(
        a is not b
        for a, b in zip(source.root.get_nodes(), clone.root.get_nodes(), strict=True)
    )

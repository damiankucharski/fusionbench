import hashlib
import itertools
import math
import struct

import numpy as np
import pytest

from fusionbench.eval.matching_panel import make_panel, stock_winners

LOW, HIGH = 1 / 64, 63 / 64


def sorted_edge(i, j):
    return (min(i, j), max(i, j))


def is_hit(bits, matching):
    return np.logical_or.reduce([bits[:, i] & bits[:, j] for i, j in matching])


def check_features(block, width, active):
    bits = block.bits
    assert bits.dtype == np.bool_
    assert bits.shape == block.scores.shape
    assert bits.shape[1] == width
    np.testing.assert_array_equal(bits.sum(1), np.full(len(bits), active))
    np.testing.assert_array_equal(block.scores, np.where(bits, HIGH, LOW))
    np.testing.assert_array_equal(
        block.preds, np.stack([1 - block.scores, block.scores], axis=-1)
    )


def check_labels(block, matching, roles, repeats=None):
    a, b, c, d = block.bits[:, roles].T
    logit = 0.2 + 1.1 * a + 0.6 * b - 1.2 * c - 0.7 * d
    expected = {
        "matching": np.where(is_hit(block.bits, matching), HIGH, LOW),
        "logistic": 1 / (1 + np.exp(-logit)),
    }
    shape = (len(block.bits),) if repeats is None else (len(block.bits), repeats)
    assert block.u.shape == shape
    assert np.all((block.u >= 0) & (block.u < 1))
    assert set(block.q) == set(block.y) == set(expected)
    for name, q in expected.items():
        np.testing.assert_allclose(block.q[name], q, rtol=1e-14, atol=0)
        truth = (block.u < q) if repeats is None else (block.u < q[:, None])
        assert block.y[name].shape == shape
        np.testing.assert_array_equal(block.y[name], truth)


@pytest.mark.parametrize("k", [4, 6])
def test_development_laws_weights_and_complete_shift_support(k):
    n = 2 * k
    reference = None
    for design, pi in [("balanced", 0.5), ("uniform", 1 / (n - 1))]:
        panel = make_panel(1981, k, design)
        assert panel.pi == pi
        matching = tuple(tuple(e) for e in panel.matching)
        assert len(matching) == k
        assert sorted(i for edge in matching for i in edge) == list(range(n))
        assert all(len(edge) == 2 for edge in matching)
        roles = tuple(panel.roles)
        assert len(roles) == len(set(roles)) == 4 and set(roles) <= set(range(n))
        if reference is not None:
            assert matching == tuple(tuple(e) for e in reference.matching)
            assert roles == tuple(reference.roles)
            for size, block in panel.test.items():
                np.testing.assert_array_equal(block.bits, reference.test[size].bits)
                np.testing.assert_array_equal(block.u, reference.test[size].u)
        reference = panel
        assert (
            len(panel.fit.bits),
            len(panel.validation.bits),
            len(panel.unlabeled.bits),
        ) == (64, 128, 128)
        for block in (panel.fit, panel.validation, panel.unlabeled):
            check_features(block, n, 2)
        for block in (panel.fit, panel.validation):
            check_labels(block, matching, roles)
        assert not np.array_equal(panel.fit.bits, panel.validation.bits[:64])
        assert not np.array_equal(panel.unlabeled.bits, panel.validation.bits)
        assert not np.array_equal(panel.fit.u, panel.validation.u[:64])

        assert set(panel.test) == {2, 3, 4}
        for active, block in panel.test.items():
            check_features(block, n, active)
            check_labels(block, matching, roles, repeats=64)
            assert len(block.bits) == math.comb(n, active)
            assert {tuple(np.flatnonzero(row)) for row in block.bits} == set(
                itertools.combinations(range(n), active)
            )
            assert not np.array_equal(block.u[:, 0], block.u[:, 1])

        assert set(panel.weights) == {
            "two_balanced",
            "two_uniform",
            "three_uniform",
            "four_uniform",
        }
        for name, active in [
            ("two_uniform", 2),
            ("three_uniform", 3),
            ("four_uniform", 4),
        ]:
            np.testing.assert_allclose(
                panel.weights[name],
                np.full(len(panel.test[active].bits), 1 / math.comb(n, active)),
            )
        two = panel.test[2].bits
        edges = math.comb(n, 2)
        balanced = panel.weights["two_balanced"]
        np.testing.assert_allclose(
            balanced, np.where(is_hit(two, matching), 0.5 / k, 0.5 / (edges - k))
        )
        for name, weight in panel.weights.items():
            assert np.sum(weight) == pytest.approx(1), name
        np.testing.assert_allclose(balanced @ two, np.full(n, 2 / n))


def test_uniform_and_balanced_train_samplers_have_the_declared_edge_masses():
    k = 4
    n = 2 * k
    edges = set(itertools.combinations(range(n), 2))
    role_pair_hits = []
    for design, pi in [("balanced", 0.5), ("uniform", 1 / (n - 1))]:
        counts = np.zeros(len(edges), dtype=int)
        for seed in range(810, 834):
            panel = make_panel(seed, k, design)
            match = {sorted_edge(i, j) for i, j in panel.matching}
            if design == "balanced":
                role_pair_hits.append(sorted_edge(*panel.roles[:2]) in match)
            # Sort within/cross separately for each independently anonymized bank.
            rank = {e: i for i, e in enumerate(sorted(match) + sorted(edges - match))}
            x = np.vstack([panel.fit.bits, panel.validation.bits, panel.unlabeled.bits])
            indices = [rank[tuple(np.flatnonzero(row))] for row in x]
            counts += np.bincount(indices, minlength=len(edges))
        probabilities = np.r_[
            np.full(k, pi / k), np.full(len(edges) - k, (1 - pi) / (len(edges) - k))
        ]
        expected = counts.sum() * probabilities
        assert np.all(np.abs(counts - expected) < 6 * np.sqrt(expected))
    # The logistic roles are not constrained to be the endpoints of matching pairs.
    assert any(role_pair_hits) and not all(role_pair_hits)


def independent_stock(paths, gt, validation, y, seed):
    from okapi import Okapi
    from okapi.operators import (
        CLOSE_THRESHOLD,
        FAR_THRESHOLD,
        MAX,
        MEAN,
        MIN,
        WEIGHTED_MEAN,
    )

    # Complete every ordinary native batch. Record the first 600 individual
    # trees and their raw losses, even if the final batch extends past 600.
    prefix = []
    raw = []
    validation_by_structure = {}

    def objective(p, y):
        loss = np.mean((p.ravel() - y.ravel()) ** 2)
        raw.append(float(loss))
        return 1 - loss

    class Observer(Okapi):
        def _calculate_fitnesses(self, trees=None):
            batch = self.population if trees is None else trees
            raw.clear()
            scores = super()._calculate_fitnesses(batch)
            for tree, loss in zip(batch, raw, strict=True):
                if len(prefix) == 600:
                    break
                key = tree.structural_key()
                prefix.append((key, loss))
                if key not in validation_by_structure:
                    p = tree.do_pred_on_another_tensors(
                        current_tensors=validation
                    ).ravel()
                    validation_by_structure[key] = float(np.mean((p - y) ** 2))
            return scores

    native = Observer(
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
    generation = 0
    while len(prefix) < 600:
        generation += 1
        native.run_iteration()
    best, fit_loss = min(prefix, key=lambda item: item[1])
    selected, _ = min(
        prefix, key=lambda item: (validation_by_structure[item[0]], item[1])
    )
    digest = hashlib.sha256()
    for key, loss in prefix:
        digest.update((repr(key) + "\n").encode() + struct.pack("<d", loss))
    return (
        best,
        selected,
        fit_loss,
        validation_by_structure[selected],
        len(validation_by_structure),
        generation,
        digest.hexdigest(),
    )


def test_stock_reproduces_unmodified_native_search_for_both_probability_layouts(
    monkeypatch,
):
    pytest.importorskip("okapi")
    from okapi import Okapi
    from okapi.backend.backend import Backend
    from okapi.globals import postprocessing_function

    from fusionbench.eval.native_switch import tensors
    from fusionbench.eval.structural_key_audit import training_files

    monkeypatch.setattr(Backend, "_current_backend", Backend._current_backend)
    monkeypatch.setattr(
        postprocessing_function,
        "_postprocessing_function",
        postprocessing_function._postprocessing_function,
    )
    panel = make_panel(207, 4, "balanced")
    train, validation = panel.fit, panel.validation
    fit_y, val_y = train.y["matching"][:16], validation.y["matching"]
    seed = 387
    with training_files(train.scores[:16], fit_y) as (paths, gt):
        expected = independent_stock(paths, gt, tensors(validation.scores), val_y, seed)

    original = Okapi._calculate_fitnesses
    calls = []

    def observe(engine, trees):
        calls.append(len(trees))
        return original(engine, trees)

    monkeypatch.setattr(Okapi, "_calculate_fitnesses", observe)
    for fit, val in [
        (train.scores, validation.scores),
        (train.preds, validation.preds),
    ]:
        calls.clear()
        result = stock_winners(fit[:16], fit_y, val, val_y, seed)
        assert result.calls == sum(calls) == 600
        assert (
            result.fit_tree.structural_key(),
            result.validation_tree.structural_key(),
            result.fit_brier,
            result.validation_brier,
            result.archive_trees,
            result.max_generation,
            result.path_digest,
        ) == expected

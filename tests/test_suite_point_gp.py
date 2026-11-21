"""Regression tests against complete, untruncated native OKAPI fitness batches."""

import hashlib
import struct

import numpy as np
import pytest

from fusionbench.eval.metrics import normalize_proba
from fusionbench.eval.suite_point_gp import fit_suite_point, predict_tree
from fusionbench.methods.okapi import OKAPIMethod

pytest.importorskip("okapi")


@pytest.fixture(autouse=True)
def restore_process_settings(monkeypatch):
    from okapi.backend.backend import Backend
    from okapi.globals import postprocessing_function

    monkeypatch.setattr(Backend, "_current_backend", Backend._current_backend)
    monkeypatch.setattr(
        postprocessing_function,
        "_postprocessing_function",
        postprocessing_function._postprocessing_function,
    )


def problem(classes):
    rng = np.random.default_rng([830, classes])
    fit = rng.dirichlet(np.ones(classes), size=(27, 7))
    val = rng.dirichlet(np.ones(classes), size=(17, 7))
    y = rng.permutation(np.arange(len(fit)) % classes)
    # The highest class is deliberately absent on validation.
    vy = rng.permutation(np.arange(len(val)) % (classes - 1))
    return fit, y, val, vy


def loss(p, labels):
    if p.shape[1] == 1:
        return float(np.mean((p[:, 0] - labels) ** 2))
    onehot = np.eye(p.shape[1])[labels]
    return 0.5 * float(np.mean(np.sum((p - onehot) ** 2, axis=1)))


def native_prefixes(tmp_path, fit, y, val, vy, seed, point, budgets):
    """Observe complete stock batches. Snapshots contain prefixes of those batches."""
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

    binary = fit.shape[-1] == 2
    reference, batch_ends, archive = {}, [], {}
    calls = generation = 0
    digest = hashlib.sha256()
    batch_losses = []

    def objective(preds, gt):
        score = loss(preds, gt.ravel())
        batch_losses.append(score)
        return 1.0 - score

    class Observer(Okapi):
        def _calculate_fitnesses(self, trees=None):
            nonlocal calls
            batch = self.population if trees is None else trees
            batch_losses.clear()
            result = super()._calculate_fitnesses(batch)
            for tree, fit_loss in zip(batch, batch_losses, strict=True):
                calls += 1
                key = tree.structural_key()
                digest.update((repr(key) + "\n").encode() + struct.pack("<d", fit_loss))
                if key not in archive:
                    pred = tree.do_pred_on_another_tensors(
                        preds_directory=validation_path
                    )
                    val_loss = loss(pred, vy)
                    canonical = (
                        np.column_stack((1 - pred[:, 0], pred[:, 0]))
                        if binary
                        else normalize_proba(pred)
                    )
                    archive[key] = fit_loss, val_loss, canonical
                if calls in budgets:
                    a = min(archive, key=lambda k: archive[k][0])
                    b = min(archive, key=lambda k: (archive[k][1], archive[k][0]))
                    reference[calls] = (
                        a,
                        b,
                        archive[a][0],
                        archive[b][1],
                        len(archive),
                        generation,
                        digest.hexdigest(),
                        archive[a][2],
                        archive[b][2],
                    )
            batch_ends.append(calls)
            return result

    adapter = OKAPIMethod()
    try:
        fit_path = adapter._write_split(fit, "fit")
        validation_path = adapter._write_split(val, "validation")
        gt_path = tmp_path / "labels.npy"
        np.save(gt_path, y)
        pop = min(fit.shape[1], 30)
        gp = Observer(
            fit_path,
            gt_path,
            pop,
            3,
            max(2, min(5, pop)),
            backend="numpy",
            seed=seed,
            objective_functions=(objective,),
            minimize_node_count=True,
            reuse_survivor_fitness=False,
            postprocessing_function=np.asarray if binary else scale_vector_to_sum_1,
            allowed_ops=(MEAN, MIN, MAX, WEIGHTED_MEAN, FAR_THRESHOLD, CLOSE_THRESHOLD),
            operator_point_mutation=point,
        )
        adapter._seed_population(gp, np.random.default_rng(seed))
        while calls < max(budgets):
            generation += 1
            gp.run_iteration()
    finally:
        adapter._tmp.cleanup()
    return reference, batch_ends


@pytest.mark.parametrize("classes", [2, 4, 5])
@pytest.mark.parametrize("point", [False, True])
def test_archive_matches_full_native_batches(tmp_path, monkeypatch, classes, point):
    from okapi import Okapi

    fit, y, val, vy = problem(classes)
    budgets = (3, 21, 49, 213)
    seed = 225
    reference, boundaries = native_prefixes(
        tmp_path, fit, y, val, vy, seed, point, budgets
    )
    assert boundaries[0] == 7 + 30
    assert 21 < boundaries[0] < 49 < boundaries[1]
    original = Okapi._calculate_fitnesses
    visits = []

    def count_calls(gp, trees):
        visits.append(len(trees))
        return original(gp, trees)

    monkeypatch.setattr(Okapi, "_calculate_fitnesses", count_calls)
    selected = fit_suite_point(
        fit, y, val, vy, seed=seed, point_mutation=point, budgets=budgets
    )
    assert sum(visits) == max(budgets)
    assert set(selected) == set(budgets)
    last_time = 0
    for calls in budgets:
        cp = selected[calls]
        assert cp.calls == calls and cp.val_tree is cp.validation_tree
        assert (
            cp.fit_tree.structural_key(),
            cp.val_tree.structural_key(),
            cp.fit_brier,
            cp.validation_brier,
            cp.n_trees,
            cp.generation,
            cp.path_digest,
        ) == reference[calls][:7]
        assert cp.elapsed_seconds >= last_time
        last_time = cp.elapsed_seconds
        for tree, expected_pred in zip(
            (cp.fit_tree, cp.val_tree), reference[calls][7:], strict=True
        ):
            np.testing.assert_allclose(
                predict_tree(tree, val), expected_pred, rtol=0, atol=2e-15
            )


def test_binary_shapes_and_interleaved_task_evaluation():
    b, by, v, vy = problem(2)
    c, cy, w, wy = problem(4)
    opts = {"seed": 243, "point_mutation": True, "budgets": (31,)}
    outputs = []
    for x, z in ((b, v), (b[:, :, 1], v[:, :, 1]), (b[:, :, 1:2], v[:, :, 1:2])):
        cp = fit_suite_point(x, by, z, vy, **opts)[31]
        outputs.append((cp, predict_tree(cp.fit_tree, z)))
    first, first_probs = outputs[0]
    for cp, p in outputs[1:]:
        assert cp.path_digest == first.path_digest
        np.testing.assert_array_equal(first_probs, p)
    multiclass = fit_suite_point(c, cy, w, wy, **opts)[31]
    mult_probs = predict_tree(multiclass.fit_tree, w)
    for z in (v, v[:, :, 1], v[:, :, 1:2]):
        np.testing.assert_array_equal(predict_tree(first.fit_tree, z), first_probs)
        np.testing.assert_array_equal(predict_tree(multiclass.fit_tree, w), mult_probs)


def test_min_with_disjoint_multiclass_support_recovers_uniform_mass():
    from okapi.node import MinNode, ValueNode
    from okapi.tree import Tree

    # One multiclass row has disjoint supports. The second shares class 1 only.
    multiclass = np.array(
        [
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            [[0.6, 0.4, 0.0, 0.0], [0.0, 0.7, 0.3, 0.0]],
        ]
    )
    binary = np.array([[0.3, 0.8], [0.7, 0.4]])
    root = ValueNode(None, multiclass[:, 0, :], "m0.npy")
    root.add_child(MinNode([ValueNode(None, multiclass[:, 1, :], "m1.npy")]))
    tree = Tree.create_tree_from_root(root)
    for _ in range(2):
        p = predict_tree(tree, multiclass)
        np.testing.assert_array_equal(p[0], np.full(4, 0.25))
        np.testing.assert_allclose(p[1], [0, 1, 0, 0], atol=4e-12)
        np.testing.assert_allclose(p.sum(1), 1, atol=2e-15)
        np.testing.assert_allclose(predict_tree(tree, binary), [[0.7, 0.3], [0.6, 0.4]])

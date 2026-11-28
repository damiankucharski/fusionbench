"""Round one changes eligibility for validation selection on one frozen GP archive."""

import hashlib
from dataclasses import dataclass

import numpy as np

from fusionbench.eval.suite_point_gp import _bind, _full, _loss, _targets, predict_tree
from fusionbench.methods.native_pattern import (
    RegularizedNativePatternOKAPI,
    _brier_skill,
)
from fusionbench.methods.okapi import _task_postprocessing


@dataclass
class FrozenChoice:
    tree_: object
    validation_tree: object
    validation_brier_: float

    def predict(self, x):
        return predict_tree(self.tree_, x)

    def validation_predict(self, x):
        """Forecast with the discovery-trained tree evaluated on the validation rows."""
        return predict_tree(self.validation_tree, x)


def fit_variants(x, y):
    """Return three fitted deployment rules and their fit-only, serializable diagnostics."""
    x = _full(x)
    n, _, classes = x.shape
    labels, _ = _targets(y, n, classes)
    order = np.random.default_rng([0, 51]).permutation(n)
    d, v = np.split(order, [7 * n // 10])
    _, discovery_target = _targets(labels[d], len(d), classes)
    _, validation_target = _targets(labels[v], len(v), classes)
    task = "binary" if classes == 2 else "multiclass"
    archive = []

    def observe(tree, pred):
        archive.append((tree.copy(), _loss(pred, discovery_target)))

    trained = RegularizedNativePatternOKAPI(seed=0, split_seed=0)
    trained.fit(x, labels, task=task, n_classes=classes, on_gp_candidate=observe)
    np.testing.assert_array_equal(v, trained.validation_indices_)
    assert len(archive) == trained.counts_["gp_unique_programs"]
    assert len({tree.structural_key() for tree, _ in archive}) == len(archive)

    # Bound the lists from discovery data alone; structural ties retain first visit.
    indices = range(len(archive))
    discovery_order = sorted(indices, key=lambda i: archive[i][1])
    menus = {
        "regularized": tuple(indices),
        "gp_top1": discovery_order[:1],
        "gp_top8": discovery_order[:8],
    }
    # Match seed_geometry_standard.remember's raw native validation scores, which
    # can differ at floating precision from canonical multiclass renormalization.
    validation_inputs = _bind(x[v])
    with _task_postprocessing(task):
        val_losses = [
            _loss(
                tree.do_pred_on_another_tensors(current_tensors=validation_inputs),
                validation_target,
            )
            for tree, _ in archive
        ]

    fitted, choices = {}, {}
    target = np.eye(classes)[labels[v]]
    for name, ids in menus.items():
        i = min(ids, key=lambda i: (val_losses[i], archive[i][1], i))
        gp = archive[i][0]
        discovery_trees = {**trained.discovery_trees_, "gp": gp}
        predictions = {k: predict_tree(t, x[v]) for k, t in discovery_trees.items()}
        family_scores = {
            k: float(_brier_skill(p, labels[v])) for k, p in predictions.items()
        }
        family = max(family_scores, key=family_scores.get)
        if name == "regularized":
            assert gp.structural_key() == trained.trees_["gp"].structural_key()
            assert (
                family == trained.chosen_
                and family_scores == trained.counts_["validation_scores"]
            )
        tree = gp if family == "gp" else trained.trees_[family]
        validation_brier = float(
            np.mean(np.sum((predictions[family] - target) ** 2, axis=1))
        )
        fitted[name] = FrozenChoice(tree, discovery_trees[family], validation_brier)
        choices[name] = {
            "family": family,
            "validation_brier": validation_brier,
            "eligible_gp": len(ids),
            "gp_visit_index": int(i),
            "gp_discovery_rank": discovery_order.index(i) + 1,
            "gp_hash": hashlib.sha256(repr(gp.structural_key()).encode()).hexdigest(),
            "gp_validation_native_half_brier": float(val_losses[i]),
            "gp_discovery_native_half_brier": float(archive[i][1]),
        }
    return fitted, {
        "archive_size": len(archive),
        "gp_calls": int(trained.counts_["score_calls"]["gp_discovery"]),
        "native_path": trained.counts_["gp_path_digest"],
        "choices": choices,
    }

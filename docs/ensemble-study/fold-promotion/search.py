"""Fit the two native GP paths and audit their frozen deployment decisions."""

import hashlib
from pathlib import Path
from runpy import run_path

import numpy as np

from fusionbench.eval.suite_point_gp import _full, _targets, predict_tree
from fusionbench.methods.native_pattern import _full_probability
from fusionbench.methods.okapi import OKAPIMethod, _task_postprocessing

HERE = Path(__file__).resolve().parent
make_founder = run_path(str(HERE.parent / "seeded-checkpoint" / "native_seed.py"))[
    "make_founder"
]
GENERATIONS = 16


def digest(key):
    return hashlib.sha256(repr(key).encode()).hexdigest()


def array_digest(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _losses(p, target):
    return np.sum((p - target) ** 2, axis=1)


def fit_search(fit_x, fit_y, *, groups, data_seed, search_seed, path):
    """Choose a fixed anchor and every deployment using only observations on the fit split."""
    from okapi import Okapi
    from okapi.fold_gain import FoldGain
    from okapi.operators import (
        CLOSE_THRESHOLD,
        FAR_THRESHOLD,
        MAX,
        MEAN,
        MIN,
        WEIGHTED_MEAN,
    )

    if path not in ("fold", "pooled"):
        raise ValueError("Choose pooled or fold search")
    x = _full(fit_x)
    n, m, c = x.shape
    if m < 2:
        raise ValueError("Two or more providers are required for native tournaments")
    labels, _ = _targets(fit_y, n, c)
    target = np.eye(c)[labels]
    archive = {}
    anchors = []
    criterion = None
    calls = proposals = 0
    history = hashlib.sha256()
    snapshots = []
    generation = 0
    group_seed = int(data_seed) + 3_000_000

    class Observed(Okapi):
        def _calculate_fitnesses(self, trees=None):
            nonlocal calls, criterion, anchors
            trees = self.population if trees is None else trees
            losses = [
                _losses(_full_probability(t.predict(clear_cache=True)), target)
                for t in trees
            ]
            if criterion is None:
                # Select the anchor from exactly the founders scored by this same
                # native batch. Their predictions are computed only once.
                anchors = [t.structural_key() for t in trees]
                best = min(
                    range(len(trees)),
                    key=lambda i: (losses[i].mean(), trees[i].nodes_count),
                )
                self.anchor_key = anchors[best]
                criterion = FoldGain(losses[best], groups, group_seed=group_seed)
            scores = []
            for tree, row_loss in zip(trees, losses, strict=True):
                key = tree.structural_key()
                pooled_brier = float(row_loss.mean())
                result = criterion.score(row_loss)
                archive.setdefault(
                    key,
                    {
                        "tree": tree,
                        "pooled_brier": pooled_brier,
                        "robust_gain": result.robust_gain,
                        "fold_gains": tuple(result.fold_gains),
                        "generation": generation,
                    },
                )
                score = (1 - 0.5 * pooled_brier) if path == "pooled" else result.fitness
                scores.append(score)
                # Encodes every ordered scoring visit, including repeated parents.
                history.update((repr(key) + "\n").encode())
            calls += len(trees)
            return np.asarray(scores, dtype=float).reshape(-1, 1)

        def _mutate_additional_population(self):
            nonlocal proposals
            changed = super()._mutate_additional_population()
            proposals += len(self.additional_population)
            return changed

    def snapshot():
        keys = tuple(t.structural_key() for t in gp.population)
        snapshots.append(
            {
                "generation": generation,
                "keys": keys,
                "active_digest": digest(keys),
                "path_digest": history.hexdigest(),
                "objective_evaluations": calls,
                "offspring_proposals": proposals,
                "unique_evaluated": len(archive),
                "best_fit_brier": min(archive[k]["pooled_brier"] for k in keys),
            }
        )

    adapter = OKAPIMethod(seed=search_seed)
    try:
        directory = adapter._write_split(x, "fit")
        gt = Path(adapter._tmp.name) / "gt.npy"
        np.save(gt, labels)
        with _task_postprocessing("binary" if c == 2 else "multiclass"):
            gp = Observed(
                directory,
                gt,
                min(m, 8),
                2,
                2,
                backend="numpy",
                seed=search_seed,
                minimize_node_count=True,
                max_depth=8,
                max_nodes=max(31, m + 1),
                allowed_ops=(
                    MEAN,
                    MIN,
                    MAX,
                    WEIGHTED_MEAN,
                    FAR_THRESHOLD,
                    CLOSE_THRESHOLD,
                ),
            )
            # All search objectives are supplied together by the native batch hook.
            mean = make_founder(gp.train_tensors, "seeded")
            gp.population.append(mean)
            gp.fitnesses = gp._selection_fitnesses(gp.population)
            anchor = gp.anchor_key
            snapshot()
            for generation in range(1, GENERATIONS + 1):
                gp.run_iteration()
                snapshot()
            final = snapshots[-1]["keys"]
            # Tie-break identically to the pooled seeded study for its original deployment.
            pooled_final = min(
                final,
                key=lambda k: (
                    archive[k]["pooled_brier"],
                    archive[k]["tree"].nodes_count,
                ),
            )
            candidates = tuple(dict.fromkeys((anchor, *final)))

            def deploy(key, rule):
                risk = (
                    archive[key]["pooled_brier"]
                    if rule == "pooled"
                    else -archive[key]["robust_gain"]
                )
                return risk, key != anchor, archive[key]["tree"].nodes_count

            choices = {
                f"{path}_{rule}": min(candidates, key=lambda k: deploy(k, rule))
                for rule in ("pooled", "fold")
            }
            if path == "pooled":
                choices = {"pooled_final": pooled_final, **choices, "anchor": anchor}
    finally:
        adapter._tmp.cleanup()
    return {
        "path": path,
        "archive": archive,
        "anchor": anchor,
        "choices": choices,
        "snapshots": snapshots,
        "founder_digest": digest(tuple(anchors)),
        "mean_key": mean.structural_key(),
        "fold_digest": array_digest(criterion.fold_ids),
        "fold_counts": np.bincount(criterion.fold_ids, minlength=4).tolist(),
        "group_seed": group_seed,
        "fit_x": x,
    }


def score_search(fitted, audit_x, audit_y, *, audit_q=None, signal_provider=None):
    """Score the frozen rules. Audit inputs and outcomes were excluded from every search choice."""
    test = _full(audit_x)
    train = fitted["fit_x"]
    c = test.shape[2]
    if test.shape[1:] != train.shape[1:]:
        raise ValueError("Fit and audit must have matching provider and class axes")
    labels, _ = _targets(audit_y, len(test), c)
    target = np.eye(c)[labels]
    q = None
    if audit_q is not None:
        q = np.asarray(audit_q, dtype=float)
        if q.ndim == 1 and c == 2:
            q = np.column_stack((1 - q, q))
        if (
            q.shape != target.shape
            or not np.allclose(q.sum(axis=1), 1.0)
            or not np.all((q >= 0) & (q <= 1))
        ):
            raise ValueError("audit_q must contain a canonical posterior per row")
    chosen = fitted["choices"]
    anchor, archive = fitted["anchor"], fitted["archive"]
    path = fitted["path"]
    means = [arr.mean(axis=1) for arr in (train, test)]
    output = {}
    for key, rec in archive.items():
        p = predict_tree(rec["tree"], test)
        same_mean = np.max(np.abs(p - means[1])) <= 1e-12
        same_signal = (
            signal_provider is not None
            and np.max(np.abs(p - test[:, signal_provider])) <= 1e-12
        )
        if same_mean or same_signal:
            fit_p = predict_tree(rec["tree"], train)
            same_mean = same_mean and np.max(np.abs(fit_p - means[0])) <= 1e-12
            same_signal = (
                same_signal
                and np.max(np.abs(fit_p - train[:, signal_provider])) <= 1e-12
            )
        output[key] = {
            "audit_brier": float(_losses(p, target).mean()),
            "audit_excess": float(_losses(p, q).mean()) if q is not None else None,
            "selected_is_mean": bool(same_mean),
            "selected_is_signal": bool(same_signal)
            if signal_provider is not None
            else None,
        }
    anchor_tree = archive[anchor]["tree"]
    common = {
        "path": path,
        "anchor_digest": digest(anchor),
        "fold_digest": fitted["fold_digest"],
        "founder_digest": fitted["founder_digest"],
        "group_seed": fitted["group_seed"],
        "anchor_is_mean_founder": anchor == fitted["mean_key"],
        "anchor_provider_id": anchor_tree.unique_value_node_ids[0]
        if anchor_tree.nodes_count == 1
        else None,
        "anchor_is_mean": output[anchor]["selected_is_mean"],
        "anchor_is_signal": output[anchor]["selected_is_signal"],
        "anchor_fit_brier": archive[anchor]["pooled_brier"],
    }
    trace = []
    for snap in fitted["snapshots"]:
        keys = snap["keys"]
        row = {k: v for k, v in snap.items() if k != "keys"}
        trace.append(
            {
                **common,
                **row,
                "count": len(keys),
                "anchor_active": anchor in keys,
                "mean_active": any(output[k]["selected_is_mean"] for k in keys),
                "signal_active": any(output[k]["selected_is_signal"] for k in keys)
                if signal_provider is not None
                else None,
                "best_active_audit_brier": min(output[k]["audit_brier"] for k in keys),
                "best_active_audit_excess": min(output[k]["audit_excess"] for k in keys)
                if q is not None
                else None,
            }
        )
    final = fitted["snapshots"][-1]
    costs = {
        k: final[k]
        for k in (
            "objective_evaluations",
            "offspring_proposals",
            "unique_evaluated",
            "path_digest",
        )
    }
    scores = []
    for method, key in chosen.items():
        rec = archive[key]
        scores.append(
            {
                **common,
                **costs,
                **output[key],
                "method": method,
                "selected_digest": digest(key),
                "selected_is_anchor": key == anchor,
                "fit_brier": rec["pooled_brier"],
                "pooled_gain": archive[anchor]["pooled_brier"] - rec["pooled_brier"],
                "robust_gain": rec["robust_gain"],
                "unique_providers": len(rec["tree"].unique_value_node_ids),
                "selected_nodes": rec["tree"].nodes_count,
                **{f"fold_gain_{i}": gain for i, gain in enumerate(rec["fold_gains"])},
            }
        )
    return scores, trace

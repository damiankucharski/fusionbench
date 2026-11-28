"""Fit native GP paths and audit their frozen deployment decisions."""

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


def joint_finalists(archive, final, anchor, fold_winner):
    """Return the veto choice and highest fold-ranked eligible survivor, in that order.

    The frozen anchor is always available. An eligible tree strictly improves on
    it by both unrounded gains. Equal fold gains prefer fewer nodes, then the
    native survivor order. The veto considers only the ordinary fold winner.
    """
    veto = fold_winner if archive[fold_winner]["supports_replacement"] else anchor
    eligible = (k for k in final if archive[k]["supports_replacement"])
    reranked = min(
        eligible,
        key=lambda k: (-archive[k]["robust_gain"], archive[k]["tree"].nodes_count),
        default=anchor,
    )
    return veto, reranked


def fit_search(
    fit_x, fit_y, *, groups, data_seed, search_seed, path, joint_study=False
):
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

    paths = ("fold", "pooled", "joint") if joint_study else ("fold", "pooled")
    if path not in paths:
        raise ValueError(f"Choose one of {paths} for this study")
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
                extra = (
                    {
                        "pooled_gain": result.pooled_gain,
                        "joint_gain": result.joint_gain,
                        "supports_replacement": result.supports_replacement,
                    }
                    if joint_study
                    else {}
                )
                archive.setdefault(
                    key,
                    {
                        "tree": tree,
                        "pooled_brier": pooled_brier,
                        "robust_gain": result.robust_gain,
                        "fold_gains": tuple(result.fold_gains),
                        "generation": generation,
                        **extra,
                    },
                )
                if path == "pooled":
                    score = 1 - 0.5 * pooled_brier
                elif path == "fold":
                    score = result.fitness
                else:
                    score = criterion.joint_fitness(result)
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
            if joint_study:
                choices[f"{path}_veto"], choices[f"{path}_joint"] = joint_finalists(
                    archive, final, anchor, choices[f"{path}_fold"]
                )
            if path == "pooled":
                choices = {"pooled_final": pooled_final, **choices, "anchor": anchor}
    finally:
        adapter._tmp.cleanup()
    return {
        **({"joint_study": True} if joint_study else {}),
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


def _posterior(q, shape, name):
    if q is None:
        return None
    q = np.asarray(q, dtype=float)
    if q.ndim == 1 and shape[1] == 2:
        q = np.column_stack((1 - q, q))
    if (
        q.shape != shape
        or not np.allclose(q.sum(axis=1), 1.0)
        or not np.all((q >= 0) & (q <= 1))
    ):
        raise ValueError(f"{name} must contain a canonical posterior per row")
    return q


def score_search(
    fitted, audit_x, audit_y, *, audit_q=None, signal_provider=None, fit_q=None
):
    """Score the frozen rules. Audit inputs and outcomes were excluded from every search choice."""
    test = _full(audit_x)
    train = fitted["fit_x"]
    c = test.shape[2]
    if test.shape[1:] != train.shape[1:]:
        raise ValueError("Fit and audit must have matching provider and class axes")
    labels, _ = _targets(audit_y, len(test), c)
    target = np.eye(c)[labels]
    q = _posterior(audit_q, target.shape, "audit_q")
    joint_study = fitted.get("joint_study", False)
    if fit_q is not None and (not joint_study or q is None):
        raise ValueError("fit_q requires a joint study and a corresponding audit_q")
    fit_q = _posterior(fit_q, (len(train), c), "fit_q")
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
        exact = fit_q is not None and np.max(np.abs(p - q)) <= 1e-12
        if same_mean or same_signal or exact:
            fit_p = predict_tree(rec["tree"], train)
            exact = exact and np.max(np.abs(fit_p - fit_q)) <= 1e-12
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
        if joint_study:
            output[key]["selected_is_exact"] = (
                bool(exact) if fit_q is not None else None
            )
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
    if joint_study:
        common["anchor_is_exact"] = output[anchor]["selected_is_exact"]
    trace = []
    for snap in fitted["snapshots"]:
        keys = snap["keys"]
        row = {k: v for k, v in snap.items() if k != "keys"}
        diagnostics = {}
        if joint_study:
            seen = [
                k
                for k, record in archive.items()
                if record["generation"] <= snap["generation"]
            ]
            n_active = sum(bool(output[k]["selected_is_exact"]) for k in set(keys))
            n_seen = sum(bool(output[k]["selected_is_exact"]) for k in seen)
            diagnostics = {
                "exact_active": bool(n_active) if fit_q is not None else None,
                "n_exact_active": n_active if fit_q is not None else None,
                "exact_ever_visited": bool(n_seen) if fit_q is not None else None,
                "n_exact_visited": n_seen if fit_q is not None else None,
                "best_ever_audit_brier": min(output[k]["audit_brier"] for k in seen),
                "best_ever_audit_excess": min(output[k]["audit_excess"] for k in seen)
                if q is not None
                else None,
            }
        trace.append(
            {
                **common,
                **row,
                **diagnostics,
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
    if joint_study:
        terminal = trace[-1]
        costs.update(
            {
                "exact_in_final": terminal["exact_active"],
                "n_exact_final": terminal["n_exact_active"],
                "exact_ever_visited": terminal["exact_ever_visited"],
                "n_exact_visited": terminal["n_exact_visited"],
                "best_final_audit_excess": terminal["best_active_audit_excess"],
                "best_ever_audit_excess": terminal["best_ever_audit_excess"],
                "best_final_audit_brier": terminal["best_active_audit_brier"],
                "best_ever_audit_brier": terminal["best_ever_audit_brier"],
            }
        )
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
                "pooled_gain": rec["pooled_gain"]
                if joint_study
                else archive[anchor]["pooled_brier"] - rec["pooled_brier"],
                **(
                    {
                        "joint_gain": rec["joint_gain"],
                        "supports_replacement": rec["supports_replacement"],
                    }
                    if joint_study
                    else {}
                ),
                "robust_gain": rec["robust_gain"],
                "unique_providers": len(rec["tree"].unique_value_node_ids),
                "selected_nodes": rec["tree"].nodes_count,
                **{f"fold_gain_{i}": gain for i, gain in enumerate(rec["fold_gains"])},
            }
        )
    return scores, trace

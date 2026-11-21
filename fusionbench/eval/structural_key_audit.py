"""Joint FusionBench/OKAPI structural-key audit on continuous MIN/MAX trees."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

from fusionbench.eval.lattice_brier_oracle import fit_lattice_brier
from fusionbench.eval.search_continuous import _split, brier, structural_key
from fusionbench.methods.averaging import TopN
from fusionbench.methods.stacking import StackingLogit

BUDGETS = (300, 1200)
GP_SEEDS = (151, 157, 163, 167)
DOMAINS = ("fit", "validation", "iid", "shift")


def exact_brier(pred, labels):
    return sum(
        (Fraction.from_float(float(p)) - int(y)) ** 2
        for p, y in zip(pred, labels, strict=True)
    ) / len(labels)


def probability_digest(arrays):
    h = hashlib.sha256()
    for p in arrays:
        h.update(np.ascontiguousarray(p, dtype="<f8").tobytes())
    return h.hexdigest()


class MaskAudit:
    """Observe both masks on the actual join, then return the assigned policy."""

    def __init__(self, policy, fit_tensors):
        from okapi.utils import first_uniques_mask

        self.legacy = lambda trees: first_uniques_mask(
            np.array([repr(t) for t in trees])
        )
        self.fixed = import_module("okapi.okapi")._unique_tree_mask
        self.policy = policy
        self.tensors = fit_tensors
        masks = np.arange(1 << len(fit_tensors), dtype=np.int64)
        self.corners = {
            name: ((masks >> j) & 1).astype(float)[:, None]
            for j, name in enumerate(fit_tensors)
        }
        self.joins = self.proposals = self.suppressed = self.changed = 0
        self.collisions = []

    def __call__(self, trees):
        old = self.legacy(trees)
        new = self.fixed(trees)
        self.joins += 1
        self.proposals += len(trees)
        assert not np.any(np.asarray(old) & ~np.asarray(new))
        hidden = np.flatnonzero(np.asarray(new) & ~np.asarray(old))
        if len(hidden):
            first = {repr(trees[i]): i for i in np.flatnonzero(old)}
            for i in hidden:
                j = first[repr(trees[i])]
                a, b = [
                    trees[k].do_pred_on_another_tensors(current_tensors=self.tensors)
                    for k in (j, i)
                ]
                changed = int(np.count_nonzero(a != b))
                corner_predictions = [
                    trees[k].do_pred_on_another_tensors(current_tensors=self.corners)
                    for k in (j, i)
                ]
                different_corners = int(
                    np.count_nonzero(corner_predictions[0] != corner_predictions[1])
                )
                self.suppressed += 1
                self.changed += int(changed > 0)
                self.collisions.append(
                    {
                        "join": self.joins,
                        "index": int(i),
                        "first_index": int(j),
                        "changed_fit_rows": changed,
                        "different_boolean_corners": different_corners,
                        "retained_tree": repr(structural_key(trees[j].root)),
                        "suppressed_tree": repr(structural_key(trees[i].root)),
                    }
                )
        return new if self.policy == "fixed" else old


class EndOfBudget(Exception):
    pass


@contextmanager
def training_files(x, y):
    with TemporaryDirectory() as directory:
        base = Path(directory)
        paths = [base / f"p{i}.npy" for i in range(x.shape[1])]
        for i, path in enumerate(paths):
            np.save(path, x[:, i : i + 1])
        gt = base / "labels.npy"
        np.save(gt, y)
        yield paths, gt


def fit_gp(paths, gt, gp_seed, policy, tensors, labels):
    from okapi import Okapi
    from okapi.operators import MAX, MIN

    audit = MaskAudit(policy, tensors[0])
    archive = {}
    checkpoints = []
    path_digest = hashlib.sha256()
    calls = 0
    fit_loss = None

    def objective(pred, y):
        nonlocal fit_loss
        fit_loss = brier(pred.ravel(), y.ravel())
        return 1 - fit_loss

    class ObservedGP(Okapi):
        def _calculate_fitnesses(self, trees=None):
            nonlocal calls
            scores = []
            for tree in self.population if trees is None else trees:
                score = super()._calculate_fitnesses([tree])[0]
                calls += 1
                key = structural_key(tree.root)
                path_digest.update((repr(key) + "\n").encode())
                if key not in archive:
                    ref = tree.copy()
                    p = ref.do_pred_on_another_tensors(
                        current_tensors=tensors[1]
                    ).ravel()
                    archive[key] = (fit_loss, brier(p, labels[1]), ref)
                if calls in BUDGETS:
                    fit = min(archive.values(), key=lambda entry: entry[0])[2]
                    val = min(archive.values(), key=lambda entry: (entry[1], entry[0]))[
                        2
                    ]
                    checkpoints.append(
                        (
                            {
                                "budget": calls,
                                "path_sha256": path_digest.hexdigest(),
                                "archive_trees": len(archive),
                                "observed_joins": audit.joins,
                                "observed_proposals": audit.proposals,
                                "repr_collisions": audit.suppressed,
                                "different_on_fit": audit.changed,
                                "elapsed_seconds": time.perf_counter() - start,
                            },
                            fit,
                            val,
                        )
                    )
                    if calls == BUDGETS[-1]:
                        raise EndOfBudget
                scores.append(score)
            return np.stack(scores)

    start = time.perf_counter()
    gp = ObservedGP(
        paths,
        gt,
        len(paths),
        2,
        2,
        backend="numpy",
        seed=gp_seed,
        allowed_ops=(MIN, MAX),
        postprocessing_function=np.asarray,
        objective_functions=(objective,),
        reuse_survivor_fitness=False,
    )
    core = import_module("okapi.okapi")
    with patch.object(core, "_unique_tree_mask", audit):
        try:
            while True:
                gp.run_iteration()
        except EndOfBudget:
            pass
    assert calls == BUDGETS[-1] and len(checkpoints) == len(BUDGETS)
    return checkpoints, audit.collisions


@dataclass
class ReferencePanel:
    data_seed: int
    train: np.ndarray
    labels: list
    tensors: list
    baseline: list
    minimum_loss: Fraction

    def scores(self, predictions):
        losses = {
            name: brier(p, y)
            for name, p, y in zip(DOMAINS, predictions, self.labels, strict=True)
        }
        return dict(
            data_seed=self.data_seed,
            **losses,
            prediction_sha256=probability_digest(predictions),
        )

    def champion_rows(self, snapshots, gp_seed, policy):
        for metadata, fit, val in snapshots:
            for name, tree in (("fit", fit), ("validation", val)):
                probs = [
                    tree.do_pred_on_another_tensors(current_tensors=t).ravel()
                    for t in self.tensors
                ]
                fit_gap = exact_brier(probs[0], self.labels[0]) - self.minimum_loss
                assert fit_gap >= 0
                yield dict(
                    **self.scores(probs),
                    gp_seed=gp_seed,
                    policy=policy,
                    champion=name,
                    fit_regret=float(fit_gap),
                    exact_hit=fit_gap == 0,
                    nodes=tree.nodes_count,
                    tree=repr(structural_key(tree.root)),
                    **metadata,
                )


def reference_panel(data_seed):
    splits = [
        _split(data_seed, 0, 128),
        _split(data_seed, 1, 128),
        _split(data_seed, 2, 8192),
        _split(data_seed, 3, 8192, 0.78),
    ]
    providers = [part[0] for part in splits]
    labels = [part[1] for part in splits]
    matrices = [p[:, :, 1] for p in providers]
    tensors = [
        {f"p{i}.npy": x[:, i : i + 1] for i in range(x.shape[1])} for x in matrices
    ]
    start = time.perf_counter()
    oracle = fit_lattice_brier(matrices[0], labels[0])
    oracle_seconds = time.perf_counter() - start
    baseline = []
    panel = ReferencePanel(
        data_seed, matrices[0], labels, tensors, baseline, oracle.minimum_loss
    )
    exact_pred = [oracle.predict(x) for x in matrices]
    assert exact_brier(exact_pred[0], labels[0]) == oracle.minimum_loss
    baseline.append(
        dict(
            method="exact_lattice",
            **panel.scores(exact_pred),
            optimum_numerator=oracle.minimum_loss.numerator,
            optimum_denominator=oracle.minimum_loss.denominator,
            terms=len(oracle.terms()),
            fit_seconds=oracle_seconds,
        )
    )
    baseline.append(
        dict(method="average", **panel.scores([x.mean(1) for x in matrices]))
    )
    top = TopN("brier").fit(providers[0], labels[0], task="binary", n_classes=2)
    baseline.append(
        dict(
            method="top_n",
            **panel.scores([top.predict(p)[:, 1] for p in providers]),
            selected_n=top.n_selected_,
            ranking=json.dumps(top.ranking_.tolist()),
            prefix_scores=json.dumps(np.asarray(top.prefix_scores_).tolist()),
        )
    )
    stack = StackingLogit().fit(providers[0], labels[0], task="binary", n_classes=2)
    baseline.append(
        dict(
            method="stack_logit",
            **panel.scores([stack.predict(p)[:, 1] for p in providers]),
        )
    )
    return panel


def panel(data_seed):
    from loguru import logger

    logger.disable("okapi")
    reference = reference_panel(data_seed)
    gp_runs, events = [], []
    with training_files(reference.train, reference.labels[0]) as (paths, gt):
        for gp_seed in GP_SEEDS:
            for policy in ("legacy", "fixed"):
                snapshots, collisions = fit_gp(
                    paths, gt, gp_seed, policy, reference.tensors, reference.labels
                )
                for evt in collisions:
                    events.append(
                        dict(data_seed=data_seed, gp_seed=gp_seed, policy=policy, **evt)
                    )
                gp_runs.extend(reference.champion_rows(snapshots, gp_seed, policy))
    return reference.baseline, gp_runs, events


def run(out, workers):
    out.mkdir(exist_ok=True, parents=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        panels = list(pool.map(panel, range(3500, 3512)))
    groups = list(zip(*panels, strict=True))
    for group, filename in zip(groups, ("baselines", "gp", "collisions"), strict=True):
        rows = [row for panel_rows in group for row in panel_rows]
        frame = pd.DataFrame(rows)
        if filename == "collisions" and frame.empty:
            frame = pd.DataFrame(
                columns=[
                    "data_seed",
                    "gp_seed",
                    "policy",
                    "join",
                    "index",
                    "first_index",
                    "changed_fit_rows",
                    "different_boolean_corners",
                    "retained_tree",
                    "suppressed_tree",
                ]
            )
        if filename == "baselines":
            # Preserve arbitrarily large exact certificate values even alongside empty cells.
            for col in ("optimum_numerator", "optimum_denominator"):
                frame[col] = [str(r[col]) if col in r else "" for r in rows]
        frame.to_csv(out / f"{filename}.csv", index=False)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--out", required=True, type=Path)
    cli.add_argument("--workers", default=1, type=int)
    opts = cli.parse_args()
    run(opts.out, opts.workers)

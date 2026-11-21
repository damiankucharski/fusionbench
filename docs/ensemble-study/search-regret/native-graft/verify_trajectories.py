"""Replay stored graft programs through native batched fitness and survivor selection.

Run with the pinned OKAPI checkout and FusionBench on PYTHONPATH. Submitted grafts
are reconstructed from their recorded operators. Fit forwards used to verify those
programs are outside the replayed evolutionary objective budget.
"""

import argparse
import csv
import gzip
import hashlib
import json
import struct
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import okapi
from loguru import logger
from okapi import Okapi
from okapi.node import CloseThresholdNode, FarThresholdNode, ValueNode
from okapi.operators import (
    CLOSE_THRESHOLD,
    FAR_THRESHOLD,
    MAX,
    MEAN,
    MIN,
    WEIGHTED_MEAN,
)
from okapi.population import choose_pareto_then_proximity
from okapi.random_state import use_rng

from fusionbench.eval.native_switch import serialize
from fusionbench.scenarios.native_switch import draw_native_switch

PIN = "5062ee43b773ef4d6e7c8d0286d7d9be3b03beda"
SIX = MEAN, MIN, MAX, WEIGHTED_MEAN, FAR_THRESHOLD, CLOSE_THRESHOLD


def integer(value):
    assert float(value).is_integer()
    return int(float(value))


@dataclass
class Candidate:
    loss: float
    validation: float
    tree: object


@dataclass
class Snapshot:
    step: int
    generation: int
    sha: str
    size: int
    fit: Candidate
    validation: Candidate


class BudgetReached(Exception):
    pass


class Replay(Okapi):
    """Observe complete native fitness batches and truncate only the final one."""

    def __init__(self, paths, gt, validation, validation_y, seed):
        self.validation_inputs = validation
        self.validation_truth = validation_y
        self.archive = {}
        self.sham_archive = None
        self.sham_step = None
        self.sham_snapshot = None
        self.calls = self.generation = 0
        self.trace = hashlib.sha256()
        self.stop = None
        self.record_at = set()
        self.records = {}
        super().__init__(
            paths,
            gt,
            len(paths),
            2,
            2,
            backend="numpy",
            seed=seed,
            allowed_ops=SIX,
            objective_functions=(self.objective,),
            postprocessing_function=np.asarray,
            reuse_survivor_fitness=False,
        )

    def objective(self, prediction, truth):
        value = float(np.mean(np.square(prediction.ravel() - truth.ravel())))
        self.losses.append(value)
        return 1.0 - value

    def _entry(self, tree, loss):
        clone = tree.copy()
        clone.mutation_chance = tree.mutation_chance
        val = clone.do_pred_on_another_tensors(
            current_tensors=self.validation_inputs
        ).ravel()
        return Candidate(
            loss, float(np.mean((val - self.validation_truth) ** 2)), clone
        )

    def snapshot(self, archive):
        return Snapshot(
            self.calls,
            self.generation,
            self.trace.hexdigest(),
            len(archive),
            min(archive.values(), key=lambda r: r.loss),
            min(archive.values(), key=lambda r: (r.validation, r.loss)),
        )

    def _calculate_fitnesses(self, trees=None):
        cohort = self.population if trees is None else trees
        if self.stop is not None:
            cohort = cohort[: self.stop - self.calls]
        self.losses = []
        # Native scores are computed as a batch, before any of our archival work.
        returned = super()._calculate_fitnesses(cohort)
        for tree, loss in zip(cohort, self.losses, strict=True):
            self.calls += 1
            key = tree.structural_key()
            self.trace.update((repr(key) + "\n").encode("utf-8"))
            self.trace.update(struct.pack("<d", loss))
            if key not in self.archive:
                self.archive[key] = self._entry(tree, loss)
            if self.sham_archive is not None and self.calls <= self.sham_step:
                self.sham_archive.setdefault(key, self.archive[key])
            if self.calls in self.record_at:
                self.records[self.calls] = self.snapshot(self.archive)
            if self.calls == self.sham_step:
                self.sham_snapshot = self.snapshot(self.sham_archive)
        if self.stop == self.calls:
            raise BudgetReached
        return returned

    def advance_generation(self):
        self.generation += 1
        self.run_iteration()

    def initial_boundary(self):
        while self.calls < 1200:
            self.advance_generation()
        return self.snapshot(self.archive)

    def run_to(self, targets):
        self.record_at = set(targets)
        self.stop = max(self.record_at)
        if self.calls in self.record_at:
            self.records[self.calls] = self.snapshot(self.archive)
        try:
            while self.calls < self.stop:
                self.advance_generation()
        except BudgetReached:
            pass
        assert set(self.records) == self.record_at

    def restore_offer(self, original, row):
        # Preserve the concrete scalar types of inherited native parameters.
        # JSON encodes their values but does not encode Python/NumPy scalar types.
        candidate = original.tree.copy()
        candidate.mutation_chance = original.tree.mutation_chance
        if row["selected_operand"]:
            site = candidate.root
            for i in json.loads(row["site_path"]):
                site = site.children[i]
            assert not site.children and site.id == row["anchor"]
            source = row["selected_operand"]
            leaf = ValueNode(None, self.train_tensors[source], source)
            cls = {"far": FarThresholdNode, "close": CloseThresholdNode}[
                row["direction"]
            ]
            site.add_child(cls([leaf], float(row["threshold"])))
            candidate.update_nodes()
        assert serialize(candidate.root) == json.loads(row["program"])
        fit = float(
            np.mean((candidate.predict().ravel() - self.gt_tensor.ravel()) ** 2)
        )
        assert fit == float(row["graft_fit_loss"])
        return candidate, self._entry(candidate, fit)

    def insert(self, incumbent, candidate, entry):
        key = candidate.structural_key()
        self.archive.setdefault(key, entry)
        already_live = any(t.structural_key() == key for t in self.population)
        if key == incumbent.tree.structural_key():
            return "unchanged", already_live
        if already_live:
            return "duplicate_parent", True
        previous = self.rng.get_state()
        combined = [*self.population, candidate]
        scores = np.vstack([self.fitnesses, np.round(1.0 - entry.loss, 3)])
        with use_rng(self.rng):
            new, new_scores = choose_pareto_then_proximity(
                combined,
                scores,
                self.objectives,
                len(self.population),
                self.minimize_node_count,
            )
        if candidate in new:
            self.population, self.fitnesses = new, new_scores
            return "admitted", True
        assert set(new) == set(self.population)
        self.rng.set_state(previous)
        return "rejected", False


def check(row, snapshot, stats):
    assert snapshot.step == integer(row["ordinary_calls"])
    assert snapshot.generation == integer(row["observed_generation"])
    assert snapshot.size == integer(row["archive_trees"])
    assert snapshot.sha == row["path_sha256"], (
        row["data_seed"],
        row["gp_seed"],
        row["method"],
        row["arm"],
    )
    champion = snapshot.fit if row["decision"] == "fit" else snapshot.validation
    assert serialize(champion.tree.root) == json.loads(row["program"])
    assert champion.loss == float(row["observed_brier"])
    stats["champion_programs_and_fit_losses"] += 1


def verify_one(paths, gt, val, val_y, seed, rows, stats):
    rows_by_key = {(r["method"], r["arm"], r["decision"]): r for r in rows}
    assert len(rows_by_key) == len(rows)
    initial = rows_by_key["gp_initial", "none", "fit"]
    offer = rows_by_key["immediate", "all_tuned", "fit"]
    n = integer(initial["base_calls"])
    tail = n + 300
    path_rows = [r for r in rows if r["path_sha256"]]
    targets = {integer(r["ordinary_calls"]) for r in path_rows}
    plain = Replay(paths, gt, val, val_y, seed)
    base = plain.initial_boundary()
    assert base.step == n and base.generation == integer(initial["base_generation"])
    candidate, entry = plain.restore_offer(base.fit, offer)
    plain.sham_archive = dict(plain.archive)
    plain.sham_archive.setdefault(candidate.structural_key(), entry)
    plain.sham_step = tail
    plain.run_to(targets)

    active = Replay(paths, gt, val, val_y, seed)
    again = active.initial_boundary()
    assert (again.step, again.generation, again.sha) == (n, base.generation, base.sha)
    candidate, entry = active.restore_offer(again.fit, offer)
    status, live = active.insert(again.fit, candidate, entry)
    active.run_to({tail})
    extension = active.records[tail]
    changed_path = extension.sha != plain.records[tail].sha
    if status != "admitted":
        assert not changed_path
        assert extension.size == plain.sham_snapshot.size
        assert (
            extension.fit.tree.structural_key()
            == plain.sham_snapshot.fit.tree.structural_key()
        )
        assert (
            extension.validation.tree.structural_key()
            == plain.sham_snapshot.validation.tree.structural_key()
        )
    stats["insert_" + status] += 1
    stats["changed_paths"] += changed_path
    stats["distinct_plain_checkpoints"] += len(targets)
    for row in path_rows:
        method = row["method"]
        if method in ("active_tail", "archive_only_tail"):
            assert row["graft_status"] == status
            assert (row["full_graft_in_survivors"] == "True") == live
            assert (row["full_graft_inserted"] == "True") == (status == "admitted")
            assert (row["active_path_changed"] == "True") == changed_path
        if method == "active_tail":
            snapshot = extension
        elif method == "archive_only_tail":
            snapshot = plain.sham_snapshot
        else:
            snapshot = plain.records[integer(row["ordinary_calls"])]
        check(row, snapshot, stats)
        if row["decision"] == "fit":
            stats["named_path_checkpoints"] += 1
    stats["paired_runs"] += 1


def verify(directory):
    root = Path(okapi.__file__).resolve().parent
    assert (
        subprocess.check_output(
            ["git", "-C", root, "rev-parse", "HEAD"], text=True
        ).strip()
        == PIN
    )
    assert not subprocess.check_output(
        ["git", "-C", root, "status", "--short", "--untracked-files=no"]
    )
    groups = defaultdict(list)
    with gzip.open(directory / "runs.csv.gz", "rt") as stream:
        for row in csv.DictReader(stream):
            if row["split"] == "fit" and row["gp_seed"]:
                groups[
                    integer(row["data_seed"]),
                    row["fit_regime"],
                    integer(row["gp_seed"]),
                ].append(row)
    assert set(groups) == {
        (d, r, s)
        for d in range(3900, 3912)
        for r in ("switch", "linear")
        for s in (293, 307)
    }
    logger.disable("okapi")
    stats = Counter()
    for (seed, regime, gp_seed), rows in sorted(groups.items()):
        fit, val = [draw_native_switch(seed, i, n) for i, n in enumerate((512, 256))]
        mix = int(regime == "linear")
        validation = {f"p{i}.npy": val.preds[:, i, 1:2] for i in range(6)}
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            paths = [folder / f"p{i}.npy" for i in range(6)]
            for i, path in enumerate(paths):
                np.save(path, fit.preds[:, i, 1:2])
            gt = folder / "labels.npy"
            np.save(gt, fit.labels(mix))
            verify_one(paths, gt, validation, val.labels(mix), gp_seed, rows, stats)
    print(json.dumps({"okapi": PIN, **dict(sorted(stats.items()))}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    verify(parser.parse_args().directory)

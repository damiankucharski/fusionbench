"""Separate stochastic search error, sampling error and family mismatch on a 4-bit cube.

Run with the optional OKAPI dependencies installed:
    python -m fusionbench.eval.search_regret --out results
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.special import expit

from fusionbench.methods.stacking import StackingLogit

TASKS = ("two_pathways", "parity", "additive")
CUBE = (np.arange(16)[:, None] >> np.arange(4)) & 1
SCORES = (1 + 6 * CUBE) / 8
PROBA = np.stack([1 - SCORES, SCORES], axis=-1)
POWERS = 1 << np.arange(16)
IID = np.full(16, 1 / 16)
SHIFT = np.array([0.80, 0.15, 0.75, 0.20])
SHIFT = np.prod(np.where(CUBE, SHIFT, 1 - SHIFT), axis=1)


@lru_cache(maxsize=1)
def family():
    from okapi.lattice_search import SmallLattice, small_lattice_programs

    programs = small_lattice_programs(4)
    values = np.stack(
        [SmallLattice(p, 4, len(programs)).predict(CUBE) for p in programs]
    ).astype(int)
    assert len(np.unique(values, axis=0)) == 166
    return values, values @ POWERS


def conditional(task):
    a, b, c, d = CUBE.T
    if task == "two_pathways":
        return 0.18 + 0.64 * ((a & b) | (c & d))
    if task == "parity":
        return 0.18 + 0.64 * (a ^ b)
    if task == "additive":
        return expit(1.5 * (a + b + c + d - 1.8))
    raise ValueError(f"unknown task {task}")


def _decisions(code):
    return (int(code) >> np.arange(16)) & 1


@dataclass(frozen=True)
class Counts:
    pos: np.ndarray
    neg: np.ndarray

    @classmethod
    def from_rows(cls, cells, y):
        return cls(
            np.bincount(cells[y], minlength=16), np.bincount(cells[~y], minlength=16)
        )

    def errors(self, decisions):
        return self.pos.sum() + decisions @ (self.neg - self.pos)

    @property
    def size(self):
        return (self.pos + self.neg).sum()


def draw(seed, split, n, weights, eta):
    rng = np.random.default_rng([seed, split])
    cells = rng.choice(16, size=n, p=weights)
    return cells, rng.random(n) < eta[cells]


def population_risk(pred, eta, weights):
    return (eta + pred * (1 - 2 * eta)) @ weights


class Problem:
    def __init__(self, task, seed, n_fit, max_fit, n_val, n_test):
        self.eta = conditional(task)
        cells, labels = draw(seed, 0, max_fit, IID, self.eta)
        self.cells, self.labels = cells[:n_fit], labels[:n_fit]
        self.fit = Counts.from_rows(self.cells, self.labels)
        self.val = Counts.from_rows(*draw(seed, 1, n_val, IID, self.eta))
        self.iid = Counts.from_rows(*draw(seed, 2, n_test, IID, self.eta))
        self.shift = Counts.from_rows(*draw(seed, 3, n_test, SHIFT, self.eta))
        tables, codes = family()
        losses = self.fit.errors(tables)
        self.best_errors = int(losses.min())
        best = np.flatnonzero(losses == self.best_errors)
        self.erm = int(codes[best[0]])
        self.tied = len(best)
        self.unrestricted = int(np.minimum(self.fit.pos, self.fit.neg).sum())
        risks = population_risk(tables, self.eta, IID)
        self.erm_min_iid, self.erm_max_iid = risks[best].min(), risks[best].max()
        self.family_oracles = {
            name: population_risk(tables, self.eta, w).min()
            for name, w in [("iid", IID), ("shift", SHIFT)]
        }

    def metrics(self, code):
        pred = _decisions(code)
        errors = int(self.fit.errors(pred))
        risk = population_risk(pred, self.eta, IID)
        shifted = population_risk(pred, self.eta, SHIFT)
        return dict(
            code=int(code),
            fit_errors=errors,
            fit_loss=errors / self.fit.size,
            val_loss=self.val.errors(pred) / self.val.size,
            iid_test_loss=self.iid.errors(pred) / self.iid.size,
            shift_test_loss=self.shift.errors(pred) / self.shift.size,
            iid_risk=risk,
            shift_risk=shifted,
            iid_minus_fit=risk - errors / self.fit.size,
            delta_family_fit_errors=errors - self.best_errors,
            above_unrestricted_fit_errors=errors - self.unrestricted,
            delta_family_iid=risk - self.family_oracles["iid"],
            delta_family_shift=shifted - self.family_oracles["shift"],
            iid_minus_exact_erm=risk
            - population_risk(_decisions(self.erm), self.eta, IID),
        )


class FitnessArchive:
    def __init__(self, fit):
        self.fit = fit
        self.errors = {}
        self.calls = 0

    def __call__(self, score, labels):
        # All candidates have length 16. Only cells with observed *fit* labels
        # enter the objective. The other decision bits track distinct functions.
        prediction = score.reshape(-1) > 0.5
        code = int(prediction @ POWERS)
        self.calls += 1
        if code not in self.errors:
            self.errors[code] = int(self.fit.errors(prediction))
        return 1 - self.errors[code] / self.fit.size


def run_search(problem, paths, label_file, seed, checkpoints, mode):
    from okapi import Okapi
    from okapi.operators import MAX, MIN

    kwargs = {"allowed_ops": (MIN, MAX)} if mode == "minmax" else {}
    archive = FitnessArchive(problem.fit)
    start = time.perf_counter()
    engine = Okapi(
        paths,
        label_file,
        population_size=4,
        population_multiplier=2,
        tournament_size=2,
        objective_functions=(archive,),
        backend="numpy",
        postprocessing_function=np.asarray,
        seed=seed,
        **kwargs,
    )
    tables, codes = family()
    known = set(codes)
    random_order = np.random.default_rng([seed, 71]).permutation(len(codes))
    rows = []
    for step in range(1, max(checkpoints) + 1):
        engine.run_iteration()
        if step not in checkpoints:
            continue
        wall = time.perf_counter() - start
        # Archive iteration order is the first visit, including after ties.
        empirical = min(archive.errors, key=archive.errors.get)
        val = min(
            archive.errors,
            key=lambda k: (problem.val.errors(_decisions(k)), archive.errors[k]),
        )
        retained = engine.population[int(np.argmax(engine.fitnesses[:, 0]))]
        retained_code = int((retained.predict().reshape(-1) > 0.5) @ POWERS)
        choices = [
            ("archive_fit", empirical),
            ("archive_val", val),
            ("retained_fit", retained_code),
        ]
        if mode == "minmax":
            if not known.issuperset(archive.errors):
                raise RuntimeError("MIN/MAX search left the certified family")
            subset = random_order[: len(archive.errors)]
            random_best = subset[np.argmin(problem.fit.errors(tables[subset]))]
            choices.append(("uniform_unique", codes[random_best]))
        for selector, code in choices:
            rows.append(
                dict(
                    mode=mode,
                    selector=selector,
                    search_seed=seed,
                    generation=step,
                    calls=archive.calls,
                    unique=len(archive.errors),
                    outside_family=sum(k not in known for k in archive.errors),
                    nodes=retained.nodes_count if selector == "retained_fit" else None,
                    seconds=wall,
                    **problem.metrics(code),
                )
            )
    return rows


def panel(config):
    task, seed, sizes, val_n, test_n, gp_seeds, checkpoints = config
    reports = []
    with TemporaryDirectory() as directory:
        path = Path(directory)
        paths = [path / f"provider{i}.npy" for i in range(4)]
        for i, file in enumerate(paths):
            np.save(file, SCORES[:, i : i + 1])
        for size in sizes:
            problem = Problem(task, seed, size, max(sizes), val_n, test_n)
            # The scoring callback uses fit-only counts. This file supplies the
            # public API's required target tensor and contains no synthetic truth.
            label_file = path / "labels.npy"
            np.save(
                label_file,
                problem.fit.pos[:, None]
                / np.maximum(1, (problem.fit.pos + problem.fit.neg)[:, None]),
            )
            bayes = int((problem.eta > 0.5) @ POWERS)
            avg = int((SCORES.mean(axis=1) > 0.5) @ POWERS)
            if np.unique(problem.labels).size < 2:
                stack = int(np.full(16, bool(problem.labels[0])) @ POWERS)
            else:
                stacker = StackingLogit().fit(
                    PROBA[problem.cells],
                    problem.labels.astype(int),
                    task="binary",
                    n_classes=2,
                )
                stack = int((stacker.predict(PROBA)[:, 1] > 0.5) @ POWERS)
            common = dict(
                task=task,
                data_seed=seed,
                n_fit=size,
                n_val=val_n,
                n_test=test_n,
                family_fit_errors=problem.best_errors,
                all_fit_errors=problem.unrestricted,
                tied_erm=problem.tied,
                min_iid_among_erm=problem.erm_min_iid,
                max_iid_among_erm=problem.erm_max_iid,
                family_iid_oracle=problem.family_oracles["iid"],
                family_shift_oracle=problem.family_oracles["shift"],
                bayes_iid=population_risk(_decisions(bayes), problem.eta, IID),
                bayes_shift=population_risk(_decisions(bayes), problem.eta, SHIFT),
            )
            for method, code in [
                ("family_erm", problem.erm),
                ("average", avg),
                ("stack_logit", stack),
                ("bayes", bayes),
            ]:
                reports.append(
                    dict(
                        **common,
                        mode=method,
                        selector=method,
                        search_seed=None,
                        generation=None,
                        calls=None,
                        unique=None,
                        outside_family=None,
                        nodes=None,
                        seconds=None,
                        **problem.metrics(code),
                    )
                )
            for mode, search_seed in product(("minmax", "default"), gp_seeds):
                reports.extend(
                    dict(**common, **row)
                    for row in run_search(
                        problem, paths, label_file, search_seed, checkpoints, mode
                    )
                )
    return reports


def run(
    out,
    data_seeds,
    gp_seeds,
    sizes,
    checkpoints,
    val_n=128,
    test_n=4096,
    workers=1,
    tasks=TASKS,
):
    if (
        not data_seeds
        or not gp_seeds
        or not sizes
        or not checkpoints
        or min(*sizes, *checkpoints, val_n, test_n, workers) < 1
    ):
        raise ValueError(
            "provide nonempty seed/budget lists and positive sample, worker and generation counts"
        )
    out = Path(out)
    out.mkdir(exist_ok=True, parents=True)
    settings = dict(
        data_seeds=list(data_seeds),
        gp_seeds=list(gp_seeds),
        fit_sizes=list(sizes),
        validation=val_n,
        test=test_n,
        checkpoints=list(checkpoints),
        tasks=list(tasks),
        score_levels=[0.125, 0.875],
        iid_cell_weights=IID.tolist(),
        shifted_cell_weights=SHIFT.tolist(),
    )
    (out / "config.json").write_text(json.dumps(settings, indent=2) + "\n")
    jobs = [
        (t, s, sizes, val_n, test_n, gp_seeds, checkpoints)
        for t, s in product(tasks, data_seeds)
    ]
    with (
        ProcessPoolExecutor(max_workers=workers) as pool,
        (out / "runs.csv").open("w", newline="") as handle,
    ):
        for i, rows in enumerate(pool.map(panel, jobs)):
            if i == 0:
                writer = csv.DictWriter(handle, fieldnames=rows[0])
                writer.writeheader()
            writer.writerows(rows)
            handle.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--data-seeds", type=int, nargs="+", default=list(range(3000, 3008))
    )
    parser.add_argument("--gp-seeds", type=int, nargs="+", default=[31, 47])
    parser.add_argument("--fit-sizes", type=int, nargs="+", default=[32, 128])
    parser.add_argument("--generations", type=int, nargs="+", default=[1, 4, 16, 48])
    parser.add_argument("--validation", type=int, default=128)
    parser.add_argument("--test", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--tasks", choices=TASKS, nargs="+", default=TASKS)
    args = parser.parse_args()
    run(
        args.out,
        args.data_seeds,
        args.gp_seeds,
        args.fit_sizes,
        args.generations,
        args.validation,
        args.test,
        args.workers,
        args.tasks,
    )

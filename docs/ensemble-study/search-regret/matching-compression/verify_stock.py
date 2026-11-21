"""Reconstruct the study data and replay every native GP evaluation independently."""

import argparse
import csv
import gzip
import hashlib
import itertools
import json
import struct
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import okapi
from loguru import logger
from okapi import Okapi
from okapi.operators import (
    CLOSE_THRESHOLD,
    FAR_THRESHOLD,
    MAX,
    MEAN,
    MIN,
    WEIGHTED_MEAN,
)

from fusionbench.eval.native_switch import serialize

OPS = (MEAN, MIN, MAX, WEIGHTED_MEAN, FAR_THRESHOLD, CLOSE_THRESHOLD)
PIN = "76f96bc64b8bd2295aef783c47153f72678501e9"
LIMIT = 600


def equal(value, reference, context):
    if value != reference:
        raise ValueError(f"{context}: reconstructed {value!r}, recorded {reference!r}")


class Cutoff(Exception):
    pass


class PlainGP(Okapi):
    """Keep the native optimizer and instrument each true Brier objective call."""

    def __init__(self, paths, gt, validation, val_y, seed):
        self.validation = validation
        self.val_y = val_y
        self.keys = set()
        self.trace = hashlib.sha256()
        self.calls = 0
        self.generation = 0
        self.champions = {"fit": None, "validation": None}
        super().__init__(
            paths,
            gt,
            len(paths),
            2,
            2,
            backend="numpy",
            seed=seed,
            allowed_ops=OPS,
            objective_functions=(self.raw_brier,),
            postprocessing_function=np.asarray,
            reuse_survivor_fitness=False,
        )

    def raw_brier(self, prediction, truth):
        self.raw_loss = float(np.mean(np.square(prediction.ravel() - truth.ravel())))
        return 1 - self.raw_loss

    def _calculate_fitnesses(self, trees=None):
        scores = []
        for tree in self.population if trees is None else trees:
            scores.append(super()._calculate_fitnesses([tree])[0])
            self.calls += 1
            key = tree.structural_key()
            self.trace.update((repr(key) + "\n").encode())
            self.trace.update(struct.pack("<d", self.raw_loss))
            if key not in self.keys:
                self.keys.add(key)
                vp = tree.do_pred_on_another_tensors(
                    current_tensors=self.validation
                ).ravel()
                val = float(np.mean(np.square(vp - self.val_y)))
                losses = {"fit": (self.raw_loss,), "validation": (val, self.raw_loss)}
                for selection, score in losses.items():
                    previous = self.champions[selection]
                    if previous is None or score < previous[0]:
                        self.champions[selection] = (
                            score,
                            tree.copy(),
                            self.raw_loss,
                            val,
                        )
            if self.calls == LIMIT:
                raise Cutoff
        return np.stack(scores)

    def replay(self):
        try:
            while True:
                self.generation += 1
                self.run_iteration()
        except Cutoff:
            equal(self.calls, LIMIT, "true objective count")


class PairBank:
    """Independently regenerate matching, control roles and the two-hot indicator rows."""

    def __init__(self, seed, k):
        self.seed, self.k = seed, k
        n = 2 * k
        self.edges = np.array(list(itertools.combinations(range(n), 2)), dtype=int)
        pairs = self.rng(0).permutation(n).reshape(k, 2)
        self.matching = tuple(sorted(tuple(sorted(map(int, pair))) for pair in pairs))
        self.matched = np.array(self.matching, dtype=int)
        self.cross = np.array(
            [e for e in map(tuple, self.edges) if e not in self.matching]
        )
        self.roles = self.rng(1).permutation(n)[:4]

    def rng(self, stream, *extra):
        return np.random.default_rng([self.seed, self.k, stream, *extra])

    def indicators(self, choices):
        x = np.zeros((len(choices), 2 * self.k), dtype=np.int8)
        x[np.arange(len(choices))[:, None], choices] = 1
        return x

    def probabilities(self, bits, law):
        if law == "matching":
            both = bits[:, self.matched[:, 0]] & bits[:, self.matched[:, 1]]
            return np.where(np.any(both, axis=1), 63 / 64, 1 / 64)
        if law != "logistic":
            raise ValueError(f"Unknown law {law}")
        a, b, c, d = bits[:, self.roles].T
        return 1 / (1 + np.exp(-(0.2 + 1.1 * a + 0.6 * b - 1.2 * c - 0.7 * d)))

    def draw(self, size, stream, design, law):
        design_id = {"balanced": 0, "uniform": 1}[design]
        generator = self.rng(stream, design_id)
        if design_id == 1:
            choices = self.edges[generator.integers(len(self.edges), size=size)]
        else:
            inside = generator.random(size) < 0.5
            positive = self.matched[generator.integers(len(self.matched), size=size)]
            negative = self.cross[generator.integers(len(self.cross), size=size)]
            choices = np.where(inside[:, None], positive, negative)
        bits = self.indicators(choices)
        q = self.probabilities(bits, law)
        u = self.rng(stream + 1, design_id).random(size)
        return (1 + 62 * bits) / 64, (u < q).astype(np.int64)

    def test(self, active_count, law):
        sets = np.array(list(itertools.combinations(range(2 * self.k), active_count)))
        bits = self.indicators(sets)
        q = self.probabilities(bits, law)
        labels = self.rng(50 + active_count).random((len(bits), 64)) < q[:, None]
        x = (1 + 62 * bits) / 64
        uniform = np.full(len(bits), 1 / len(bits))
        if active_count == 2:
            matched = np.fromiter((tuple(s) in self.matching for s in sets), bool)
            balanced = np.where(matched, 0.5 / self.k, 0.5 / len(self.cross))
            return x, q, labels, uniform, balanced
        return x, q, labels, uniform, None


def tensors(x):
    return {f"p{i}.npy": x[:, i : i + 1] for i in range(x.shape[1])}


@dataclass
class Split:
    x: np.ndarray
    q: np.ndarray
    y: np.ndarray
    w: np.ndarray | None

    def losses(self, prediction):
        if self.y.ndim == 1:
            per_state = np.square(prediction - self.y)
        else:
            per_state = np.square(prediction[:, None] - self.y).mean(axis=1)
        return {
            "observed_brier": float(np.average(per_state, weights=self.w)),
            "conditional_excess": float(
                np.average(np.square(prediction - self.q), weights=self.w)
            ),
        }


def build_splits(bank, design, n, law):
    splits = {}
    for tag, stream, total in (("fit", 10, 64), ("validation", 20, 128)):
        scores, labels = bank.draw(total, stream, design, law)
        if tag == "fit":
            scores, labels = scores[:n], labels[:n]
        q = bank.probabilities(scores > 0.5, law)
        splits[tag] = Split(scores, q, labels, None)
    for size, name in ((2, "two_uniform"), (3, "three_uniform"), (4, "four_uniform")):
        scores, q, labels, w, balanced = bank.test(size, law)
        splits[name] = Split(scores, q, labels, w)
        if size == 2:
            splits["two_balanced"] = Split(scores, q, labels, balanced)
    return splits


def run_engine(splits, gp_seed):
    fit, val = splits["fit"], splits["validation"]
    with TemporaryDirectory() as d:
        base = Path(d)
        paths = []
        for i, column in enumerate(fit.x.T):
            path = base / f"p{i}.npy"
            np.save(path, column[:, None])
            paths.append(path)
        gt = base / "labels.npy"
        np.save(gt, fit.y)
        engine = PlainGP(paths, gt, tensors(val.x), val.y, gp_seed)
        engine.replay()
    return engine


def within_error(result, saved, maxima, metric):
    error = abs(result - float(saved))
    if not error <= 1e-14:
        raise ValueError(f"{metric} error {error:g}: {result!r} vs {saved!r}")
    maxima[metric] = max(maxima[metric], error)


def verify_row(p, split, saved, maxima, counts):
    equal(
        hashlib.sha256(np.asarray(p, dtype="<f8").tobytes()).hexdigest(),
        saved["prediction_sha256"],
        "prediction hash",
    )
    for metric, value in split.losses(p).items():
        within_error(value, saved[metric], maxima, metric)
    counts["prediction_sha256"] += 1
    counts["metrics"] += 2


def verify_group(job):
    seed, k, design, stores = job
    logger.disable("okapi")
    counts, maxima = Counter(), defaultdict(float)
    bank = PairBank(seed, k)
    for (n, law), (runs, gps, oracle) in stores.items():
        splits = build_splits(bank, design, n, law)
        equal(set(oracle), set(splits), "exhaustive reference split names")
        for name, support in splits.items():
            verify_row(support.q, support, oracle[name], maxima, counts)
            counts["independently_generated_law_rows"] += 1
        for gp_seed, row in runs.items():
            context = (seed, k, design, n, law, gp_seed)
            engine = run_engine(splits, gp_seed)
            for field, result in (
                ("calls", engine.calls),
                ("archive_trees", len(engine.keys)),
                ("generation", engine.generation),
            ):
                equal(result, int(row[field]), (context, field))
            equal(engine.trace.hexdigest(), row["path_sha256"], (context, "path"))
            counts["native_trajectories"] += 1
            counts["objective_calls"] += engine.calls
            counts["archived_structures"] += len(engine.keys)
            for decision, (_, tree, train_loss, val_loss) in engine.champions.items():
                champion_score = train_loss if decision == "fit" else val_loss
                within_error(
                    champion_score,
                    row[f"{decision}_brier"],
                    maxima,
                    "raw_champion_loss",
                )
                counts["champions"] += 1
                for name, split in splits.items():
                    saved = gps[(gp_seed, decision, name)]
                    equal(
                        serialize(tree.root),
                        json.loads(saved["model"]),
                        (context, decision, "program"),
                    )
                    equal(
                        tree.nodes_count,
                        int(float(saved["nodes"])),
                        (context, decision, "nodes"),
                    )
                    p = tree.do_pred_on_another_tensors(
                        current_tensors=tensors(split.x)
                    ).ravel()
                    verify_row(p, split, saved, maxima, counts)
                    counts["gp_champion_split_rows"] += 1
            equal(
                len(gps),
                2 * len(runs) * len(splits),
                (context, "expected two GP decisions"),
            )
        counts["banks_by_law_and_label_budget"] += 1
    return dict(counts), dict(maxima)


def read_rows(path):
    with gzip.open(path, mode="rt", newline="") as f:
        yield from csv.DictReader(f)


def row_id(row):
    return (
        int(row["data_seed"]),
        int(row["k"]),
        row["train_design"],
        int(row["labels"]),
        row["fit_law"],
    )


def put(mapping, key, value):
    if key in mapping:
        raise ValueError(f"Duplicate row key {key}")
    mapping[key] = value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output", type=Path, help="directory containing GP and run tables"
    )
    parser.add_argument("--workers", type=int, default=1, choices=(1, 2))
    args = parser.parse_args()
    checkout = Path(okapi.__file__).resolve().parent
    pin = subprocess.check_output(
        ["git", "-C", checkout, "rev-parse", "HEAD"], text=True
    ).strip()
    equal(pin, PIN, "native OKAPI revision")
    clean = subprocess.check_output(
        ["git", "-C", checkout, "status", "--porcelain", "-uno"], text=True
    )
    equal(clean, "", "unmodified native source")
    tables = defaultdict(lambda: ({}, {}, {}))
    for r in read_rows(args.output / "gp.csv.gz"):
        put(tables[row_id(r)][0], int(r["gp_seed"]), r)
    for r in read_rows(args.output / "runs.csv.gz"):
        if r["method"] == "known_law":
            put(tables[row_id(r)][2], r["split"], r)
        elif r["method"] == "stock_gp":
            key = (int(float(r["gp_seed"])), r["decision"], r["split"])
            put(tables[row_id(r)][1], key, r)

    groups = defaultdict(dict)
    expected = set(
        itertools.product(
            range(4400, 4416),
            (4, 6),
            ("balanced", "uniform"),
            (16, 64),
            ("matching", "logistic"),
        )
    )
    equal(set(tables), expected, "all prespecified bank keys")
    for (s, k, design, n, law), stores in tables.items():
        equal(set(stores[0]), {521, 523}, "native GP search seeds")
        groups[(s, k, design)][(n, law)] = stores
    counts, maxima = Counter(), defaultdict(float)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result, errors in pool.map(
            verify_group, ((*index, data) for index, data in groups.items())
        ):
            counts.update(result)
            for metric, error in errors.items():
                maxima[metric] = max(maxima[metric], error)
    equal(counts["native_trajectories"], 512, "independent native GP trajectories")
    equal(counts["objective_calls"], 307200, "true native objective calls")
    equal(counts["champions"], 1024, "fit and validation champions")
    print(
        json.dumps(
            {
                "okapi_sha": pin,
                "bank_groups": len(groups),
                "checks": dict(counts),
                "maximum_absolute_error": dict(maxima),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

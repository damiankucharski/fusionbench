"""Independent native stock/point replay. Resume an incremental journal and compare it to the raw table."""

import argparse
import csv
import gzip
import hashlib
import itertools
import json
import os
import struct
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import okapi
from loguru import logger
from okapi import Okapi
from okapi.functions import scale_vector_to_sum_1
from okapi.node import MaxNode, MeanNode, MinNode, ValueNode
from okapi.operators import (
    CLOSE_THRESHOLD,
    FAR_THRESHOLD,
    MAX,
    MEAN,
    MIN,
    WEIGHTED_MEAN,
)
from okapi.tree import Tree
from sklearn.metrics import f1_score, log_loss

from fusionbench.eval.full_suite_point import data_for
from fusionbench.eval.native_switch import serialize
from fusionbench.scenarios.registry import SCENARIOS

PIN = "db7b2b15c7064ad34f638e5e93a8cdb01e5271b8"
BUDGETS = (600, 1200)
NATIVE_OPS = (MEAN, MIN, MAX, WEIGHTED_MEAN, FAR_THRESHOLD, CLOSE_THRESHOLD)


def truth(expression, message):
    if not expression:
        raise ValueError(message)


def revision():
    folder = Path(okapi.__file__).resolve().parent
    sha = subprocess.check_output(
        ["git", "-C", folder, "rev-parse", "HEAD"], text=True
    ).strip()
    truth(sha == PIN, ("OKAPI revision", sha))
    status = subprocess.check_output(
        ["git", "-C", folder, "status", "--porcelain", "-uno"], text=True
    )
    truth(not status, ("modified OKAPI", status))


def digest(values):
    return hashlib.sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest()


def add_shallow_trees(gp, seed):
    """Rebuild the 30 public adapter seeds independently from native provider identifiers."""
    rng = np.random.default_rng(seed)
    n = len(gp.ids)
    for _ in range(30):
        subset = rng.choice(n, size=rng.integers(2, min(4, n) + 1), replace=False)
        leaves = [ValueNode(None, gp.models[i], gp.ids[i]) for i in subset]
        op = (MinNode, MaxNode, MeanNode)[rng.integers(3)]
        leaves[0].add_child(op(leaves[1:]))
        gp.population.append(Tree.create_tree_from_root(leaves[0]))
    gp.population_size = min(40, gp.population_size + 30)
    gp.fitnesses = None


def bind(values):
    x = values[:, :, 1:] if values.shape[-1] == 2 else values
    return {f"m{i}.npy": x[:, i, :] for i in range(x.shape[1])}


def half_brier(p, y):
    if p.shape[1] == 1:
        return float(np.mean(np.square(p[:, 0] - y)))
    return float(np.mean(np.sum(np.square(p - np.eye(p.shape[1])[y]), axis=1))) / 2


def positive_or_simplex(p, classes):
    if classes == 2:
        pos = p[:, 0].clip(0, 1)
        return np.column_stack((1 - pos, pos))
    bounded = np.maximum(p, 1e-12)
    return bounded / np.sum(bounded, axis=1, keepdims=True)


def eval_split(tree, tensors, labels, oracle, classes):
    raw = tree.do_pred_on_another_tensors(current_tensors=tensors)
    p = positive_or_simplex(raw, classes)
    canonical = np.maximum(p, 1e-12)
    canonical /= canonical.sum(axis=1, keepdims=True)
    decisions = canonical.argmax(1)
    return {
        "rows": len(p),
        "prediction_sha256": digest(p),
        "oracle_sha256": digest(oracle) if oracle is not None else "",
        "brier": float(np.mean(np.sum((p - np.eye(classes)[labels]) ** 2, axis=1))),
        "conditional_excess": None
        if oracle is None
        else float(np.mean(np.sum((p - oracle) ** 2, axis=1))),
        "macro_f1": float(
            f1_score(labels, decisions, average="macro", zero_division=0)
        ),
        "accuracy": float(np.mean(decisions == labels)),
        "log_loss": float(log_loss(labels, canonical, labels=list(range(classes)))),
    }


@lru_cache(maxsize=1)
def bank(scenario, seed):
    sc, splits = data_for(scenario, seed)
    return sc.n_classes, {s: (bind(x), y, q) for s, (x, y, q) in splits.items()}


class End(Exception):
    pass


def run_job(key):
    logger.disable("okapi")
    scenario, data_seed, search_seed, point = key
    classes, splits = bank(scenario, data_seed)
    x, fit_y, _ = splits["fit"]
    vx, val_y, _ = splits["validation"]
    arch, checkpoints = set(), []
    best_fit = best_val = None
    count = gen = 0
    path = hashlib.sha256()
    last = None

    def objective(p, labels):
        nonlocal last
        last = half_brier(p, labels.ravel())
        return 1 - last

    class IndependentObserver(Okapi):
        def _calculate_fitnesses(self, trees=None):
            nonlocal best_fit, best_val, count
            scores = []
            for tree in self.population if trees is None else trees:
                scores.append(super()._calculate_fitnesses([tree])[0])
                fit_loss = last
                fingerprint = tree.structural_key()
                count += 1
                path.update((repr(fingerprint) + "\n").encode())
                path.update(struct.pack("<d", fit_loss))
                if fingerprint not in arch:
                    arch.add(fingerprint)
                    vp = tree.do_pred_on_another_tensors(current_tensors=vx)
                    val_loss = half_brier(vp, val_y)
                    if best_fit is None or fit_loss < best_fit[0]:
                        best_fit = (fit_loss, tree.copy())
                    if best_val is None or (val_loss, fit_loss) < best_val[:2]:
                        best_val = (val_loss, fit_loss, tree.copy())
                if count in BUDGETS:
                    cp = {
                        "calls": count,
                        "archive_trees": len(arch),
                        "generation": gen,
                        "path_digest": path.hexdigest(),
                        "champions": {},
                    }
                    for decision, row in (("fit", best_fit), ("validation", best_val)):
                        champion = row[-1]
                        cp["champions"][decision] = {
                            "program": serialize(champion.root),
                            "nodes": champion.nodes_count,
                            "loss": row[0],
                            "splits": {
                                s: eval_split(champion, *block, classes)
                                for s, block in splits.items()
                            },
                        }
                    checkpoints.append(cp)
                if count == BUDGETS[-1]:
                    raise End
            return np.array(scores)

    with TemporaryDirectory() as temp:
        store = Path(temp) / "providers"
        store.mkdir()
        for name, tensor in x.items():
            np.save(store / name, tensor.astype(np.float64))
        y_path = Path(temp) / "labels.npy"
        np.save(y_path, fit_y)
        pop = min(30, len(x))
        gp = IndependentObserver(
            store,
            y_path,
            pop,
            3,
            min(5, pop),
            seed=search_seed,
            backend="numpy",
            allowed_ops=NATIVE_OPS,
            objective_functions=(objective,),
            minimize_node_count=True,
            operator_point_mutation=point,
            reuse_survivor_fitness=False,
            postprocessing_function=np.asarray
            if classes == 2
            else scale_vector_to_sum_1,
        )
        add_shallow_trees(gp, search_seed)
        try:
            while True:
                gen += 1
                gp.run_iteration()
        except End:
            pass
    truth(len(checkpoints) == 2, ("checkpoints", key))
    return {
        "okapi_sha": PIN,
        "key": key,
        "classes": classes,
        "providers": len(x),
        "checkpoints": checkpoints,
    }


def order():
    jobs = list(
        itertools.product(SCENARIOS, range(4600, 4608), (541, 547), (False, True))
    )
    names = tuple(SCENARIOS)

    def key(j):
        scenario, seed, gp, arm = j
        stage = (
            0
            if (seed, gp) == (4600, 541)
            else 1
            if scenario in ("saturated", "multiclass_committee")
            else 2
        )
        return stage, seed, gp, names.index(scenario), arm

    return sorted(jobs, key=key)


def load_journal(path):
    data = {}
    if path.exists():
        with path.open() as f:
            for line in f:
                r = json.loads(line)
                key = tuple(r["key"])
                truth(key not in data and r["okapi_sha"] == PIN, ("journal row", key))
                data[key] = r
    return data


def collect(filename, workers):
    known = load_journal(filename)
    tasks = [k for k in order() if k not in known]
    with (
        ProcessPoolExecutor(max_workers=workers) as pool,
        filename.open("a") as open_log,
    ):
        for result in pool.map(run_job, tasks, chunksize=1):
            open_log.write(
                json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n"
            )
            open_log.flush()
            os.fsync(open_log.fileno())


def compare(filename, runs, *, partial=False):
    data = load_journal(filename)
    wanted = set(order())
    truth(bool(data) and set(data) <= wanted, "Invalid replay journal")
    if not partial:
        truth(set(data) == wanted and len(data) == 608, "Incomplete replay journal")
    counts = Counter()
    observed = set()
    maxima = defaultdict(float)
    with gzip.open(runs, "rt", newline="") as f:
        for r in csv.DictReader(f):
            if r["method"] not in ("point_on", "point_off"):
                continue
            point = r["method"] == "point_on"
            k = r["scenario"], int(r["data_seed"]), int(float(r["gp_seed"])), point
            if partial and k not in data:
                continue
            bank_record = data[k]
            cp = bank_record["checkpoints"][BUDGETS.index(int(r["checkpoint"]))]
            chosen = cp["champions"][r["selection"]]
            metrics = chosen["splits"][r["split"]]
            uid = (*k, cp["calls"], r["selection"], r["split"])
            truth(uid not in observed, ("duplicate raw row", uid))
            observed.add(uid)
            for field, record in (
                ("classes", bank_record),
                ("providers", bank_record),
                ("calls", cp),
                ("archive_trees", cp),
                ("generation", cp),
                ("nodes", chosen),
                ("rows", metrics),
            ):
                truth(
                    int(float(r[field])) == record[field],
                    (uid, field, r[field], record[field]),
                )
            truth(
                json.loads(r["program"]) == chosen["program"],
                (uid, "serialized program"),
            )
            for field, d in (
                ("path_digest", cp),
                ("prediction_sha256", metrics),
                ("oracle_sha256", metrics),
            ):
                truth(r[field] == d[field], (uid, field, r[field], d[field]))
            if r["selection"] == r["split"]:
                # The optimizer sees raw native outputs. The reporting step clips
                # probabilities (and normalizes multiclass rows) on the same data.
                discrepancy = abs(chosen["loss"] - 0.5 * float(r["brier"]))
                truth(
                    discrepancy <= 4e-12,
                    (uid, "optimization vs reported Brier", discrepancy),
                )
                counts["raw_vs_half_full_brier"] += 1
                maxima["raw_vs_half_full_brier"] = max(
                    maxima["raw_vs_half_full_brier"], discrepancy
                )
            for metric in (
                "brier",
                "macro_f1",
                "accuracy",
                "log_loss",
                "conditional_excess",
            ):
                if metrics[metric] is None:
                    truth(not r[metric], (uid, "unavailable oracle"))
                    continue
                diff = abs(metrics[metric] - float(r[metric]))
                truth(diff <= 1e-14, (uid, metric, diff))
                counts["scores"] += 1
                maxima[metric] = max(maxima[metric], diff)
    expected = {
        (*k, cp["calls"], decision, split)
        for k, rs in data.items()
        for cp in rs["checkpoints"]
        for decision, choice in cp["champions"].items()
        for split in choice["splits"]
    }
    truth(observed == expected, ("missing raw rows", len(expected - observed)))
    return {
        "okapi_sha": PIN,
        "scenarios": len({k[0] for k in data}),
        "complete": set(data) == wanted,
        "trajectories": len(data),
        "objective_calls": len(data) * BUDGETS[-1],
        "path_digests": len(data) * len(BUDGETS),
        "champions": 2 * len(data) * len(BUDGETS),
        "predictions": len(observed),
        "metrics": counts["scores"],
        "raw_vs_half_full_brier": counts["raw_vs_half_full_brier"],
        "maximum_absolute_error": dict(maxima),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "journal", type=Path, help="incremental JSONL outside the repository"
    )
    parser.add_argument("--workers", type=int, choices=(1, 2, 3, 4), default=2)
    parser.add_argument(
        "--compare",
        type=Path,
        help="compare a complete journal to the compressed raw result table",
    )
    parser.add_argument(
        "--partial",
        action="store_true",
        help="allow a prefix comparison on a journal snapshot",
    )
    args = parser.parse_args()
    revision()
    if args.compare:
        print(
            json.dumps(
                compare(args.journal, args.compare, partial=args.partial),
                indent=2,
                sort_keys=True,
            )
        )
    else:
        collect(args.journal, args.workers)


if __name__ == "__main__":
    main()

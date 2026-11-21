"""Replay all three initializations on 19 suite cases and both multiclass follow-ups.

The older full-suite audit supplies the independent split metrics and stock seed
constructor. The two four-input shapes are constructed here from the protocol.
"""

import argparse
import csv
import gzip
import hashlib
import importlib.util
import itertools
import json
import struct
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import okapi
from loguru import logger
from okapi import Okapi
from okapi.functions import scale_vector_to_sum_1
from okapi.node import MaxNode, MinNode, ValueNode
from okapi.tree import Tree

from fusionbench.eval.native_switch import serialize
from fusionbench.scenarios.registry import SCENARIOS

PIN = "3eae0e5480cba0471cc5fda71b0c90f35b6a73f2"
ARMS = ("stock", "branch", "chain")


def historical_verifier():
    path = (
        Path(__file__).resolve().parents[1]
        / "full-suite-point"
        / "verify_trajectories.py"
    )
    spec = importlib.util.spec_from_file_location("full_suite_replay", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


old = historical_verifier()


def check(condition, *context):
    if not condition:
        raise ValueError(str(context))


def ensure_revision():
    src = Path(okapi.__file__).resolve().parent
    actual = subprocess.check_output(
        ["git", "-C", src, "rev-parse", "HEAD"], text=True
    ).strip()
    check(actual == PIN, "expected pinned OKAPI", actual)
    status = subprocess.check_output(
        ["git", "-C", src, "status", "--porcelain", "-uno"], text=True
    )
    check(not status, "changed native source", status)


def jobs():
    first = [(name, 7000, 907) for name in SCENARIOS]
    multi = [(name, 7005, 919) for name in ("saturated", "multiclass_committee")]
    return [(*job, arm) for job, arm in itertools.product(first + multi, ARMS)]


def attach_extra_trees(engine, seed, arm):
    if arm == "stock":
        old.add_shallow_trees(engine, seed)
        return
    rng = np.random.default_rng(seed)
    count = len(engine.ids)
    for _ in range(30):
        if count >= 4:
            selected = rng.choice(count, 4, replace=False)
        else:
            padded = [*range(count), *rng.choice(count, 4 - count, replace=True)]
            selected = np.asarray(padded)[rng.permutation(4)]
        inner, outer = (
            (MinNode, MaxNode) if rng.integers(2) == 0 else (MaxNode, MinNode)
        )
        a, b, c, d = [
            ValueNode(None, engine.models[i], engine.ids[i]) for i in selected
        ]
        a.add_child(inner([b]))
        a.add_child(outer([c]))
        (c if arm == "branch" else a).add_child(inner([d]))
        engine.population.append(Tree.create_tree_from_root(a))
    engine.population_size = min(40, engine.population_size + 30)
    engine.fitnesses = None


class Stop(Exception):
    pass


def replay(job):
    name, data_seed, gp_seed, arm = job
    classes, splits = old.bank(name, data_seed)
    x, y, _ = splits["fit"]
    vx, vy, _ = splits["validation"]
    fingerprints, structures = set(), set()
    initial_rows, snapshots = [], []
    path = hashlib.sha256()
    champion = chosen = None
    fit_loss = pred_hash = None
    searched = generation = 0

    def objective(values, truth):
        nonlocal fit_loss, pred_hash
        fit_loss = old.half_brier(values, truth.ravel())
        pred_hash = old.digest(values)
        return 1 - fit_loss

    class Observed(Okapi):
        def _calculate_fitnesses(self, trees=None):
            nonlocal searched, champion, chosen
            scores = []
            for t in self.population if trees is None else trees:
                scores.append(super()._calculate_fitnesses([t])[0])
                searched += 1
                key = t.structural_key()
                fingerprints.add(pred_hash)
                key_bytes = repr(key).encode()
                path.update(key_bytes + b"\n" + struct.pack("<d", fit_loss))
                if searched <= initial_count:
                    initial_rows.append(
                        {
                            "index": searched - 1,
                            "origin": "singleton" if searched <= n_base else "seed",
                            "nodes": t.nodes_count,
                            "fit_prediction_sha256": pred_hash,
                            "genotype_sha256": hashlib.sha256(key_bytes).hexdigest(),
                            "native_fit_brier": fit_loss,
                            "program": serialize(t.root),
                        }
                    )
                if key not in structures:
                    structures.add(key)
                    v = old.half_brier(
                        t.do_pred_on_another_tensors(current_tensors=vx), vy
                    )
                    if champion is None or fit_loss < champion[0]:
                        champion = fit_loss, t.copy()
                    if chosen is None or (v, fit_loss) < chosen[:2]:
                        chosen = v, fit_loss, t.copy()
                if searched in (initial_count, 600, 1200):
                    checkpoints = {
                        "checkpoint": "initial"
                        if searched == initial_count
                        else str(searched),
                        "generation": generation,
                        "calls": searched,
                        "fit_calls": searched,
                        "validation_calls": len(structures),
                        "archive_trees": len(structures),
                        "path_digest": path.hexdigest(),
                        "archive_unique_fit_predictions": len(fingerprints),
                        "choices": {},
                    }
                    for mode, entry in (("fit", champion), ("validation", chosen)):
                        winner = entry[-1]
                        checkpoints["choices"][mode] = {
                            "program": serialize(winner.root),
                            "nodes": winner.nodes_count,
                            "archive_selection_brier": entry[0],
                            "splits": {
                                s: old.eval_split(winner, *block, classes)
                                for s, block in splits.items()
                            },
                        }
                    snapshots.append(checkpoints)
                if searched == 1200:
                    raise Stop
            return np.array(scores)

    with TemporaryDirectory() as temp:
        base = Path(temp)
        providers = base / "preds"
        providers.mkdir()
        for name_col, col in x.items():
            np.save(providers / name_col, np.asarray(col, dtype=np.float64))
        gt = base / "y.npy"
        np.save(gt, y)
        pop = min(30, len(x))
        engine = Observed(
            providers,
            gt,
            pop,
            3,
            min(5, pop),
            backend="numpy",
            seed=gp_seed,
            allowed_ops=old.NATIVE_OPS,
            objective_functions=(objective,),
            minimize_node_count=True,
            operator_point_mutation=False,
            reuse_survivor_fitness=False,
            postprocessing_function=np.asarray
            if classes == 2
            else scale_vector_to_sum_1,
        )
        n_base = len(engine.population)
        attach_extra_trees(engine, gp_seed, arm)
        initial_count = len(engine.population)
        try:
            while True:
                generation += 1
                engine.run_iteration()
        except Stop:
            pass
    diversity = {
        "initial_trees": initial_count,
        "initial_singletons": n_base,
        "initial_seed_trees": initial_count - n_base,
        "initial_unique_genotypes": len({r["genotype_sha256"] for r in initial_rows}),
        "initial_unique_fit_predictions": len(
            {r["fit_prediction_sha256"] for r in initial_rows}
        ),
        "seed_unique_fit_predictions": len(
            {r["fit_prediction_sha256"] for r in initial_rows[n_base:]}
        ),
    }
    return {
        "job": job,
        "okapi_sha": PIN,
        "initial": initial_rows,
        "diversity": diversity,
        "checkpoints": snapshots,
        "classes": classes,
        "providers": len(x),
    }


def load_journal(journal):
    rows = {}
    if journal.exists():
        with journal.open() as f:
            for line in f:
                row = json.loads(line)
                key = tuple(row["job"])
                check(
                    row["okapi_sha"] == PIN and key not in rows, "invalid journal", key
                )
                rows[key] = row
    return rows


def collect(journal):
    logger.disable("okapi")
    done = load_journal(journal)
    with journal.open("a") as f:
        for job in jobs():
            if job not in done:
                f.write(
                    json.dumps(replay(job), allow_nan=False, separators=(",", ":"))
                    + "\n"
                )
                f.flush()


def measure_metric(name, actual, expected, errors):
    gap = abs(float(actual) - expected)
    check(gap <= 1e-14, name, actual, expected)
    errors[name] = max(gap, errors[name])


def compare(journal, folder):
    recorded = load_journal(journal)
    check(
        set(recorded) == set(jobs()) and len(recorded) == 63,
        "incomplete journal",
        len(recorded),
    )
    observed_initial = set()
    observed_scores = set()
    counts, errors = Counter(), defaultdict(float)

    def records(table):
        with gzip.open(folder / table, "rt", newline="") as f:
            for row in csv.DictReader(f):
                if row["method"] in ARMS:
                    key = (
                        row["scenario"],
                        int(row["data_seed"]),
                        int(float(row["gp_seed"])),
                        row["method"],
                    )
                    if key in recorded:
                        yield key, row, recorded[key]

    for job, r, record in records("standard-initial.csv.gz"):
        index = int(r["index"])
        tag = (*job, index)
        check(tag not in observed_initial, "duplicate initial", tag)
        observed_initial.add(tag)
        w = record["initial"][index]
        for field in ("origin", "fit_prediction_sha256", "genotype_sha256"):
            check(r[field] == w[field], tag, field)
        check(
            int(r["nodes"]) == w["nodes"] and json.loads(r["program"]) == w["program"],
            tag,
            "initial program",
        )
        measure_metric(
            "native_fit_brier", r["native_fit_brier"], w["native_fit_brier"], errors
        )
    for job, r, record in records("standard.csv.gz"):
        tick, mode, split = r["checkpoint"], r["selection"], r["split"]
        tag = (*job, tick, mode, split)
        check(tag not in observed_scores, "duplicate winner", tag)
        observed_scores.add(tag)
        cp = next(v for v in record["checkpoints"] if v["checkpoint"] == tick)
        winner = cp["choices"][mode]
        metrics = winner["splits"][split]
        for field, value in (
            record["diversity"]
            | {
                key: cp[key]
                for key in (
                    "generation",
                    "calls",
                    "fit_calls",
                    "validation_calls",
                    "archive_trees",
                    "archive_unique_fit_predictions",
                )
            }
            | {
                "nodes": winner["nodes"],
                "rows": metrics["rows"],
                "classes": record["classes"],
                "providers": record["providers"],
            }
        ).items():
            check(int(float(r[field])) == value, tag, field, r[field], value)
        check(r["path_digest"] == cp["path_digest"], tag, "path digest")
        check(json.loads(r["program"]) == winner["program"], tag, "winner program")
        for field in ("prediction_sha256", "oracle_sha256"):
            check(r[field] == metrics[field], tag, field)
        measure_metric(
            "archive_selection_brier",
            r["archive_selection_brier"],
            winner["archive_selection_brier"],
            errors,
        )
        for field in (
            "brier",
            "conditional_excess",
            "macro_f1",
            "log_loss",
            "accuracy",
        ):
            if metrics[field] is not None:
                measure_metric(field, r[field], metrics[field], errors)
                counts["metrics"] += 1
            else:
                check(not r[field], tag, field)
    expected_initial = {
        (*j, i) for j, bank in recorded.items() for i in range(len(bank["initial"]))
    }
    expected_records = {
        (*j, cp["checkpoint"], mode, split)
        for j, b in recorded.items()
        for cp in b["checkpoints"]
        for mode, w in cp["choices"].items()
        for split in w["splits"]
    }
    check(observed_initial == expected_initial, "missing initial rows")
    check(observed_scores == expected_records, "missing winner rows")
    return {
        "okapi_sha": PIN,
        "scenarios": len(SCENARIOS),
        "trajectories": 63,
        "objective_calls": 63 * 1200,
        "initial_programs": len(observed_initial),
        "path_digests": 63 * 3,
        "champions": 63 * 3 * 2,
        "prediction_hashes": len(observed_scores),
        "metrics": counts["metrics"],
        "maximum_absolute_error": dict(errors),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "journal", type=Path, help="external incremental JSONL for the 63 runs"
    )
    parser.add_argument(
        "--compare",
        type=Path,
        help="folder containing the two complete standard result tables",
    )
    args = parser.parse_args()
    ensure_revision()
    if args.compare:
        print(json.dumps(compare(args.journal, args.compare), indent=2, sort_keys=True))
    else:
        collect(args.journal)


if __name__ == "__main__":
    main()

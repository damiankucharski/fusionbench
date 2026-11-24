"""Replay native stock and retype without the experimental mutation mixin."""

import argparse
import hashlib
import itertools
import json
import struct
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from okapi import Okapi
from okapi.functions import scale_vector_to_sum_1

from fusionbench.eval.full_suite_point import data_for
from fusionbench.eval.opportunity_wine import build_wine_fold
from fusionbench.scenarios.registry import STANDARD_V1_SCENARIOS as SCENARIOS

# Use the independently built native seeds, input binding and loss from the previous experiment.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "full-suite-point"))
from verify_trajectories import NATIVE_OPS, add_shallow_trees, bind, half_brier


@lru_cache(maxsize=1)
def training(name, bank):
    if name == "wine":
        block = build_wine_fold(bank).fit
        return block.preds, block.y
    _, splits = data_for(name, bank)
    inputs, labels, _ = splits["fit"]
    return inputs, labels


def run(job):
    name, bank, start, arm = job
    raw, y = training(name, bank)
    classes = raw.shape[-1]
    providers = bind(raw)
    signed = hashlib.sha256()
    seen = set()
    count = 0
    generation = 0
    best = np.inf
    checkpoints = {}
    batch_losses = []

    def objective(p, y):
        loss = half_brier(p, y.ravel())
        batch_losses.append(loss)
        return 1 - loss

    def snapshot():
        return {
            "path_digest": signed.hexdigest(),
            "calls": count,
            "generation": generation,
            "archive_trees": len(seen),
            "raw_fit": best,
        }

    class Watch(Okapi):
        def _calculate_fitnesses(self, trees=None):
            nonlocal best, count
            cohort = self.population if trees is None else trees
            batch_losses.clear()
            result = super()._calculate_fitnesses(cohort)
            for tree, loss in zip(cohort, batch_losses, strict=True):
                key = tree.structural_key()
                signed.update((repr(key) + "\n").encode())
                signed.update(struct.pack("<d", loss))
                seen.add(key)
                best = min(loss, best)
                count += 1
                if count == 1200:
                    checkpoints["calls"] = snapshot()
            return result

    with TemporaryDirectory() as tmp:
        directory = Path(tmp) / "providers"
        directory.mkdir()
        for provider, tensor in providers.items():
            np.save(directory / provider, tensor)
        labels_file = Path(tmp) / "labels.npy"
        np.save(labels_file, y)
        pop = min(30, len(providers))
        engine = Watch(
            directory,
            labels_file,
            pop,
            3,
            min(5, pop),
            backend="numpy",
            seed=start,
            objective_functions=(objective,),
            allowed_ops=NATIVE_OPS,
            minimize_node_count=True,
            operator_point_mutation=arm == "retype",
            reuse_survivor_fitness=False,
            postprocessing_function=np.asarray
            if classes == 2
            else scale_vector_to_sum_1,
        )
        add_shallow_trees(engine, start)
        while len(checkpoints) < 2:
            generation += 1
            engine.run_iteration()
            if generation == 10:
                checkpoints["generation"] = snapshot()
    return {
        "scenario": name,
        "data_seed": bank,
        "gp_seed": start,
        "method": arm,
        "checkpoints": checkpoints,
    }


def jobs():
    # Keep adjacent tasks on the same Wine fold for cached independent provider refits.
    synthetic = itertools.product(
        SCENARIOS, range(5100, 5106), (661, 673), ("stock", "retype")
    )
    external = itertools.product(("wine",), range(5), (661, 673), ("stock", "retype"))
    return itertools.chain(external, synthetic)


def collect(journal, workers):
    keys = lambda r: (r["scenario"], r["data_seed"], r["gp_seed"], r["method"])
    existing = (
        {keys(json.loads(line)) for line in journal.read_text().splitlines()}
        if journal.exists()
        else set()
    )
    missing = [j for j in jobs() if j not in existing]
    with journal.open("a") as out, ProcessPoolExecutor(max_workers=workers) as pool:
        for record in pool.map(run, missing):
            out.write(json.dumps(record) + "\n")
            out.flush()


def verify(journal, folder):
    raw = pd.concat(
        [
            pd.read_csv(folder / name, float_precision="round_trip")
            for name in ("runs.csv.gz", "wine-runs.csv.gz")
        ]
    )
    rows = raw[
        raw.method.isin(("stock", "retype"))
        & raw.split.eq("fit")
        & raw.selection.eq("fit")
    ]
    lookup = rows.set_index(
        ["scenario", "data_seed", "gp_seed", "method", "checkpoint"],
        verify_integrity=True,
    )
    expected = set(jobs())
    checked = 0
    fit_calls = 0
    for line in journal.read_text().splitlines():
        r = json.loads(line)
        key = tuple(r[k] for k in ("scenario", "data_seed", "gp_seed", "method"))
        expected.remove(key)
        fit_calls += max(v["calls"] for v in r["checkpoints"].values())
        for stage, wanted in r["checkpoints"].items():
            actual = lookup.loc[(*key, stage)]
            for metric in ("calls", "generation", "path_digest", "archive_trees"):
                assert actual[metric] == wanted[metric], (
                    key,
                    stage,
                    metric,
                    actual[metric],
                    wanted[metric],
                )
            assert abs(actual.brier - 2 * wanted["raw_fit"]) < 5e-12
            checked += 1
    assert not expected and checked == len(lookup)
    report = {
        "unmodified_stock_and_retype_paths": checked // 2,
        "checkpoints": checked,
        "verified_archived_training_evaluations_through_later_endpoint": fit_calls,
    }
    (folder / "native-verified.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("journal", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    if args.compare:
        verify(args.journal, args.compare)
    else:
        collect(args.journal, args.workers)

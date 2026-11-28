"""Trace native GP proposals, scored batches and survivors without audit access."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from fusionbench.eval.native_switch import serialize
from fusionbench.eval.suite_point_gp import _bind
from fusionbench.methods.averaging import SimpleAverage, TopN
from fusionbench.methods.okapi import OKAPIMethod, _task_postprocessing
from fusionbench.methods.stacking import StackingLR
from fusionbench.scenarios.registry import get_scenario

CONDITIONS = ("redundant_null", "redundant_positive", "committee", "native_switch")
OBJECTIVES = ("brier", "paired_se")
GENERATIONS = 6
CALL_BUDGET = 240


def draw(condition, bank):
    if condition.startswith("redundant"):
        scenario = get_scenario(
            "redundant_pool",
            n_candidates=31,
            positive_control=condition == "redundant_positive",
            role_seed=48100 + bank,
        )
    elif condition == "native_switch":
        scenario = get_scenario(condition, role_seed=48100 + bank)
    else:
        scenario = get_scenario(condition)
    splits = tuple(
        scenario.sample_split(n, np.random.default_rng(seed + bank), domain)
        for n, seed, domain in [
            (48, 49100, "fit"),
            (1024, 50100, "fit"),
            (2048, 51100, "test"),
        ]
    )
    return scenario, splits


def row_score(loss, pool_loss, policy):
    risk = loss.mean()
    if policy == "paired_se":
        risk += np.std(loss - pool_loss, ddof=1) / np.sqrt(len(loss))
    return float(np.clip(1 - risk, 0, 1))


class Trace:
    """Only support tensors and immutable native keys are needed during evolution."""

    def __init__(self):
        self.key_to_id = {}
        self.trees = []
        self.snapshots = []
        self.scored = []
        self.fitness = {}
        self.proposals = []

    def ids(self, trees):
        result = []
        for tree in trees:
            key = tree.structural_key()
            if key not in self.key_to_id:
                self.key_to_id[key] = len(self.trees)
                self.trees.append(tree)
            result.append(self.key_to_id[key])
        return result

    def snapshot(self, population):
        active = self.ids(population)
        self.snapshots.append(
            {"active": active, "offered": self.proposals, "calls": len(self.scored)}
        )


def search(fit_x, fit_y, seed, policy, *, replay=False):
    """No audit or test inputs are accepted. Replay is an observer-free native run."""
    from okapi import Okapi

    tape = Trace()
    pool_loss = (fit_x[:, :, 1].mean(axis=1) - fit_y) ** 2

    def objective(p, gt):
        loss = (p[:, 0] - np.asarray(gt).ravel()) ** 2
        return row_score(loss, pool_loss, policy)

    class Observed(Okapi):
        def _mutate_additional_population(self):
            count = super()._mutate_additional_population()
            # Native then removes structural duplicates, reevaluates and truncates.
            tape.proposals = tape.ids(self.additional_population)
            return count

        def _calculate_fitnesses(self, trees=None):
            trees = self.population if trees is None else trees
            scores = super()._calculate_fitnesses(trees)
            ids = tape.ids(trees)
            tape.scored.extend(ids)
            for i, value in zip(ids, scores[:, 0], strict=True):
                old = tape.fitness.setdefault(i, float(value))
                assert old == value
            return scores

    with _task_postprocessing("binary"):
        adapter = OKAPIMethod(seed=seed, fitness="brier")
        try:
            inputs = adapter._write_split(fit_x, "fit")
            labels = Path(adapter._tmp.name) / "gt.npy"
            np.save(labels, fit_y)
            n_models = min(fit_x.shape[1], adapter.population_size)
            gp = (Okapi if replay else Observed)(
                inputs,
                labels,
                n_models,
                adapter.population_multiplier,
                min(adapter.tournament_size, n_models),
                seed=seed,
                backend="numpy",
                objective_functions=(objective,),
            )
            adapter._seed_population(gp, np.random.default_rng(seed))
            # The stock iteration does this if fitnesses is None. Calling it here
            # gives generation zero its actual support fitness and consumes no RNG.
            gp.fitnesses = gp._selection_fitnesses(gp.population)
            if replay:
                keys = []
            for generation in range(GENERATIONS + 1):
                if generation:
                    gp.run_iteration()
                if replay:
                    keys.append(tuple(t.structural_key() for t in gp.population))
                else:
                    tape.snapshot(gp.population)
                    # Confirm the whole native survivor fitness vector and ordering.
                    expected = [tape.fitness[i] for i in tape.snapshots[-1]["active"]]
                    np.testing.assert_array_equal(
                        np.round(expected, 3), gp.fitnesses[:, 0]
                    )
        finally:
            adapter._tmp.cleanup()
    if replay:
        return keys
    # All structural mutations returned new trees; previously observed references
    # still identify exactly the programs first offered.
    assert all(
        tape.trees[i].structural_key() == key for key, i in tape.key_to_id.items()
    )
    assert set(tape.fitness) == set(range(len(tape.trees)))
    assert len(tape.scored) >= CALL_BUDGET
    return tape


def full_brier(probability, labels):
    return float(2 * np.mean((probability - labels) ** 2))


def evaluate(tape, splits):
    # Rebind on support+audit+test only after the last native selection has finished.
    inputs = _bind(np.concatenate([part[0] for part in splits]))
    cuts = np.cumsum([len(part[1]) for part in splits])[:-1]
    records = []
    with _task_postprocessing("binary"):
        for i, tree in enumerate(tape.trees):
            pred = tree.do_pred_on_another_tensors(current_tensors=inputs)
            p = np.asarray(pred).ravel()
            assert np.all((p >= 0) & (p <= 1))
            train, audit, test = np.split(p, cuts)
            # A fixed large audit input bank for semantic diversity (no labels).
            signature = hashlib.sha256(
                np.round(audit, 10).astype("<f8").tobytes()
            ).hexdigest()
            records.append(
                {
                    "support": full_brier(train, splits[0][1]),
                    "audit": full_brier(audit, splits[1][1]),
                    "test": full_brier(test, splits[2][1]),
                    "fitness": tape.fitness[i],
                    "nodes": tree.nodes_count,
                    "behavior": signature,
                    "program": serialize(tree.root),
                }
            )
    return records


def summarize_pool(ids, records):
    def pick(key):
        return min(ids, key=lambda i: (key(records[i]), records[i]["nodes"]))

    # min breaks any remaining ties by list order (stable, including duplicates).
    champions = {
        "champion": pick(lambda r: -np.round(r["fitness"], 3)),
        "audit_pick": pick(lambda r: r["audit"]),
        "test_oracle": pick(lambda r: r["test"]),
    }
    row = {
        "count": len(ids),
        "genotypes": len(set(ids)),
        "behaviors": len({records[i]["behavior"] for i in ids}),
        "median_test": float(np.median([records[i]["test"] for i in ids])),
        "median_nodes": float(np.median([records[i]["nodes"] for i in ids])),
    }
    for label, i in champions.items():
        row[label + "_id"] = i
        for metric in ("support", "audit", "test", "nodes"):
            row[label + "_" + metric] = records[i][metric]
    return row


def run_case(args):
    from loguru import logger

    logger.disable("okapi")
    condition, bank, check = args
    scenario, splits = draw(condition, bank)
    x, y, _ = splits[0]
    prefix = {"condition": condition, "bank": bank}
    control_rows = []
    for method in (SimpleAverage(), TopN(metric="brier"), StackingLR()):
        method.fit(x, y, task="binary", n_classes=2)
        values = [full_brier(method.predict(a)[:, 1], b) for a, b, _ in splits]
        control_rows.append(
            {
                **prefix,
                "method": method.name,
                **dict(zip(("support", "audit", "test"), values, strict=True)),
            }
        )
    values = [
        full_brier(scenario.bayes_oracle(a, latent)[:, 1], b) for a, b, latent in splits
    ]
    control_rows.append(
        {
            **prefix,
            "method": "posterior",
            **dict(zip(("support", "audit", "test"), values, strict=True)),
        }
    )
    traces = []
    rows = []
    for policy in OBJECTIVES:
        tape = search(x, y, 52100 + bank, policy)
        if check and bank == 0:
            plain = search(x, y, 52100 + bank, policy, replay=True)
            observed = [
                tuple(tape.trees[i].structural_key() for i in snap["active"])
                for snap in tape.snapshots
            ]
            assert observed == plain, "Observer altered a native survivor trajectory"
        records = evaluate(tape, splits)
        meta = {**prefix, "objective": policy}
        archive = {}
        for generation, snap in enumerate(tape.snapshots):
            archive.update(
                dict.fromkeys(snap["active"] if generation == 0 else snap["offered"])
            )
            assert set(snap["active"]).issubset(archive)
            # Every distinct proposed individual entered an actual scoring batch,
            # either in this generation or as an already scored structural duplicate.
            assert set(archive) == set(tape.scored[: snap["calls"]])
            for label, ids in [("active", snap["active"]), ("archive", list(archive))]:
                rows.append(
                    {
                        **meta,
                        "generation": generation,
                        "pool": label,
                        "calls": snap["calls"],
                        **summarize_pool(ids, records),
                    }
                )
        visit_ids = list(dict.fromkeys(tape.scored[:CALL_BUDGET]))
        rows.append(
            {
                **meta,
                "generation": -1,
                "pool": "archive240",
                "calls": CALL_BUDGET,
                **summarize_pool(visit_ids, records),
            }
        )
        traces.append(
            {
                **meta,
                "programs": records,
                "generations": tape.snapshots,
                "native_score_visits": tape.scored,
            }
        )
    return control_rows, rows, traces


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--banks", type=int, default=8)
    parser.add_argument(
        "--verify-observer",
        action="store_true",
        help="Replay bank zero on plain OKAPI in each condition/objective",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    tasks = [
        (c, i, args.verify_observer) for c in CONDITIONS for i in range(args.banks)
    ]
    controls, stages = [], []
    with (
        (args.output / "traces.jsonl.gz").open("wb") as raw,
        gzip.GzipFile(
            fileobj=raw,
            filename="",
            mode="wb",
            mtime=0,
        ) as traces,
        ProcessPoolExecutor(max_workers=args.workers) as pool,
    ):
        for control, stage, runs in pool.map(run_case, tasks):
            controls.extend(control)
            stages.extend(stage)
            for run in runs:
                traces.write(
                    (
                        json.dumps(run, separators=(",", ":"), allow_nan=False) + "\n"
                    ).encode()
                )
            print(runs[-1]["condition"], runs[-1]["bank"], flush=True)
    write_csv(args.output / "controls.csv", controls)
    write_csv(args.output / "stages.csv", stages)


if __name__ == "__main__":
    main()

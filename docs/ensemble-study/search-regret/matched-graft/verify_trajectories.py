"""Reproduce matched-parent runs with native batched fitness and no lineage markers.

Run from the FusionBench checkout on the recorded OKAPI commit. The batched
objective observer is shared with the preceding independent verification. This
script uses its own admission and fork implementation. Saved graft parameters
are replay inputs; the proposal optimizer itself is not run.
"""

import argparse
import copy
import csv
import gzip
import hashlib
import json
import runpy
import struct
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import okapi
from loguru import logger
from okapi.population import choose_pareto_then_proximity
from okapi.random_state import use_rng

from fusionbench.eval.native_switch import serialize
from fusionbench.scenarios.native_switch import draw_native_switch

previous = (
    Path(__file__).resolve().parents[1] / "native-graft" / "verify_trajectories.py"
)
Replay = runpy.run_path(str(previous))["Replay"]
PIN = "4c784f6a217675d1305d378eaeb525738c37761d"
OFFERS = ("fit_pair_fit_cut", "fit_pair_half", "duplicate_fit_cut", "duplicate_half")
BRANCHES = (*OFFERS, "reorder", "archive_only")
PLAIN = ("plain_tail", "plain_forward_tail", "plain_candidate_tail")


def integer(text):
    value = float(text)
    assert value.is_integer()
    return int(value)


def rng_digest(state):
    algorithm, vector, index, has_spare, spare = state
    h = hashlib.sha256(algorithm.encode())
    h.update(np.asarray(vector, dtype="<u4").tobytes())
    h.update(struct.pack("<iid", index, has_spare, spare))
    return h.hexdigest()


def parent_state(trees, scores):
    h = hashlib.sha256()
    hashes = []
    for tree in trees:
        key = tree.structural_key()
        hashes.append(hashlib.sha256(repr(key).encode()).hexdigest())
        nodes = tree.root.get_nodes()
        # Native selection indexes each tree's stored node lists. Their order
        # can differ from breadth-first even for the same expression.
        positions = {id(node): ix for ix, node in enumerate(nodes)}
        order = tuple(
            tuple(positions[id(node)] for node in tree.nodes[kind])
            for kind in ("value_nodes", "op_nodes")
        )
        h.update((repr((key, tree.mutation_chance, order)) + "\n").encode())
    h.update(np.asarray(scores, dtype="<f8").tobytes())
    return h.hexdigest(), tuple(hashes)


def fork(source, archive):
    child = copy.copy(source)
    # deepcopy preserves node-list order, parameter types, mutation chance and
    # parent links. Native Tree.copy() rebuilds the two node lists breadth-first.
    child.population = copy.deepcopy(source.population)
    child.fitnesses = source.fitnesses.copy()
    child.rng = np.random.RandomState(0)
    child.rng.set_state(source.rng.get_state())
    child.additional_population = []
    child.losses = []
    child.objective_functions = (child.objective,)
    child.archive = archive.copy()
    child.trace = source.trace.copy()
    child.records = {}
    child.record_at = set()
    child.stop = None
    return child


def reconstruct(engine, winner, published):
    offers = {}
    shared = engine.archive.copy()
    for name in OFFERS:
        r = published["immediate_" + name, "fit"]
        operand = r["chosen_partner"]
        if operand and name.startswith("duplicate"):
            operand = r["anchor"]
        cut = 0.5 if name.endswith("half") else r["fitted_threshold"]
        # restore_offer appends exactly this native operator and scores the
        # rebuilt program outside the counted trajectory. It also compares
        # the full expression to the serialized final program.
        tree, entry = engine.restore_offer(
            winner,
            {
                "selected_operand": operand,
                "site_path": r["site_path"],
                "anchor": r["anchor"],
                "threshold": cut,
                "direction": r["direction"],
                "program": r["program"],
                "graft_fit_loss": r["observed_brier"],
            },
        )
        assert entry.validation == float(r["validation_observed_brier"])
        offers[name] = tree, entry.loss
        shared.setdefault(tree.structural_key(), entry)
    return offers, shared


def decide(engine, incumbent, tree, loss):
    parents = engine.population
    rng_copy = np.random.RandomState(0)
    rng_copy.set_state(engine.rng.get_state())
    identity_order = list(range(len(parents)))

    def unchanged(status):
        return status, identity_order, None, None, engine.rng.get_state()

    if tree.structural_key() == incumbent.structural_key():
        return unchanged("unchanged")
    if any(parent.structural_key() == tree.structural_key() for parent in parents):
        return unchanged("duplicate_parent")
    augmented = [*parents, tree]
    scores = np.vstack([engine.fitnesses, np.round(1.0 - loss, 3)])
    with use_rng(rng_copy):
        kept, _ = choose_pareto_then_proximity(
            augmented,
            scores,
            engine.objectives,
            len(parents),
            engine.minimize_node_count,
        )
    ix = {id(t): i for i, t in enumerate(augmented)}
    selected = [ix[id(t)] for t in kept]
    if len(parents) not in selected:
        assert set(selected) == set(identity_order)
        return unchanged("rejected")
    slot = selected.index(len(parents))
    [displaced] = set(identity_order).difference(selected)
    selected[slot] = -1
    return "admitted", selected, slot, displaced, rng_copy.get_state()


def customize(branch, name, selection, offers):
    status, order, _, displaced, post_rng = selection
    if status != "admitted" or name == "archive_only":
        return None
    old, old_fitness = branch.population, branch.fitnesses
    if name == "reorder":
        replacement, fitness = old[displaced], old_fitness[displaced]
    else:
        tree, loss = offers[name]
        replacement = copy.deepcopy(tree)
        fitness = np.asarray([1.0 - loss]).round(3)
    branch.population = [replacement if i == -1 else old[i] for i in order]
    branch.fitnesses = np.stack([fitness if i == -1 else old_fitness[i] for i in order])
    branch.rng.set_state(post_rng)
    return float(fitness[0])


def verify_winners(snap, name, saved, stats):
    for selection, actual in (("fit", snap.fit), ("validation", snap.validation)):
        r = saved[name, selection]
        assert snap.sha == r["path_sha256"], (name, selection, "ordered native scores")
        assert snap.size == integer(r["archive_trees"]), (name, "archive size")
        assert snap.step == integer(r["calls"]), (name, "calls")
        assert serialize(actual.tree.root) == json.loads(r["program"]), (
            name,
            selection,
            "program",
        )
        assert actual.loss == float(r["observed_brier"]), (name, selection, "fit loss")
        assert actual.validation == float(r["validation_observed_brier"]), (
            name,
            selection,
            "validation loss",
        )
        stats["exact_winners"] += 1
    stats["exact_checkpoints"] += 1


def check_ordered_fields(r, parents, scores, digests):
    assert tuple(json.loads(r["state_parent_keys"])) == digests
    assert tuple(json.loads(r["state_parent_fitness"])) == tuple(
        float(row[0]) for row in scores
    )
    assert tuple(json.loads(r["state_parent_mutation_chances"])) == tuple(
        t.mutation_chance for t in parents
    )


def replay_bank(paths, gt, validation, validation_y, seed, saved):
    stats = Counter()
    engine = Replay(paths, gt, validation, validation_y, seed)
    base = engine.initial_boundary()
    r0 = saved["gp_initial", "fit"]
    verify_winners(base, "gp_initial", saved, stats)
    assert r0["base_generation"] and base.generation == integer(r0["base_generation"])
    assert base.step == integer(r0["base_calls"])
    stats["boundaries"] += 1
    offers, shared = reconstruct(engine, base.fit, saved)
    stats["reconstructed_offers"] += len(offers)
    choices = {}
    for name, (tree, loss) in offers.items():
        choices[name] = decide(engine, base.fit.tree, tree, loss)
        assert choices[name][0] == saved["immediate_" + name, "fit"]["free_admission"]
        stats["shadow_admissions"] += 1
    chosen = choices[OFFERS[0]]
    status, order, slot, displaced, post_rng = chosen
    before = rng_digest(engine.rng.get_state())
    after = rng_digest(post_rng)
    assert r0["status"] == status
    assert (r0["admitted"] == "True") == (status == "admitted")
    assert json.loads(r0["parent_order"]) == order
    assert (integer(r0["slot"]) if r0["slot"] else None) == slot
    assert (integer(r0["displaced"]) if r0["displaced"] else None) == displaced
    assert r0["rng_before"] == before and r0["rng_after_admission"] == after

    charged = integer(r0["graft_forwards"]) + integer(r0["variant_forwards"])
    tested = integer(r0["threshold_candidates"])
    tail = base.step + 300
    budgets = (tail, tail + charged, tail + charged + tested)
    plain = fork(engine, engine.archive)
    plain.run_to(budgets)
    for name, cut in zip(PLAIN, budgets, strict=True):
        verify_winners(plain.records[cut], name, saved, stats)
    stats["unintervened_calls"] += max(budgets)
    stats["status_" + status] += 1

    for name in BRANCHES:
        r = saved[name, "fit"]
        arm = fork(engine, shared)
        slot_fitness = customize(arm, name, chosen, offers)
        assert r["state_rng_before"] == before
        assert r["state_rng_after"] == rng_digest(arm.rng.get_state())
        population, hashes = parent_state(arm.population, arm.fitnesses)
        assert population == r["state_population_digest"], (
            name,
            "population with node indices",
        )
        check_ordered_fields(r, arm.population, arm.fitnesses, hashes)
        assert integer(r["state_duplicate_structures"]) == len(hashes) - len(
            set(hashes)
        )
        assert slot_fitness == (
            float(r["state_slot_fitness"]) if r["state_slot_fitness"] else None
        )
        arm.run_to([tail])
        verify_winners(arm.records[tail], name, saved, stats)
        assert r["state_rng_final"] == rng_digest(arm.rng.get_state()), (
            name,
            "final RNG",
        )
        if status != "admitted" or name == "archive_only":
            assert arm.records[tail].sha == plain.records[tail].sha
        stats["checked_tails"] += 1
        stats["ordinary_tail_calls"] += tail - base.step
        stats["ordered_parent_records"] += len(hashes)
    return stats


def check_group(item):
    (seed, regime, gp_seed), rows = item
    fit, val = [draw_native_switch(seed, i, n) for i, n in enumerate((512, 256))]
    draw_seed = int(np.random.SeedSequence([seed, gp_seed, 937]).generate_state(1)[0])
    assert all(integer(r["graft_seed"]) == draw_seed for r in rows.values())
    if regime == "close":
        fit, val = [replace(s, close=True, threshold=37 / 64) for s in (fit, val)]
    mix = int(regime == "linear")
    val_x = {f"p{i}.npy": val.preds[:, i, 1:2] for i in range(6)}
    with TemporaryDirectory() as tmp:
        paths = [Path(tmp) / f"p{i}.npy" for i in range(6)]
        for i, path in enumerate(paths):
            np.save(path, fit.preds[:, i, 1:2])
        gt = Path(tmp) / "labels.npy"
        np.save(gt, fit.labels(mix))
        try:
            return replay_bank(paths, gt, val_x, val.labels(mix), gp_seed, rows)
        except Exception as exc:
            exc.add_note(f"data_seed={seed}, regime={regime}, gp_seed={gp_seed}")
            raise


def read_data(path):
    groups = defaultdict(dict)
    val = {}
    with gzip.open(path, "rt") as f:
        for r in csv.DictReader(f):
            if not r["gp_seed"] or r["split"] not in ("fit", "validation"):
                continue
            case = integer(r["data_seed"]), r["fit_regime"], integer(r["gp_seed"])
            key = r["method"], r["decision"]
            if r["split"] == "fit":
                assert key not in groups[case]
                groups[case][key] = r
            else:
                assert (case, key) not in val
                val[case, key] = r["observed_brier"]
    expected = {
        (i, r, j)
        for i in range(4100, 4116)
        for r in ("far", "close", "linear")
        for j in (311, 313)
    }
    assert set(groups) == expected
    for case, rows in groups.items():
        for key, r in rows.items():
            r["validation_observed_brier"] = val.pop((case, key))
    assert not val
    return sorted(groups.items())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    root = Path(okapi.__file__).resolve().parent
    commit = subprocess.check_output(
        ["git", "-C", root, "rev-parse", "HEAD"], text=True
    ).strip()
    assert commit == PIN, (commit, PIN)
    assert not subprocess.check_output(
        ["git", "-C", root, "status", "--short", "--untracked-files=no"]
    )
    logger.disable("okapi")
    groups = read_data(args.directory / "runs.csv.gz")
    results = Counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for checked in pool.map(check_group, groups):
            results.update(checked)
    print(json.dumps({"okapi": commit, **dict(sorted(results.items()))}, indent=2))


if __name__ == "__main__":
    main()

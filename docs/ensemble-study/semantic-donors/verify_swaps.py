"""Force native positioned swaps in preselected recorded populations and verify chances."""

import argparse
import gzip
import hashlib
import itertools
import json
from collections import Counter
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path
from runpy import run_path

import numpy as np
import pandas as pd

from fusionbench.eval.donor_crossover import analyze_population

LEVELS = np.array([0.1, 0.35, 0.65, 0.9])
RANKS = np.asarray(list(itertools.product(LEVELS, repeat=4)))
CORNERS = ((np.arange(16)[:, None] >> np.arange(4)) & 1).astype(float)
X = np.vstack([CORNERS, RANKS])
MASKS = [((RANKS >= t) * (1 << np.arange(4))).sum(axis=1) for t in LEVELS[1:]]


def build(record):
    from okapi.node import MaxNode, MinNode, ValueNode
    from okapi.tree import Tree

    def node(r):
        children = list(map(node, r["children"]))
        if r["type"] == "ValueNode":
            i = int(r["id"][1:-4])
            return ValueNode(children, X[:, i : i + 1], r["id"])
        return {"MinNode": MinNode, "MaxNode": MaxNode}[r["type"]](children)

    return Tree(node(record))


def physical(tree):
    out = np.asarray(tree.predict(clear_cache=True)).ravel()
    corner = out[:16]
    assert np.isin(corner, [0, 1]).all()
    grades = sum(corner[mask].astype(int) for mask in MASKS)
    np.testing.assert_array_equal(LEVELS[grades], out[16:])
    return sum(1 << i for i in np.flatnonzero(corner).tolist())


def paths(node, prefix=()):
    yield prefix, node
    for k, child in enumerate(node.children):
        yield from paths(child, prefix + (k,))


def at(tree, path):
    cur = tree.root
    for i in path:
        cur = cur.children[i]
    return cur


def child(host, host_path, donor, donor_path):
    tree = host.copy()
    tree.replace_at(at(tree, host_path), at(donor, donor_path).copy_subtree())
    tree.update_nodes()
    return physical(tree)


def enumerate_swaps(trees):
    from okapi.node import ValueNode
    from okapi.tree import Tree

    sites, roots, ingredients = [], [], []
    for tree in trees:
        roots.append(physical(tree))
        val, op, vv = [], [], set()
        for path, node in paths(tree.root):
            if isinstance(node, ValueNode):
                val.append(path)
                vv.add(physical(Tree(node.copy_subtree())))
            else:
                op.append(path)
        sites.append((val, op))
        ingredients.append(vv)
    distributions = {}
    swaps = 0
    for i, j in itertools.combinations_with_replacement(range(len(trees)), 2):
        groups = [(sites[i][0], sites[j][0])]
        if sites[i][1] and sites[j][1]:
            groups.append((sites[i][1], sites[j][1]))
        mass = Counter()
        for left, right in groups:
            probability = Fraction(1, len(groups) * len(left) * len(right))
            for a, b in itertools.product(left, right):
                first = child(trees[i], a, trees[j], b)
                second = child(trees[j], b, trees[i], a)
                mass[(first, second)] += probability
                swaps += 1
        assert sum(mass.values()) == 1
        distributions[i, j] = mass
    return distributions, roots, ingredients, swaps


def join_counts(values, roots, targets):
    output = {}
    for prefix in ("", "parents_free_"):
        for strict in (False, True):
            for kind in ("union", "intersection"):
                winning = set()
                for i, j in itertools.combinations(range(len(roots)), 2):
                    if prefix and (roots[i] in targets or roots[j] in targets):
                        continue
                    for a, b in itertools.product(values[i], values[j]):
                        if strict and (a in targets or b in targets):
                            continue
                        if ((a | b) if kind == "union" else (a & b)) in targets:
                            winning.add((i, j))
                output[prefix + ("strict_" if strict else "") + kind + "_pairs"] = len(
                    winning
                )
    return output


def expectation(physical_pairs, roots, subtrees, fitness, targets):
    n = len(roots)
    contestants = list(itertools.permutations(range(n), 2))
    wins = Counter(
        i if abs(1.0 - fitness[i]) <= abs(1.0 - fitness[j]) else j
        for i, j in contestants
    )
    w = [Fraction(wins[i], len(contestants)) for i in range(n)]
    pair = [[Fraction() for _ in range(n)] for _ in range(n)]
    for (i, j), hist in physical_pairs.items():
        chance = sum(
            (m for (a, b), m in hist.items() if a in targets or b in targets),
            Fraction(),
        )
        pair[i][j] = pair[j][i] = chance
    out = {
        "parent_signatures": roots,
        "parent_weights": list(map(float, w)),
        "per_pair": np.asarray(pair, float),
        "target_present": any(i in targets for i in roots),
        "available": any(c for row in pair for c in row),
        "new_available": any(
            pair[i][j]
            for i, j in itertools.product(range(n), repeat=2)
            if roots[i] not in targets and roots[j] not in targets
        ),
        "root_joins": join_counts([{v} for v in roots], roots, targets),
        "subtree_joins": join_counts(subtrees, roots, targets),
    }
    for name, weights in (("p_uniform", [Fraction(1, n)] * n), ("p_tournament", w)):
        overall = productive = free = Fraction()
        for i, j in itertools.product(range(n), repeat=2):
            mass = weights[i] * weights[j]
            overall += mass * pair[i][j]
            if roots[i] not in targets and roots[j] not in targets:
                free += mass
                productive += mass * pair[i][j]
        out.update(
            {
                name: float(overall),
                name + "_new": float(productive),
                name + "_parents_free": float(free),
                name + "_given_parents_free": float(productive / free)
                if free
                else None,
            }
        )
    return out


def audit(folder):
    from loguru import logger
    from okapi.globals import use_postprocessing_function

    logger.disable("okapi")
    tracepath = folder / "traces.jsonl.gz"
    focuspath = folder / "focal.csv.gz"
    focal = pd.read_csv(focuspath, float_precision="round_trip")
    prior = folder.parent / "selection-precision" / "verify_lattice.py"
    tables = run_path(str(prior))["signatures"]()
    indices = {tuple(x): i for i, x in enumerate(RANKS)}
    corners = [indices[tuple(LEVELS[np.asarray(o * 3, int)])] for o in CORNERS]
    semantic = [
        sum(1 << j for j, high in enumerate(r[corners] > 0.5) if high) for r in tables
    ]
    snapshots = swaps = targets_checked = fields = 0
    maximum_delta = 0.0

    def equal(a, b):
        nonlocal maximum_delta
        if isinstance(b, dict):
            assert a == b
        elif b is None:
            assert a is None or pd.isna(a)
        else:
            delta = np.max(np.abs(np.asarray(a, float) - np.asarray(b, float)))
            assert delta <= 2e-15, (a, b, delta)
            maximum_delta = max(maximum_delta, float(delta))

    with gzip.open(tracepath, "rt") as file, use_postprocessing_function(np.asarray):
        for line in file:
            trace = json.loads(line)
            if trace["data_seed"] not in (6600, 6601) or trace["gp_seed"] != 809:
                continue
            for g in trace["generations"]:
                if g["generation"] not in (1, 5, 10, 15):
                    continue
                snapshots += 1
                trees = [build(trace["programs"][i]) for i in g["parents"]]
                physical_pairs, roots, subs, count = enumerate_swaps(trees)
                swaps += count
                target_sets = {
                    "population": {semantic[trace["goals"]["population_rule"]]},
                    "erm": {semantic[i] for i in trace["goals"]["erm_rules"]},
                    "first_parent": {roots[0]},
                    "two_parents": set(roots[:2]),
                }
                for tag, targets in target_sets.items():
                    targets_checked += 1
                    expected = expectation(
                        physical_pairs, roots, subs, g["parent_fitness"], targets
                    )
                    observed = asdict(
                        analyze_population(trees, g["parent_fitness"], targets)
                    )
                    assert expected.keys() == observed.keys()
                    for k, v in expected.items():
                        equal(observed[k], v)
                    if tag not in ("population", "erm"):
                        continue
                    rows = focal[
                        (focal.law == trace["law"])
                        & (focal.data_seed == trace["data_seed"])
                        & (focal.gp_seed == 809)
                        & (focal.generation == g["generation"])
                        & (focal.target == tag)
                    ]
                    assert len(rows) == 1, (tag, focal.target.unique())
                    row = rows.iloc[0]
                    for k, v in expected.items():
                        if isinstance(v, dict):
                            equal({a: row[f"{k}_{a}"] for a in v}, v)
                        elif isinstance(v, (list, np.ndarray)):
                            equal(json.loads(row[k]), v)
                        else:
                            equal(row[k], v)
                        fields += 1
    assert snapshots == 16
    return {
        "trace_sha256": hashlib.sha256(tracepath.read_bytes()).hexdigest(),
        "focal_sha256": hashlib.sha256(focuspath.read_bytes()).hexdigest(),
        "data_seeds": [6600, 6601],
        "search_seed": 809,
        "generations": [1, 5, 10, 15],
        "snapshots": snapshots,
        "physically_swapped_site_pairs": swaps,
        "physically_scored_children": 2 * swaps,
        "population_target_checks": targets_checked,
        "archived_field_groups_checked": fields,
        "max_abs_probability_error": maximum_delta,
        "evaluation_rows_per_child": {"Boolean_corners": 16, "graded_atoms": 256},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    (args.folder / "swap-verification.json").write_text(
        json.dumps(audit(args.folder), indent=2) + "\n"
    )

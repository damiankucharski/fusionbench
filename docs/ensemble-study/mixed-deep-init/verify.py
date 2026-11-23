"""Independent audit of the frozen starter pairs, GP budgets and reported contrasts."""

import csv
import gzip
import json
import math
from collections import defaultdict, Counter
from pathlib import Path
from statistics import fmean, stdev

import numpy as np
from scipy.stats import t

from fusionbench.scenarios.registry import SCENARIOS
from fusionbench.eval.full_suite_point import SHIFTS

out = Path(__file__).resolve().parent
focal = "compositional_switch"
arms = ("stock", "mixed_branch", "mixed_chain")
baselines = ("mean", "top_n", "stack_logit")
heldonly = ("top_n_640", "stack_logit_640")


def read_csv(name):
    openr = gzip.open if name.endswith(".gz") else open
    with openr(out / name, "rt", newline="") as stream:
        return list(csv.DictReader(stream))


def arith(tree):
    # A ValueNode is reduced left to right by its ordered OperatorNode children.
    assert tree["type"] == "ValueNode"
    result = tree["id"]
    for op in tree["children"]:
        assert len(op["children"]) == 1 and op["type"] in ("MinNode", "MaxNode")
        result = result, op["type"], arith(op["children"][0])
    return result


def inorder(expr):
    if isinstance(expr, str):
        return [expr]
    return [*inorder(expr[0]), expr[1], *inorder(expr[2])]


def depth(expr, d=0):
    if isinstance(expr, str):
        return [d]
    return depth(expr[0], d + 1) + depth(expr[2], d + 1)


def walk(node):
    yield node
    for child in node["children"]:
        yield from walk(child)


with gzip.open(out / "trees.jsonl.gz", "rt") as stream:
    trees = [json.loads(line) for line in stream]
tgroups = defaultdict(dict)
for tree in trees:
    key = (tree["scenario"], tree["data_seed"], tree["gp_seed"])
    assert tree["method"] not in tgroups[key]
    tgroups[key][tree["method"]] = tree
    nodes = list(walk(tree["program"]))
    assert len(nodes) == tree["nodes"]
    assert (
        len({n["id"] for n in nodes if n["type"] == "ValueNode"}) == tree["providers"]
    )
    assert tree["calls"] == 600 and 0 < tree["validation_calls"] <= tree["calls"]
    assert len(tree["path_digest"]) == 64 and tree["seconds"] >= 0
    assert tree["method"] in arms


def gpseeds(bank):
    return (2 * bank + 17, 2 * bank + 18)


pairs = 0
tjobs = set()
expected_scores = set()
for scenario, cls in SCENARIOS.items():
    width = len(cls().model_ids())
    ids = sorted(f"m{i}.npy" for i in range(width))
    splits = ["fit", "validation", "iid"] + (["shift"] if scenario in SHIFTS else [])
    for bank in range(92300, 92308 if scenario == focal else 92304):
        for split in splits:
            for arm in baselines + (heldonly if split in ("iid", "shift") else ()):
                expected_scores.add((scenario, bank, None, arm, split))
        for seed in gpseeds(bank):
            key = (scenario, bank, seed)
            tjobs.add(key)
            group = tgroups[key]
            assert set(group) == set(arms)
            for arm in arms:
                for split in splits:
                    expected_scores.add((scenario, bank, seed, arm, split))
            assert all(len(group[a]["last_six"]) == 6 for a in arms)
            # Independently reconstruct the six ordinary final slots from all 30 random stock plans.
            rng = np.random.default_rng(seed)
            for indiv in range(30):
                k = rng.integers(2, min(width, 4) + 1)
                leaves = [ids[i] for i in rng.choice(width, size=k, replace=False)]
                op = ("MinNode", "MaxNode", "MeanNode")[rng.integers(3)]
                if indiv >= 24:
                    p = group["stock"]["last_six"][indiv - 24]
                    assert [
                        n["id"] for n in walk(p) if n["type"] == "ValueNode"
                    ] == leaves
                    assert [q["type"] for q in p["children"]] == [op]
                    assert len(p["children"][0]["children"]) == k - 1
            rng = np.random.default_rng(np.random.SeedSequence([seed, 8301]))
            for b, c in zip(
                group["mixed_branch"]["last_six"],
                group["mixed_chain"]["last_six"],
                strict=True,
            ):
                eb, ec = arith(b), arith(c)
                ib, ic = inorder(eb), inorder(ec)
                assert ib == ic
                assert depth(eb) == [3] * 8 and depth(ec) == [7, 7, 6, 5, 4, 3, 2, 1]
                assert len(ib) == 15 and len(list(walk(b))) == len(list(walk(c))) == 15
                pid, ops = ib[::2], ib[1::2]
                if width >= 8:
                    order = rng.choice(width, 8, replace=False)
                    assert len(set(pid)) == 8
                else:
                    order = rng.permutation(
                        [*range(width), *rng.choice(width, 8 - width)]
                    )
                    assert set(pid) == set(ids)
                internal = "MinNode" if rng.integers(2) else "MaxNode"
                external = "MaxNode" if internal == "MinNode" else "MinNode"
                assert pid == [ids[i] for i in order] and ops == [
                    internal,
                    external,
                    internal,
                    internal,
                    internal,
                    external,
                    internal,
                ]
                pairs += 1
assert set(tgroups) == tjobs and len(trees) == 480

scores = {}
groups = defaultdict(list)
for r in read_csv("scores.csv.gz"):
    key = (
        r["scenario"],
        int(r["data_seed"]),
        int(float(r["gp_seed"])) if r["gp_seed"] else None,
        r["method"],
        r["split"],
    )
    assert key not in scores and len(r["prediction_sha256"]) == 64
    val = float(r["brier"])
    assert 0 <= val <= 2
    scores[key] = val
    groups[key[0], key[1], key[4], key[3]].append(val)
assert set(scores) == expected_scores, (
    len(scores),
    len(expected_scores),
    set(scores) ^ expected_scores,
)
scenarios = defaultdict(dict)
gaps = defaultdict(list)
for (scenario, bank, split, method), values in groups.items():
    n = 2 if method in arms else 1
    assert len(values) == n
    scenarios[scenario, split].setdefault(method, []).append(fmean(values))
    if method == arms[1]:
        gaps[scenario, split].append(
            fmean(
                scores[scenario, bank, g, arms[1], split]
                - scores[scenario, bank, g, arms[2], split]
                for g in gpseeds(bank)
            )
        )
means = read_csv("scenario-means.csv")
assert len(means) == len(scenarios) == 66
for r in means:
    by_method = scenarios[r["scenario"], r["split"]]
    for m in arms + baselines + heldonly:
        if m not in by_method:
            assert not r[m]
        else:
            assert abs(fmean(by_method[m]) - float(r[m])) < 1e-13
paired = read_csv("paired.csv")
assert len(paired) == len(gaps) == 66
for r in paired:
    x = gaps[r["scenario"], r["split"]]
    n = len(x)
    assert int(r["count"]) == n
    m, half = fmean(x), stdev(x) * float(t.ppf(0.975, n - 1)) / math.sqrt(n)
    for label, num in [("mean", m), ("ci_low", m - half), ("ci_high", m + half)]:
        assert abs(float(r[label]) - num) < 1e-13
    if r["scenario"] == focal and r["split"] == "iid":
        print(
            "FOCAL",
            len(x),
            m,
            m - half,
            m + half,
            "wins",
            sum(v < 0 for v in x),
            [round(v, 6) for v in x],
        )
for split in ("iid", "shift"):
    print(
        "suite",
        split,
        {
            m: round(
                fmean(
                    fmean(method[m])
                    for (s, sp), method in scenarios.items()
                    if sp == split
                ),
                8,
            )
            for m in arms + baselines + heldonly
        },
    )
print(
    "verified",
    len(trees),
    "trajectories,",
    sum(tr["calls"] for tr in trees),
    "evaluations,",
    pairs,
    "paired 15-node seeds,",
    len(scores),
    "score rows,",
    len(gaps),
    "paired analyses",
)

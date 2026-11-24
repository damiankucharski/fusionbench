"""Reconstruct the saved pruning paths and audit their selection, costs and test scores."""

import csv
import gzip
import hashlib
import json
from collections import defaultdict
from math import sqrt
from pathlib import Path
from statistics import fmean, stdev

import numpy as np
from scipy.stats import t

from fusionbench.eval.full_suite_point import SHIFTS
from fusionbench.scenarios.registry import STANDARD_V1_SCENARIOS as SCENARIOS

FOLDER = Path(__file__).resolve().parent
SOURCE = FOLDER.parent / "mixed-deep-init"
FOCAL = "compositional_switch"
ARMS = ("stock", "mixed_chain", "mixed_branch")
BASELINES = ("mean", "top_n_640", "stack_640", "top_n_896", "stack_896")


def json_lines(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def csv_rows(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def key(row, field):
    return row["scenario"], int(row["data_seed"]), int(row["gp_seed"]), row[field]


def walk(node, path=()):
    yield path, node
    for index, child in enumerate(node["children"]):
        yield from walk(child, path + (index,))


def size(tree):
    nodes = [node for _, node in walk(tree)]
    return len({node["id"] for node in nodes if "id" in node}), len(nodes)


def bypass(node, indices):
    index, *rest = indices
    assert 0 <= index < len(node["children"])
    children = node["children"].copy()
    if rest:
        children[index] = bypass(children[index], rest)
    else:
        assert node["type"] == "ValueNode" and children[index]["type"] != "ValueNode"
        del children[index]
    return {**node, "children": children}


def draw(scenario, seed, stream, count):
    model = SCENARIOS[scenario](**(SHIFTS[scenario] if stream == 2303 else {}))
    predictions, y, _ = model._generate(count, np.random.default_rng([seed, stream]))
    x = np.asarray(predictions, dtype=np.float64)
    return x, np.eye(model.n_classes)[np.asarray(y).ravel()]


def probabilities(tree, x):
    """Interpret an ordered native value/update program directly using NumPy."""
    classes = x.shape[2]

    def evaluate(node):
        assert node["type"] == "ValueNode"
        col = int(node["id"].removeprefix("m").removesuffix(".npy"))
        current = x[:, col, 1:] if classes == 2 else x[:, col]
        for op in node["children"]:
            others = [evaluate(c) for c in op["children"]]
            kind = op["type"]
            if kind in ("FarThresholdNode", "CloseThresholdNode"):
                values = np.stack([current, *others])
                distance = np.square(values - op["threshold"])
                ix = (np.argmin if op["close"] else np.argmax)(distance, axis=0)
                current = np.take_along_axis(values, ix[None], axis=0)[0]
            elif kind == "WeightedMeanNode":
                w, *weights = op["weights"]
                current = current * w
                for child, weight in zip(others, weights, strict=True):
                    current += child * weight
            elif kind == "MeanNode":
                current = current.copy()
                for child in others:
                    current += child
                current /= 1 + len(others)
            else:
                assert kind in ("MinNode", "MaxNode")
                combine = np.minimum if kind == "MinNode" else np.maximum
                for child in others:
                    current = combine(current, child)
            if classes > 2:
                current = current + (current.sum(axis=1, keepdims=True) == 0)
                current = current / current.sum(axis=1, keepdims=True)
        return current

    values = evaluate(tree)
    if classes == 2:
        pos = np.clip(values[:, 0], 0, 1)
        return np.column_stack([1 - pos, pos])
    clipped = np.clip(values, 1e-12, None)
    return clipped / clipped.sum(axis=1, keepdims=True)


def brier(tree, data):
    x, targets = data
    return float(np.square(probabilities(tree, x) - targets).sum(axis=1).mean())


def interval(bank_values):
    avg = fmean(bank_values)
    half = (
        stdev(bank_values) * t.ppf(0.975, len(bank_values) - 1) / sqrt(len(bank_values))
    )
    return avg, avg - half, avg + half


def near(left, right):
    assert abs(float(left) - float(right)) < 1e-12, (left, right)


def main():
    original = {key(r, "method"): r for r in json_lines(SOURCE / "trees.jsonl.gz")}
    validation = {
        (
            r["scenario"],
            int(r["data_seed"]),
            int(float(r["gp_seed"])),
            r["method"],
        ): float(r["brier"])
        for r in csv_rows(SOURCE / "scores.csv.gz")
        if r["split"] == "validation" and r["gp_seed"]
    }
    paths = {key(r, "arm"): r for r in json_lines(FOLDER / "paths.jsonl.gz")}
    raw = csv_rows(FOLDER / "scores.csv.gz")
    scores = {
        (
            r["scenario"],
            int(r["data_seed"]),
            int(float(r["gp_seed"])) if r["gp_seed"] else None,
            r["arm"],
            r["method"],
            r["split"],
        ): r
        for r in raw
    }
    assert len(scores) == len(raw)
    prefixes = json_lines(FOLDER / "topn-prefixes.jsonl.gz")
    ranked = {(r["scenario"], r["data_seed"], r["labels"]): r for r in prefixes}
    assert len(ranked) == len(prefixes) == 160
    scenarios = defaultdict(list)
    gaps = defaultdict(list)
    tested, covered, path_steps, fit_calls = set(), set(), 0, 0
    for name, factory in SCENARIOS.items():
        width = len(factory().model_ids())
        for bank in range(92300, 92308 if name == FOCAL else 92304):
            splits = ("iid", "shift") if name in SHIFTS else ("iid",)
            fit = draw(name, bank, 0, 384)
            val = draw(name, bank, 1, 256)
            prune = draw(name, bank, 2301, 256)
            test = {
                s: draw(name, bank, 2302 if s == "iid" else 2303, 4096) for s in splits
            }
            for n in (640, 896):
                r = ranked[name, bank, n]
                assert (
                    sorted(r["ranking"]) == list(range(width))
                    and len(r["prefix_brier"]) == width
                )
                assert r["selected"] == np.argmin(r["prefix_brier"]) + 1
            for split in splits:
                for method in BASELINES:
                    tag = (name, bank, None, "baseline", method, split)
                    kept = scores[tag]
                    covered.add(tag)
                    count = (
                        ranked[name, bank, int(method.rsplit("_", 1)[1])]["selected"]
                        if method.startswith("top_n")
                        else width
                    )
                    assert int(kept["providers"]) == count and not kept["nodes"]
            for arm in ARMS:
                for seed in (2 * bank + 17, 2 * bank + 18):
                    tag = (name, bank, seed, arm)
                    tested.add(tag)
                    data, tree = paths[tag], original[tag]["program"]
                    stages = data["steps"]
                    evolved = [tree]
                    assert stages[0]["cut"] is None and abs(data["replay_gap"]) <= 1e-9
                    near(abs(brier(tree, val) - validation[tag]), data["replay_gap"])
                    evaluations = 1
                    for idx, stage in enumerate(stages):
                        if idx:
                            moves = [
                                p
                                for p, node in walk(tree)
                                if node["type"] != "ValueNode"
                            ]
                            chosen = tuple(stage["cut"])
                            assert chosen in moves
                            evaluations += len(moves)
                            # Rank every fit-only alternative using preorder as the final tie-breaker.
                            options = []
                            for move in moves:
                                alternative = bypass(tree, move)
                                options.append(
                                    (brier(alternative, fit), *size(alternative), move)
                                )
                            assert min(options)[-1] == chosen, (
                                tag,
                                idx,
                                min(options),
                                chosen,
                            )
                            tree = bypass(tree, chosen)
                            evolved.append(tree)
                        assert idx == stage["step"] and size(tree) == (
                            stage["providers"],
                            stage["nodes"],
                        )
                        near(brier(tree, fit), stage["fit_brier"])
                        near(brier(tree, prune), stage["prune_brier"])
                    assert data["fit_evaluations"] == evaluations
                    path_steps += len(stages)
                    fit_calls += evaluations
                    selected = min(
                        (
                            s
                            for s in stages
                            if s["prune_brier"] <= stages[0]["prune_brier"] + 0.005
                        ),
                        key=lambda s: (s["providers"], s["nodes"], s["prune_brier"]),
                    )["step"]
                    assert (
                        data["selected"] == selected
                        and data["endpoint"] == len(evolved) - 1
                    )
                    assert size(evolved[-1]) == (1, 1)
                    for method, idx in [
                        ("intact", 0),
                        ("pruned", selected),
                        ("endpoint", len(evolved) - 1),
                    ]:
                        for split in splits:
                            k = tag + (method, split)
                            covered.add(k)
                            row = scores[k]
                            x, labels = test[split]
                            p = probabilities(evolved[idx], x)
                            assert (
                                row["prediction_sha256"]
                                == hashlib.sha256(
                                    p.astype("<f8", copy=False).tobytes()
                                ).hexdigest()
                            ), k
                            near(np.square(p - labels).sum(1).mean(), row["brier"])
                            assert size(evolved[idx]) == (
                                int(row["providers"]),
                                int(float(row["nodes"])),
                            )
                for split in splits:
                    for metric in ("brier", "providers", "nodes"):
                        diffs = []
                        for seed in (2 * bank + 17, 2 * bank + 18):
                            idx = (name, bank, seed, arm)
                            x = float(scores[idx + ("pruned", split)][metric]) - float(
                                scores[idx + ("intact", split)][metric]
                            )
                            diffs.append(x if metric == "brier" else -x)
                        gaps[name, arm, split, metric].append(fmean(diffs))
    assert (
        len(tested) == 480
        and tested == set(original) == set(paths) == set(validation)
        and covered == set(scores)
    )
    # Equal starts per bank and equal bank weights within each family.
    bankmeans = defaultdict(list)
    for r in raw:
        for metric in ("brier", "providers", "nodes"):
            if r[metric]:
                bankmeans[
                    r["scenario"],
                    r["arm"],
                    r["method"],
                    r["split"],
                    metric,
                    r["data_seed"],
                ].append(float(r[metric]))
    for k, values in bankmeans.items():
        scenarios[k[:-1]].append(fmean(values))
    checked = set()
    for r in csv_rows(FOLDER / "means.csv"):
        for metric in ("brier", "providers", "nodes"):
            if r[metric]:
                k = (r["scenario"], r["arm"], r["method"], r["split"], metric)
                assert k not in checked
                checked.add(k)
                near(fmean(scenarios[k]), r[metric])
    assert checked == set(scenarios)
    paired = csv_rows(FOLDER / "paired.csv")
    assert len(paired) == len(gaps)
    checked = set()
    for r in paired:
        k = r["scenario"], r["arm"], r["split"], r["metric"]
        assert k not in checked
        checked.add(k)
        values = gaps[k]
        assert len(values) == int(r["banks"])
        for val, field in zip(interval(values), ("mean", "low", "high"), strict=True):
            near(val, r[field])
    assert checked == set(gaps)
    primary = FOCAL, "mixed_branch", "iid"
    print("Primary IID delta:", interval(gaps[primary + ("brier",)]))
    print("Primary providers saved:", interval(gaps[primary + ("providers",)]))
    print(
        f"{len(tested)} trees, {path_steps} stages, {fit_calls} path construction evaluations, "
        f"{len(scores)} complete score rows, {len(gaps)} verified paired contrasts"
    )


if __name__ == "__main__":
    main()

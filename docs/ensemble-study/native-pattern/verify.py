"""Independently replay saved native trees, reference predictors and fixed-suite contrasts."""

import csv
import gzip
import hashlib
import itertools
import json
import runpy
from math import sqrt
from pathlib import Path
from statistics import fmean, stdev

import numpy as np
from scipy.stats import t
from threadpoolctl import threadpool_limits

from fusionbench.eval.full_suite_point import SHIFTS
from fusionbench.scenarios.registry import STANDARD_V1_SCENARIOS as SCENARIOS

FOLDER = Path(__file__).resolve().parent
# These predictors were written independently of the fitting and benchmarking code.
previous = runpy.run_path(str(FOLDER.parent / "oof-gp-stack" / "verify.py"))
logistic = previous["logistic"]
rank_providers = previous["top_prefix"]
predict_prefix = previous["average_prefix"]


def rows(name):
    opener = gzip.open if name.endswith(".gz") else open
    with opener(FOLDER / name, "rt", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def records(name):
    with gzip.open(FOLDER / name, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def fingerprint(p):
    return hashlib.sha256(np.asarray(p, dtype="<f8").tobytes()).hexdigest()


def near(a, b, tolerance=1e-12):
    np.testing.assert_allclose(a, b, rtol=0, atol=tolerance)


def ci(sample):
    mean = fmean(sample)
    half = stdev(sample) * float(t.ppf(0.975, len(sample) - 1)) / sqrt(len(sample))
    return mean, mean - half, mean + half


METHODS = ("native", "gp", "flat", "augmented", "mean", "top_n", "logit", "prob_lr")
ORDINARY = {
    "MinNode",
    "MaxNode",
    "MeanNode",
    "WeightedMeanNode",
    "CloseThresholdNode",
    "FarThresholdNode",
}
HEADS = {"BinaryAffineNode", "WeightedLogitMeanNode"}


def draw(name, seed, stream):
    generator = SCENARIOS[name](**(SHIFTS[name] if stream == 2 else {}))
    x, y, _ = generator._generate(
        640 if stream == 0 else 4096, np.random.default_rng([seed, stream])
    )
    return x, np.asarray(y).ravel()


def walk(node):
    yield node
    for child in node["children"]:
        yield from walk(child)


def validate_program(tree, kind, width, classes):
    allowed = {f"m{i}.npy" for i in range(width)}
    nodes = list(walk(tree))
    assert tree["type"] == "ValueNode"
    assert all(n["id"] in allowed for n in nodes if n["type"] == "ValueNode")
    operators = [n for n in nodes if n["type"] != "ValueNode"]
    assert all(op["type"] in ORDINARY | HEADS for op in operators)
    heads = [op for op in operators if op["type"] in HEADS]
    if kind == "gp":
        assert not heads
        return
    assert len(heads) == 1 and heads[0] is tree["children"][-1]
    head = heads[0]
    params = head["coefficients"] if classes == 2 else head["weights"]
    assert np.isfinite(params).all() and len(head["children"]) + 1 == len(params)
    if classes == 2:
        assert (
            head["type"] == "BinaryAffineNode"
            and head["transform"] == "logit"
            and np.isfinite(head["intercept"])
        )
    else:
        assert head["type"] == "WeightedLogitMeanNode"
    terminals = [tree["id"]] + [n["id"] for n in head["children"][: width - 1]]
    assert set(terminals) == allowed
    assert len(params) == width if kind == "flat" else width <= len(params) <= width + 3
    if kind == "flat":
        assert len(operators) == 1


def native_prediction(tree, x):
    """Execute the archived program directly on raw probabilities, in child order."""
    x = np.asarray(x, dtype=np.float64)
    classes = x.shape[-1]

    def evaluate(node):
        assert node["type"] == "ValueNode"
        index = int(node["id"].removeprefix("m").removesuffix(".npy"))
        output = x[:, index, 1:] if classes == 2 else x[:, index]
        for operator in node["children"]:
            inputs = [output, *(evaluate(c) for c in operator["children"])]
            kind = operator["type"]
            if kind == "BinaryAffineNode":
                linear = 0.0
                for inp, w in zip(inputs, operator["coefficients"], strict=True):
                    bounded = np.clip(inp, 1e-6, 1.0 - 1e-6)
                    linear = linear + w * np.log(bounded / (1.0 - bounded))
                linear = linear + operator["intercept"]
                tail = np.exp(-np.abs(linear))
                pos = ((linear >= 0) + (linear < 0) * tail) / (1.0 + tail)
                yes = np.clip(pos, 1e-12, 1.0)
                no = np.clip(1.0 - pos, 1e-12, 1.0)
                output = yes / (yes + no)
            elif kind == "WeightedLogitMeanNode":
                terms = [
                    w * np.log(np.clip(inp, 1e-6, 1 - 1e-6))
                    for w, inp in zip(operator["weights"], inputs, strict=True)
                ]
                output = np.exp(np.clip(sum(terms), -30.0, 30.0))
            elif kind in ("MinNode", "MaxNode"):
                fn = np.minimum if kind == "MinNode" else np.maximum
                output = fn.reduce(inputs)
            elif kind == "MeanNode":
                output = sum(inputs) / len(inputs)
            elif kind == "WeightedMeanNode":
                output = sum(
                    w * inp for w, inp in zip(operator["weights"], inputs, strict=True)
                )
            else:
                assert kind in ("FarThresholdNode", "CloseThresholdNode")
                arr = np.stack(inputs)
                dis = (arr - operator["threshold"]) ** 2
                index = (np.argmin if operator["close"] else np.argmax)(dis, axis=0)
                output = np.take_along_axis(arr, index[None], axis=0)[0]
            if classes > 2:
                output = output + (output.sum(1, keepdims=True) == 0)
                output = output / output.sum(1, keepdims=True)
        return output

    p = evaluate(tree)
    if classes == 2:
        pos = p[:, 0].clip(0, 1)
        return np.column_stack((1 - pos, pos))
    p = np.maximum(p, 1e-12)
    return p / p.sum(1, keepdims=True)


def audit_aggregates(scores):
    comparisons = rows("contrasts.csv")
    expected = {
        (split, method)
        for split in ("iid", "shift")
        for method in METHODS
        if method != "native"
    }
    assert (
        len(comparisons) == len(expected)
        and {(r["split"], r["reference"]) for r in comparisons} == expected
    )
    for row in comparisons:
        split, ref = row["split"], row["reference"]
        names = [name for name in SCENARIOS if split == "iid" or name in SHIFTS]
        by_family = {
            name: [
                scores[name, i, split, "native"] - scores[name, i, split, ref]
                for i in range(12)
            ]
            for name in names
        }
        macro = [fmean(by_family[name][i] for name in names) for i in range(12)]
        near(ci(macro), [float(row[k]) for k in ("delta", "low", "high")])
        means = {name: fmean(by_family[name]) for name in names}
        leave_out = [fmean(v for s, v in means.items() if s != name) for name in names]
        near(
            [min(leave_out), max(leave_out), max(means.values())],
            [float(row[k]) for k in ("loo_min", "loo_max", "worst_delta")],
        )
        assert int(row["families"]) == len(names) and int(row["wins"]) == sum(
            v < 0 for v in means.values()
        )
        assert means[row["worst"]] == max(means.values())
        if split == "iid" and ref in ("logit", "prob_lr"):
            print(f"native minus {ref}:", ci(macro), "wins", row["wins"])
    family = rows("families.csv")
    cells = {
        (name, sp)
        for name in SCENARIOS
        for sp in (("iid", "shift") if name in SHIFTS else ("iid",))
    }
    assert (
        len(family) == len(cells)
        and {(r["scenario"], r["split"]) for r in family} == cells
    )
    for row in family:
        for method in METHODS:
            near(
                fmean(
                    scores[row["scenario"], i, row["split"], method] for i in range(12)
                ),
                float(row[method]),
            )
    print(
        "IID Brier:",
        sorted(
            (
                (
                    m,
                    fmean(
                        scores[name, i, "iid", m]
                        for name in SCENARIOS
                        for i in range(12)
                    ),
                )
                for m in METHODS
            ),
            key=lambda v: v[1],
        ),
    )


def audit_bank(name, rep, seed, bank, scores):
    x, y = draw(name, seed, 0)
    width, classes = x.shape[1:]
    assert (
        bank["data_seed"] == seed
        and bank["scenario"] == name
        and bank["replicate"] == rep
    )
    ranking, prefix_losses = rank_providers(x, y)
    assert (
        bank["ranking"] == list(ranking)
        and bank["top_n"] == int(np.argmin(prefix_losses)) + 1
    )
    near(prefix_losses, bank["prefix_brier"])
    chosen = bank["chosen"]
    trees = bank["trees"]
    assert set(trees) == {"flat", "augmented", "gp"} and chosen in trees
    for kind, program in trees.items():
        validate_program(program, kind, width, classes)
    order = np.random.default_rng(np.random.SeedSequence([seed, 51])).permutation(640)
    validation = order[448:]
    target = np.eye(classes)[y[validation]]
    counts = bank["counts"]
    assert counts["discovery_sha256"] == fingerprint(order[:448])
    assert counts["validation_sha256"] == fingerprint(validation)
    assert (
        counts["discovery_rows"],
        counts["validation_rows"],
        counts["refit_rows"],
    ) == (448, 192, 640)
    assert (
        counts["fitness"] == "one_minus_half_full_brier"
        and counts["gp_selection_scale"] == "unit_brier_3_decimals"
    )
    calls = counts["score_calls"]
    assert calls["gp_discovery"] == 600 and calls["family_validation"] == 3
    assert 0 < counts["gp_unique_programs"] == calls["gp_validation"] <= 600
    assert (
        all(v >= 1 for v in calls.values())
        and sum(calls.values()) == counts["total_score_calls"]
    )
    screen = counts["screen"]
    assert screen["archive_programs"] == counts["gp_unique_programs"]
    selected = screen["selected"]
    eligible = screen["unique_novel_features"]
    assert (
        len(selected) <= min(3, eligible)
        and eligible <= screen["non_singletons"] <= screen["archive_programs"]
    )
    rounds = min(len(selected) + 1, 3, eligible)
    assert (
        screen["evaluations"]
        == calls["feature_screen"]
        == 1 + 4 * sum(eligible - j for j in range(rounds))
    )
    history = [screen["initial_score"], *(entry["score"] for entry in selected)]
    assert all(later > earlier for earlier, later in itertools.pairwise(history))
    assert all(item["weight"] in (-1, -0.25, 0.25, 1) for item in selected)
    assert len({item["archive_index"] for item in selected}) == len(selected)
    eligible_head = trees["augmented"]["children"][-1]
    assert len(eligible_head["children"]) + 1 == width + len(selected)
    scores_v = counts["validation_scores"]
    discoveries = bank["discovery_trees"]
    assert discoveries["gp"] == trees["gp"]
    assert set(discoveries) == set(scores_v) == set(trees)
    for kind, program in discoveries.items():
        validate_program(program, kind, width, classes)
        p = native_prediction(program, x[validation])
        value = 1.0 - 0.5 * ((p - target) ** 2).sum(1).mean()
        near(value, scores_v[kind])
    assert max(("flat", "augmented", "gp"), key=scores_v.get) == chosen
    for split, stream in [("iid", 1)] + ([("shift", 2)] if name in SHIFTS else []):
        z, labels = draw(name, seed, stream)
        raw = {m: native_prediction(tree, z) for m, tree in trees.items()}
        raw["native"] = raw[chosen]
        raw["mean"] = z.mean(axis=1)
        raw["top_n"] = predict_prefix(z, ranking[: bank["top_n"]])
        for m, logit in [("logit", True), ("prob_lr", False)]:
            raw[m] = logistic(x, y, z, logit)
        target = np.eye(classes)[labels]
        for m, p in raw.items():
            row = scores[name, rep, split, m]
            assert (
                int(row["data_seed"]) == seed
                and np.isfinite(p).all()
                and p.shape == target.shape
            )
            assert p.min() >= 0 and p.max() <= 1.0 + 1e-12
            near(p.sum(1), np.ones(len(p)))
            near(float(row["brier"]), ((p - target) ** 2).sum(1).mean())
            assert fingerprint(p) == row["prediction_sha256"], (name, seed, split, m)
    return chosen


def main():
    banks = records("banks.jsonl.gz")
    raw = rows("scores.csv.gz")
    grouped = {(r["scenario"], r["replicate"]): r for r in banks}
    scores = {
        (r["scenario"], int(r["replicate"]), r["split"], r["method"]): r for r in raw
    }
    assert len(raw) == len(scores) == 2688 and len(banks) == len(grouped) == 228
    choices = {m: 0 for m in ("flat", "augmented", "gp")}
    with threadpool_limits(limits=1):
        for index, name in enumerate(SCENARIOS):
            for rep in range(12):
                seed = 1460000 + 1000 * index + rep
                chosen = audit_bank(name, rep, seed, grouped[name, rep], scores)
                choices[chosen] += 1
    audit_aggregates({tag: float(row["brier"]) for tag, row in scores.items()})
    print("Native tree selections:", choices)
    print(
        f"{len(banks)} banks, {len(banks) * 3} pre-refit validation scores, {len(raw)} independent test Brier scores and SHA-256 fingerprints."
    )


if __name__ == "__main__":
    main()

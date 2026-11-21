"""Audit stored champion expressions independently of signatures and OKAPI nodes."""

import argparse
import ast
import csv
from fractions import Fraction
from hashlib import sha256
from pathlib import Path

import numpy as np

from fusionbench.scenarios.gates import TwoPathways

DOMAINS = ("fit", "validation", "iid", "shift")


def read(file):
    with file.open() as stream:
        return list(csv.DictReader(stream))


def evaluate(tree, columns):
    name, operators = tree
    value = columns[name]
    for op, arguments in operators:
        combine = {"MIN": np.minimum, "MAX": np.maximum}[op]
        for arg in arguments:
            value = combine(value, evaluate(arg, columns))
    return value


def number_nodes(tree):
    return 1 + sum(number_nodes(child) for child in tree[1])


def data(seed):
    columns, labels = [], []
    for index, (n, accuracy) in enumerate(
        ((128, 0.9), (128, 0.9), (8192, 0.9), (8192, 0.78))
    ):
        p, y, _ = TwoPathways(rep=2, accuracy=accuracy)._generate(
            n, np.random.default_rng([seed, index])
        )
        columns.append({f"VN[p{i}.npy]": p[:, i, 1] for i in range(p.shape[1])})
        labels.append(y)
    return columns, labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("replay", type=Path)
    args = parser.parse_args()
    reference = {
        int(r["data_seed"]): Fraction(
            int(r["optimum_numerator"]), int(r["optimum_denominator"])
        )
        for r in read(args.results / "baselines.csv")
        if r["method"] == "exact_lattice"
    }
    rows = read(args.results / "gp.csv")
    assert len(rows) == 12 * 4 * 3 * 2 * 2
    replay = {
        (int(r["data_seed"]), int(r["gp_seed"]), int(r["calls"])): r
        for r in read(args.replay)
    }
    assert len(replay) == 96
    panels = {seed: data(seed) for seed in reference}
    split_comparisons = 0
    for row in rows:
        seed = int(row["data_seed"])
        tree = ast.literal_eval(row["tree"])
        assert number_nodes(tree) == int(row["nodes"])
        columns, labels = panels[seed]
        digest = sha256()
        for name, column, y in zip(DOMAINS, columns, labels, strict=True):
            predictions = evaluate(tree, column)
            digest.update(np.asarray(predictions, dtype="<f8").tobytes(order="C"))
            assert float(np.mean((predictions - y) ** 2)) == float(row[name])
            split_comparisons += 1
            if name == "fit":
                exact_loss = sum(
                    (Fraction(float(v)) - int(target)) ** 2
                    for v, target in zip(predictions, y, strict=True)
                ) / len(y)
                assert exact_loss >= reference[seed]
                assert float(exact_loss - reference[seed]) == float(row["fit_regret"])
                assert (exact_loss == reference[seed]) == (row["exact_hit"] == "True")
        assert digest.hexdigest() == row["prediction_sha256"]
        if (
            row["policy"] == "structural"
            and row["champion"] == "fit"
            and row["checkpoint"].startswith("calls_")
        ):
            key = (seed, int(row["gp_seed"]), int(row["calls"]))
            original = replay.pop(key)
            assert (
                sha256((row["tree"] + "\n").encode()).hexdigest()
                == original["fit_champion_sha256"]
            )
            assert float(original["fit"]) == float(row["fit"])
    assert not replay
    print(
        f"96/96 native structural-arm fits and winning trees match. {len(rows)} champion digests, exact fit regrets, exact hit flags and node counts match. {split_comparisons} independently evaluated Brier values match exactly."
    )


if __name__ == "__main__":
    main()

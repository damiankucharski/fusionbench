"""Reconstruct default-suite data, baseline fits and ordered native-tree predictions."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fusionbench.scenarios.registry import SCENARIOS

SHIFTS = {
    "correlated": {"corr": 0.1},
    "calibrated_suppressor": {"corr": 0},
    "polarity_fault": {"correlation": 0},
    "weakest_link": {"accuracy": 0.76},
    "two_pathways": {"accuracy": 0.78},
    "parity": {"accuracy": 0.78},
    "native_switch": {"mixing": 1},
    "native_close_switch": {"mixing": 1},
    "compositional_switch": {"corner_probability": 0},
}


def normalized(p):
    x = np.maximum(p, 1e-12)
    return x / x.sum(1, keepdims=True)


def digest(p):
    return hashlib.sha256(np.asarray(p, dtype="<f8").tobytes()).hexdigest()


def features(p):
    x = p.reshape(len(p), -1)
    x = np.clip(x, 1e-6, 1 - 1e-6)
    return np.log(x / (1 - x))


def read_program(program, x):
    binary = x.shape[2] == 2

    def value(record):
        i = int(record["id"].removeprefix("m").removesuffix(".npy"))
        result = x[:, i, 1:2] if binary else x[:, i, :]
        for op in record["children"]:
            child = [result, *(value(v) for v in op["children"])]
            kind = op["type"]
            if kind == "WeightedMeanNode":
                weights = op["weights"]
                result = child[0] * weights[0]
                for weight, v in zip(weights[1:], child[1:], strict=True):
                    result += weight * v
            elif kind == "MeanNode":
                result = child[0].copy()
                for v in child[1:]:
                    result += v
                result = result / len(child)
            elif kind == "MinNode":
                result = np.minimum.reduce(child)
            elif kind == "MaxNode":
                result = np.maximum.reduce(child)
            elif kind in ("CloseThresholdNode", "FarThresholdNode"):
                choices = np.stack(child)
                dist = abs(choices - op["threshold"])
                rank = dist.argmin(0) if op["close"] else dist.argmax(0)
                result = np.take_along_axis(choices, rank[None], axis=0)[0]
            else:
                raise ValueError(kind)
            if not binary:
                total = result.sum(1, keepdims=True)
                raw = result + (total == 0)
                result = raw / raw.sum(1, keepdims=True)
        return result

    arr = value(program)
    return np.column_stack((1 - arr[:, 0], arr[:, 0])) if binary else normalized(arr)


def baseline_fit(x, y):
    c = x.shape[-1]
    one = np.eye(c)[y]

    def loss(p):
        return np.mean(np.sum((normalized(p) - one) ** 2, axis=1))

    order = np.argsort([loss(x[:, i]) for i in range(x.shape[1])], kind="stable")
    cum = np.zeros_like(x[:, 0])
    prefix = []
    for i, idx in enumerate(order, 1):
        cum += x[:, idx]
        prefix.append(loss(cum / i))
    n = int(np.argmin(prefix)) + 1
    classifier = make_pipeline(
        StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
    ).fit(features(x), y)

    def pred(p):
        prefix_p = np.zeros_like(p[:, 0])
        for i in order[:n]:
            prefix_p += p[:, i]
        stack = np.zeros((len(p), c))
        stack[:, classifier.classes_.astype(int)] = classifier.predict_proba(
            features(p)
        )
        return {
            "mean": p.mean(1),
            "top_n": normalized(prefix_p / n),
            "stack_logit": normalized(stack),
        }

    return pred, order, n, prefix


def verify(path):
    frame = pd.read_csv(path / "runs.csv.gz", float_precision="round_trip")
    assert set(frame.scenario) == set(SCENARIOS) and len(SCENARIOS) == 19
    assert set(frame.data_seed) == set(range(4600, 4608))
    assert set(frame[frame.split == "shift"].scenario) == set(SHIFTS)
    cnt = 0
    bitwise = 0
    unique_replays = 0
    max_error = 0.0
    for (name, seed), g in frame.groupby(["scenario", "data_seed"]):
        scenario = SCENARIOS[name]()
        splits = {}
        for idx, (split, n) in enumerate(
            (("fit", 384), ("validation", 256), ("iid", 2048), ("shift", 2048))
        ):
            if split == "shift" and name not in SHIFTS:
                continue
            gen = scenario if split != "shift" else SCENARIOS[name](**SHIFTS[name])
            x, y, extra = gen._generate(n, np.random.default_rng([seed, idx]))
            splits[split] = (x, y, gen.bayes_oracle(x, extra))
        fit, fy, _ = splits["fit"]
        predict, rank, nsel, prefix = baseline_fit(fit, fy)
        for split, (x, y, oracle) in splits.items():
            records = g[g.split == split]
            assert len(records) == 19
            assert (records.classes == x.shape[2]).all() and (
                records.providers == x.shape[1]
            ).all()
            all_models = predict(x)
            for row in records.itertuples():
                if row.method == "top_n":
                    assert (
                        row.n_selected == nsel
                        and json.loads(row.ranking) == rank.tolist()
                    )
                    np.testing.assert_allclose(
                        json.loads(row.prefix_scores), prefix, atol=2e-15, rtol=0
                    )
                if row.method.startswith("point_") and row.program not in all_models:
                    all_models[row.program] = read_program(json.loads(row.program), x)
                    unique_replays += 1
                p = all_models[
                    row.program if row.method.startswith("point_") else row.method
                ]
                evaluations = normalized(p)
                c = x.shape[2]
                expected = {
                    "brier": np.mean(np.sum((p - np.eye(c)[y]) ** 2, 1)),
                    "conditional_excess": np.mean(np.sum((p - oracle) ** 2, 1)),
                    "macro_f1": f1_score(
                        y, evaluations.argmax(1), average="macro", zero_division=0
                    ),
                    "accuracy": accuracy_score(y, evaluations.argmax(1)),
                    "log_loss": log_loss(y, evaluations, labels=np.arange(c)),
                }
                assert digest(oracle) == row.oracle_sha256
                if digest(p) == row.prediction_sha256:
                    bitwise += 1
                error = max(abs(v - getattr(row, k)) for k, v in expected.items())
                if error > 5e-12:
                    raise AssertionError((name, seed, split, row.method, error))
                max_error = max(max_error, error)
                cnt += 1
    pairs = pd.read_csv(path / "paired.csv", dtype={"checkpoint": str})
    for entry in pairs.itertuples():
        subset = frame[
            (frame.scenario == entry.scenario)
            & (frame.split == entry.split)
            & (frame.checkpoint.astype(str) == entry.checkpoint)
            & (frame.selection == entry.selection)
        ]
        scores = subset.groupby(["data_seed", "method"])[entry.metric].mean().unstack()
        diff = scores.point_on - scores.point_off
        margin = t.ppf(0.975, 7) * diff.std(ddof=1) / np.sqrt(8)
        np.testing.assert_allclose(
            [entry.difference, entry.low_95, entry.high_95],
            [diff.mean(), diff.mean() - margin, diff.mean() + margin],
            atol=3e-14,
            rtol=1e-13,
        )
    outcome = {
        "scenario_data_cells": 19 * 8,
        "vectors": cnt,
        "bitwise_vectors": bitwise,
        "native_decodings": unique_replays,
        "checked_metric_values": 5 * cnt,
        "paired_contrasts": len(pairs),
        "max_absolute_metric_error": max_error,
        "input_sha256": hashlib.sha256((path / "runs.csv.gz").read_bytes()).hexdigest(),
    }
    print(json.dumps(outcome, indent=2))
    (path / "predictions-verified.json").write_text(
        json.dumps(outcome, indent=2) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    verify(parser.parse_args().directory)

"""Independent check of stored programs, exact gate minima and cross-domain losses."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fusionbench.scenarios.native_switch import draw_native_switch


def evaluate(program, x):
    result = x[:, int(program["id"][1:-4])]
    for op in program["children"]:
        operands = [result] + [evaluate(child, x) for child in op["children"]]
        kind = op["type"]
        if kind == "MinNode":
            result = np.minimum.reduce(operands)
        elif kind == "MaxNode":
            result = np.maximum.reduce(operands)
        elif kind == "MeanNode":
            result = np.mean(operands, axis=0)
        elif kind == "WeightedMeanNode":
            result = sum(w * p for w, p in zip(op["weights"], operands, strict=True))
        elif kind in ("FarThresholdNode", "CloseThresholdNode"):
            options = np.asarray(operands)
            distance = np.square(options - op["threshold"])
            index = distance.argmin(0) if op["close"] else distance.argmax(0)
            result = options[index, np.arange(len(x))]
        else:
            raise TypeError(kind)
    return result


def literal_optimum(x, y, fixed):
    idx = np.arange(257) if not fixed else np.array([128])
    t = idx[:, None] / 256
    best = None
    ties = 0
    winner = None
    for i in range(x.shape[1]):
        for j in range(x.shape[1]):
            if i == j:
                continue
            a, b = x[:, i], x[:, j]
            da, db = (a - t) ** 2, (b - t) ** 2
            for kind, use in (("close", da <= db), ("far", da >= db)):
                cost = np.sum(
                    np.square(64 * (np.where(use, a, b) - y)), axis=1, dtype=np.int64
                )
                local = int(cost.min())
                count = int((cost == local).sum())
                if best is None or local < best:
                    best, ties = local, count
                    winner = (i, j, kind, float(t[cost.argmin(), 0]))
                elif best == local:
                    ties += count
    return best, ties, winner


def verify(path):
    runs = pd.read_csv(path / "runs.csv.gz", float_precision="round_trip")
    largest = 0.0
    checked = 0
    exact = 0
    for (seed, regime), group in runs.groupby(["data_seed", "fit_regime"]):
        fit, val, test = [
            draw_native_switch(seed, i, n) for i, n in enumerate((512, 256, 8192))
        ]
        mixing = float(regime == "linear")
        splits = {
            "fit": (fit, mixing),
            "validation": (val, mixing),
            "switch_test": (test, 0),
            "weak_test": (test, 0.5),
            "linear_test": (test, 1),
        }
        programs = group[group.program.notna()]
        for method in ("gate_free_threshold", "gate_fixed_threshold"):
            saved = json.loads(programs[programs.method == method].iloc[0].program)
            best, ties, winner = literal_optimum(
                fit.preds[:, :, 1], fit.labels(mixing), method == "gate_fixed_threshold"
            )
            assert (
                saved["parent"],
                saved["child"],
                saved["direction"],
                saved["threshold"],
            ) == winner
            assert (
                saved["loss_numerator"] == best and saved["minimizers_on_grid"] == ties
            )
            exact += 1
        cache = {}
        for r in programs.itertuples():
            sample, mix = splits[r.split]
            key = (r.program, id(sample))
            if key not in cache:
                prog = json.loads(r.program)
                x = sample.preds[:, :, 1]
                if "direction" in prog:
                    a, b = x[:, prog["parent"]], x[:, prog["child"]]
                    da, db = (a - prog["threshold"]) ** 2, (b - prog["threshold"]) ** 2
                    pred = np.where(
                        da <= db if prog["direction"] == "close" else da >= db, a, b
                    )
                else:
                    pred = evaluate(prog, x)
                cache[key] = pred
            pred = cache[key]
            losses = [
                np.mean((pred - y) ** 2)
                for y in (sample.labels(mix), sample.probability(mix))
            ]
            err = np.abs(np.asarray(losses) - [r.observed_brier, r.conditional_excess])
            largest = max(largest, float(err.max()))
            assert err.max() < 1e-13, (r.method, seed, r.split, err)
            checked += 2
    print(
        f"Verified {exact} exact fit certificates with brute force, {checked} stored losses, max absolute error {largest:.3g}."
    )


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("path", type=Path)
    verify(cli.parse_args().path)

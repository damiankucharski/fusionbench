"""Restore and save the reported multiclass test vectors, checking each original hash."""

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fusionbench.eval.full_suite_point import SHIFTS, fingerprint, scenario_block
from fusionbench.eval.gate_necessity import native_tree
from fusionbench.eval.suite_point_gp import _bind, predict_tree
from fusionbench.methods.averaging import TopN
from fusionbench.methods.stacking import StackingLogit, StackingLR
from fusionbench.scenarios.registry import SCENARIOS

out = Path(__file__).resolve().parent


def main():
    scores = pd.read_csv(out / "scores.csv.gz").set_index(
        ["data_seed", "split", "method"]
    )
    data = {}
    with gzip.open(out / "banks.jsonl.gz", "rt") as handle:
        for row in map(json.loads, handle):
            name, seed = row["scenario"], row["data_seed"]
            source = SCENARIOS[name]()
            classes = source.n_classes
            if classes == 2:
                continue
            x, y, _ = scenario_block(source, seed, 0, 640)
            components = [
                m.fit(x, y, task="multiclass", n_classes=classes)
                for m in (StackingLogit(), StackingLR(), TopN("brier"))
            ]
            tree = native_tree(row["tree"], _bind(x))
            tests = [("iid", source, 1)]
            if name in SHIFTS:
                tests.append(("shift", SCENARIOS[name](**SHIFTS[name]), 2))
            for split, scenario, stream in tests:
                z, _, _ = scenario_block(scenario, seed, stream, 4096)
                a, b, top = (m.predict(z) for m in components)
                gp = predict_tree(tree, z)
                comp = np.stack([a, b, gp], axis=1)
                probs = {
                    "hybrid": np.einsum("nmc,m->nc", comp, row["weights"]),
                    "linear_blend": np.einsum(
                        "nmc,m->nc", comp[:, :2], row["linear_weights"]
                    ),
                    "logit": a,
                    "prob_lr": b,
                    "gp": gp,
                    "mean": z.mean(1),
                    "top_n": top,
                }
                for label, p in probs.items():
                    assert (
                        fingerprint(p)
                        == scores.loc[(seed, split, label), "prediction_sha256"]
                    )
                    data.setdefault(name, {})[
                        f"r{row['replicate']}_{split}_{label}"
                    ] = p
    for name, arrays in data.items():
        np.savez_compressed(out / f"{name}-probabilities.npz", **arrays)
        print(name, len(arrays), "verified full probability matrices")


if __name__ == "__main__":
    main()

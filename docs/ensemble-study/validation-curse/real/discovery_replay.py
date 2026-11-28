"""Replay frozen fits, measure relative validation optimism before postselection refits."""

import argparse
import gzip
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from run import STUDY, archive, digest, row_brier, source_ids
from summarize import INDEX, add_aggregates
from threadpoolctl import threadpool_limits

from fusionbench.eval.suite_point_gp import predict_tree
from fusionbench.methods.averaging import SimpleAverage
from fusionbench.methods.search_guard import SearchGuardedNativeOKAPI


def replay(job):
    meta, expected = job
    pool, rep = meta["pool"], meta["replicate"]
    b, s = archive()["load_bank"](pool, rep)
    all_ids, _ = source_ids(pool, rep)
    row_pos = {int(row): i for i, row in enumerate(all_ids)}
    pos = [row_pos[key] for key in meta["fit_rows"]]
    x, y = b.fit_preds[pos], b.fit_y[pos]
    logger.disable("okapi")
    rows = []
    with threadpool_limits(limits=1):
        for z in (0, 1):
            model = SearchGuardedNativeOKAPI(seed=s, split_seed=s, z=z).fit(
                x, y, task=b.task, n_classes=b.n_classes
            )
            if digest(model.predict(b.test_preds)) != expected[z]:
                raise AssertionError("Replayed deployment lost bitwise parity")
            v = model.validation_indices_
            if not np.array_equal(
                all_ids[pos][v], meta[f"search_z{z}"]["selection_row_ids"]
            ):
                raise AssertionError("Replayed validation records changed")
            tree = model.discovery_trees_[model.chosen_]
            risks = {}
            for split, xx, yy in [
                ("validation", x[v], y[v]),
                ("test", b.test_preds, b.test_y),
            ]:
                p = predict_tree(tree, model._query(xx))
                mean = SimpleAverage().predict(xx)
                risks[split + "_minus_mean"] = float(
                    np.mean(row_brier(p, yy) - row_brier(mean, yy))
                )
            rows.append(
                {
                    "pool": pool,
                    "dataset": meta["dataset"],
                    "replicate": rep,
                    "z": z,
                    "choice": model.chosen_,
                    "relative_optimism": risks["test_minus_mean"]
                    - risks["validation_minus_mean"],
                    "gp_unique_programs": model.counts_["gp_unique_programs"],
                    **risks,
                    **model.counts_["score_calls"],
                }
            )
    return rows


def run(out, workers):
    with gzip.open(out / "banks.jsonl.gz", "rt") as f:
        banks = [json.loads(line) for line in f]
    scores = pd.read_csv(out / "scores.csv.gz").set_index(
        ["pool", "replicate", "method"]
    )
    jobs = [
        (
            m,
            [
                scores.loc[
                    (m["pool"], m["replicate"], f"search_z{z}"), "prediction_sha256"
                ]
                for z in (0, 1)
            ],
        )
        for m in banks
    ]
    with ProcessPoolExecutor(max_workers=workers) as proc:
        data = pd.DataFrame(row for result in proc.map(replay, jobs) for row in result)
    data.to_csv(out / "discovery-replay.csv", index=False)
    pd.concat(
        {
            k: add_aggregates(data.pivot(index=INDEX, columns="z", values=k))
            for k in [
                "validation_minus_mean",
                "test_minus_mean",
                "relative_optimism",
                "gp_unique_programs",
                "gp_discovery",
            ]
        },
        names=["statistic", "pool"],
    ).to_csv(out / "pool-discovery.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=STUDY / "results")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    run(args.out, args.workers)

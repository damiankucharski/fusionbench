"""Run the paired 19-scenario experiment defined in protocol.md."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import t

from fusionbench.eval.full_suite_point import data_for, fingerprint
from fusionbench.eval.mutation_opportunity import evaluate_bank
from fusionbench.eval.native_switch import serialize
from fusionbench.eval.seed_geometry_standard import fit_seed_geometry
from fusionbench.eval.suite_point_gp import predict_tree
from fusionbench.scenarios.registry import STANDARD_V1_SCENARIOS as SCENARIOS

OUT = Path(__file__).resolve().parent
FOCAL = "compositional_switch"
ARMS = ("stock", "mixed_branch", "mixed_chain")


def referenced_ids(node):
    ids = {node.id} if hasattr(node, "id") else set()
    for child in node.children:
        ids.update(referenced_ids(child))
    return ids


def panel(job):
    logger.disable("okapi")
    name, bank = job
    scenario, splits = data_for(name, bank)
    x, y, _ = splits["fit"]
    vx, vy, _ = splits["validation"]
    baselines = evaluate_bank(name, bank, splits, arms=())
    pooled = (np.concatenate([x, vx]), np.concatenate([y, vy]), None)
    combined = evaluate_bank(name, bank, {**splits, "fit": pooled}, arms=())
    for r in combined:
        if r["method"] != "mean" and r["split"] in ("iid", "shift"):
            baselines.append({**r, "method": r["method"] + "_640"})
    fields = ("scenario", "data_seed", "method", "split", "brier", "prediction_sha256")
    rows = [{k: r[k] for k in fields} for r in baselines]
    trees = []
    for seed in (2 * bank + 17, 2 * bank + 18):
        stock = None
        for arm in ARMS:
            cp = fit_seed_geometry(x, y, vx, vy, seed=seed, shape=arm, budgets=(600,))[
                600
            ]
            tree = cp.val_tree
            initial = cp.initial_rows
            shared = [r["genotype_sha256"] for r in initial[:-6]]
            if arm == "stock":
                stock = shared
            elif shared != stock or any(r["nodes"] != 15 for r in initial[-6:]):
                raise AssertionError("Mismatched shared starters or deep program size")
            if cp.calls != 600:
                raise AssertionError("Mismatch in training evaluation budget")
            key = {"scenario": name, "data_seed": bank, "gp_seed": seed, "method": arm}
            trees.append(
                dict(
                    **key,
                    calls=cp.calls,
                    validation_calls=cp.n_trees,
                    path_digest=cp.path_digest,
                    nodes=tree.nodes_count,
                    providers=len(referenced_ids(tree.root)),
                    seconds=cp.elapsed_seconds,
                    program=serialize(tree.root),
                    last_six=[json.loads(r["program"]) for r in initial[-6:]],
                )
            )
            for split, (z, labels, _) in splits.items():
                pred = predict_tree(tree, z)
                if not np.isfinite(pred).all() or not np.allclose(pred.sum(1), 1):
                    raise ValueError("Invalid probabilities")
                loss = (
                    np.square(pred - np.eye(scenario.n_classes)[labels]).sum(1).mean()
                )
                if split == "validation" and not np.isclose(
                    loss, 2 * cp.validation_brier
                ):
                    raise AssertionError("Validation winner did not replay")
                rows.append(
                    dict(
                        **key,
                        split=split,
                        brier=loss,
                        prediction_sha256=fingerprint(pred),
                    )
                )
    return rows, trees


def summarize(df):
    bank = (
        df.groupby(["scenario", "data_seed", "split", "method"]).brier.mean().unstack()
    )
    means = bank.groupby(["scenario", "split"]).mean()
    gaps = (bank.mixed_branch - bank.mixed_chain).groupby(["scenario", "split"])
    paired = gaps.agg(["mean", "sem", "count"])
    margin = paired["sem"] * t.ppf(0.975, paired["count"] - 1)
    paired["ci_low"], paired["ci_high"] = (
        paired["mean"] - margin,
        paired["mean"] + margin,
    )
    means.to_csv(OUT / "scenario-means.csv")
    paired.drop(columns="sem").to_csv(OUT / "paired.csv")


def main(workers):
    start = perf_counter()
    jobs = [
        (name, bank)
        for name in SCENARIOS
        for bank in range(92300, 92308 if name == FOCAL else 92304)
    ]
    rows, trees = [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for index, (r, g) in enumerate(pool.map(panel, jobs), 1):
            rows.extend(r)
            trees.extend(g)
            if index % 20 == 0:
                print(f"{index}/{len(jobs)} data banks complete", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "scores.csv.gz", index=False)
    pd.DataFrame(trees).to_json(
        OUT / "trees.jsonl.gz", orient="records", lines=True, double_precision=15
    )
    summarize(df)
    print(
        f"{len(trees)} trajectories, {perf_counter() - start:.1f} wall seconds",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3)
    main(parser.parse_args().workers)

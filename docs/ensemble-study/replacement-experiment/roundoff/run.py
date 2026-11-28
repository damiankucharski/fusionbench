"""Repeat the raw-promotion arm, changing only the numerical threshold at zero."""

import argparse
import importlib.util
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from loguru import logger
from okapi.replacement import select_brier_replacement
from threadpoolctl import threadpool_limits

from fusionbench.eval.replacement_search import run_replacement_search
from fusionbench.eval.suite_point_gp import predict_tree

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
TOL = 1e-12


def load_inputs(name):
    spec = importlib.util.spec_from_file_location(
        f"{name}_inputs", STUDY / name / "inputs.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


controlled = load_inputs("controlled")
panel = load_inputs("panel")


def raw_with_tolerance(*args, **kwargs):
    result = select_brier_replacement(*args, **kwargs)
    assert result.rule == "raw"
    eligible = tuple(i for i, gain in enumerate(result.gains) if gain > TOL)
    return replace(
        result,
        eligible=eligible,
        index=max(eligible, key=result.gains.__getitem__, default=None),
    )


def run(job):
    suite, name, rep = job
    logger.disable("okapi")
    if suite == "controlled":
        fit = controlled.generate(rep, 21, 128)
        test = controlled.generate(rep, 41, 2048)
        x, y, seed = fit.predictions, fit.labels[name], controlled.SEARCH_BASE + rep
        samples = [controlled.generate(rep, i, 128) for i in range(31, 37)]
        blocks = [(b.predictions, b.labels[name]) for b in samples]
        incumbent = "mean"
    else:
        _, _, seed, full, labels, tx, ty = panel.load_bank(name, rep)
        x, y = full[:256], labels[:256]
        blocks = [(full[i : i + 64], labels[i : i + 64]) for i in range(256, 640, 64)]
        incumbent = "best_single"
    with (
        threadpool_limits(limits=1),
        patch("okapi.replacement.select_brier_replacement", raw_with_tolerance),
    ):
        result = run_replacement_search(
            x, y, blocks, seed=seed, policy="open_raw", incumbent=incumbent
        )
    row = {
        "suite": suite,
        "condition": name,
        "rep": rep,
        "path": result.path_digest,
        "tolerant_id": result.champion_id,
        "tolerant_promotions": sum(s.promoted is not None for s in result.generations),
    }
    if suite == "controlled":
        targets, _, _ = controlled.support_targets(rep)
        q = targets[name]
        p = predict_tree(result.tree, controlled.SUPPORT)[:, 1]
        metrics = controlled.score(p, q, test, name)
        excess = [
            float(
                2
                * np.square(
                    predict_tree(
                        result.records[g.incumbent_after].tree, controlled.SUPPORT
                    )[:, 1]
                    - q
                ).mean()
            )
            for g in result.generations
        ]
        row.update(
            tolerant_exact_brier=metrics["exact_brier"],
            tolerant_test_brier=metrics["test_brier"],
            tolerant_ever_nonoptimal=any(v > TOL for v in excess),
            tolerant_final_beats_mean=metrics["exact_excess"] + TOL
            < 2 * np.square(targets["null"] - q).mean(),
        )
    else:
        row["tolerant_test_brier"] = panel.metrics(predict_tree(result.tree, tx), ty)[
            "brier"
        ]
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2)
    workers = parser.parse_args().workers
    jobs = [("controlled", law, i) for law in controlled.POOLS for i in range(48)]
    jobs.extend(
        ("panel", name, i) for name in panel.SCENARIOS for i in range(panel.REPEATS)
    )
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = pd.DataFrame(pool.map(run, jobs))
    source = pd.read_csv(
        STUDY / "controlled/results/search.csv", keep_default_na=False, na_values=[""]
    ).query("policy == 'open_raw'")
    old = source.rename(
        columns={
            "law": "condition",
            "bank": "rep",
            "deploy_id": "primary_id",
            "path_digest": "primary_path",
            "exact_brier": "primary_exact_brier",
            "test_brier": "primary_test_brier",
            "ever_nonoptimal_after": "primary_ever_nonoptimal",
            "final_better_than_mean": "primary_final_beats_mean",
            "n_promotions": "primary_promotions",
        }
    ).assign(suite="controlled")
    source = pd.read_csv(STUDY / "panel/results/scores.csv").query(
        "method == 'open_raw'"
    )
    old_panel = source.rename(
        columns={
            "scenario": "condition",
            "champion_id": "primary_id",
            "path_digest": "primary_path",
            "brier": "primary_test_brier",
            "promotions": "primary_promotions",
        }
    ).assign(suite="panel")
    ids = ["suite", "condition", "rep"]
    primary = pd.concat((old, old_panel), ignore_index=True)
    merged = results.merge(
        primary[[*ids, *[c for c in primary.columns if c.startswith("primary_")]]],
        on=ids,
        validate="one_to_one",
    )
    assert len(merged) == len(jobs), f"Only {len(merged)} of {len(jobs)} banks joined"
    mismatch = merged.loc[merged.path != merged.primary_path, ids]
    assert mismatch.empty, f"Native path differs on {mismatch.to_dict('records')}"
    merged = merged.drop(columns="primary_path")
    for metric in ("exact_brier", "test_brier", "promotions"):
        merged["delta_" + metric] = (
            merged["tolerant_" + metric] - merged["primary_" + metric]
        )
    merged.to_csv(HERE / "banks.csv", index=False)


if __name__ == "__main__":
    main()

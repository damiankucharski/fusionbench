"""Evaluate the single native OKAPI tree and external reference methods separately."""

import argparse
import gzip
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from fusionbench.methods.native_pattern import NativePatternOKAPI
from loguru import logger
from scipy.stats import t

from fusionbench.eval.full_suite_point import SHIFTS, fingerprint, scenario_block
from fusionbench.eval.native_switch import serialize
from fusionbench.methods.averaging import TopN
from fusionbench.methods.stacking import StackingLogit, StackingLR
from fusionbench.scenarios.registry import SCENARIOS

OUT = Path(__file__).resolve().parent
REFERENCES = ["logit", "prob_lr", "gp", "flat", "augmented", "top_n", "mean"]


def native_program(root):
    result = serialize(root)

    def add_parameters(node, record):
        if type(node).__name__ == "BinaryAffineNode":
            record.update(
                coefficients=list(node.coefficients),
                intercept=node.intercept,
                transform=node.transform,
            )
        for child, item in zip(node.children, record["children"], strict=True):
            add_parameters(child, item)

    add_parameters(root, result)
    return result


def plain(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(type(obj).__name__)


def panel(job):
    logger.disable("okapi")
    idx, name, rep = job
    seed = 1460000 + 1000 * idx + rep
    key = {"scenario": name, "data_seed": seed, "replicate": rep}
    scenario = SCENARIOS[name]()
    c = scenario.n_classes
    task = "binary" if c == 2 else "multiclass"
    x, y, _ = scenario_block(scenario, seed, 0, 640)
    start = perf_counter()
    method = NativePatternOKAPI(seed=seed).fit(x, y, task=task, n_classes=c)
    elapsed = perf_counter() - start
    start = perf_counter()
    baseline = {
        label: m.fit(x, y, task=task, n_classes=c)
        for label, m in [
            ("logit", StackingLogit()),
            ("prob_lr", StackingLR()),
            ("top_n", TopN("brier")),
        ]
    }
    baseline_seconds = perf_counter() - start
    top = baseline["top_n"]
    assert top.n_selected_ == int(np.argmin(top.prefix_scores_)) + 1
    assert len(top.prefix_scores_) == x.shape[1]
    meta = {
        **key,
        "chosen": method.chosen_,
        "trees": {k: native_program(tree.root) for k, tree in method.trees_.items()},
        "discovery_trees": {
            k: native_program(tree.root) for k, tree in method.discovery_trees_.items()
        },
        "counts": method.counts_,
        "top_n": int(top.n_selected_),
        "ranking": top.ranking_.tolist(),
        "prefix_brier": top.prefix_scores_.tolist(),
        "fit_seconds": elapsed,
        "controls_seconds": baseline_seconds,
    }
    rows = []
    blocks = [("iid", scenario, 1)]
    if name in SHIFTS:
        blocks.append(("shift", SCENARIOS[name](**SHIFTS[name]), 2))
    for split, generator, stream in blocks:
        assert generator.model_ids() == scenario.model_ids()
        z, labels, _ = scenario_block(generator, seed, stream, 4096)
        pmap = method.component_predictions(z)
        deployed = method.predict(z)
        if not np.array_equal(pmap[method.chosen_], deployed):
            raise AssertionError("Wrapper differs from a single chosen native tree")
        pmap["native"] = deployed
        pmap.update({label: m.predict(z) for label, m in baseline.items()})
        pmap["mean"] = z.mean(1)
        target = np.eye(c)[labels]
        for label, p in pmap.items():
            if (
                p.shape != target.shape
                or not np.isfinite(p).all()
                or p.min() < -1e-12
                or not np.allclose(p.sum(1), 1, rtol=0, atol=2e-12)
            ):
                raise ValueError(f"Invalid probabilities in {name}/{label}")
            rows.append(
                {
                    **key,
                    "split": split,
                    "method": label,
                    "brier": np.square(p - target).sum(1).mean(),
                    "prediction_sha256": fingerprint(p),
                }
            )
    return rows, meta


def summarize(df):
    pivot = df.pivot(
        index=["scenario", "replicate", "split"], columns="method", values="brier"
    )
    pivot.groupby(["scenario", "split"]).mean().to_csv(OUT / "families.csv")
    contrasts = []
    for ref in REFERENCES:
        gap = pivot.native - pivot[ref]
        for split, v in gap.groupby("split"):
            repl = v.groupby("replicate").mean()
            fam = v.groupby("scenario").mean()
            half = t.ppf(0.975, len(repl) - 1) * repl.sem()
            loo = (fam.sum() - fam) / (len(fam) - 1)
            contrasts.append(
                {
                    "reference": ref,
                    "split": split,
                    "delta": repl.mean(),
                    "low": repl.mean() - half,
                    "high": repl.mean() + half,
                    "wins": int((fam < 0).sum()),
                    "families": len(fam),
                    "loo_min": loo.min(),
                    "loo_max": loo.max(),
                    "worst": fam.idxmax(),
                    "worst_delta": fam.max(),
                }
            )
    pd.DataFrame(contrasts).to_csv(OUT / "contrasts.csv", index=False)


def main(workers):
    start = perf_counter()
    jobs = [(i, name, rep) for i, name in enumerate(SCENARIOS) for rep in range(12)]
    scores, manifests = [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, (rows, meta) in enumerate(pool.map(panel, jobs), 1):
            scores.extend(rows)
            manifests.append(meta)
            if i % 48 == 0:
                print(f"{i}/{len(jobs)} banks complete", flush=True)
    df = pd.DataFrame(scores)
    df.to_csv(OUT / "scores.csv.gz", index=False)
    with gzip.open(OUT / "banks.jsonl.gz", "wt") as handle:
        for meta in manifests:
            print(json.dumps(meta, default=plain, separators=(",", ":")), file=handle)
    summarize(df)
    print(
        f"{len(jobs)} banks; {len(scores)} outcomes; {perf_counter() - start:.1f} wall seconds",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3)
    main(parser.parse_args().workers)

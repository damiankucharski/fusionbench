"""Run the frozen benchmark and report equal-family contrasts by independent replicate."""

import argparse
import gzip
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import t

from fusionbench.eval.full_suite_point import SHIFTS, fingerprint, scenario_block
from fusionbench.eval.native_switch import serialize
from fusionbench.methods.averaging import TopN
from fusionbench.methods.crossfit_okapi import CrossFitOKAPI
from fusionbench.scenarios.registry import SCENARIOS

OUT = Path(__file__).resolve().parent
REFERENCES = ["logit", "prob_lr", "linear_blend", "gp", "mean", "top_n"]


def panel(job):
    logger.disable("okapi")
    idx, name, rep = job
    seed = 1320000 + 1000 * idx + rep
    source = SCENARIOS[name]()
    classes = source.n_classes
    task = "binary" if classes == 2 else "multiclass"
    x, y, _ = scenario_block(source, seed, 0, 640)
    start = perf_counter()
    method = CrossFitOKAPI(seed).fit(x, y, task=task, n_classes=classes)
    elapsed = perf_counter() - start
    top = TopN("brier").fit(x, y, task=task, n_classes=classes)
    if top.n_selected_ != int(np.argmin(top.prefix_scores_)) + 1:
        raise AssertionError("Bad Top-N prefix choice")
    key = {"scenario": name, "replicate": rep, "data_seed": seed}
    info = dict(
        **key,
        weights=method.weights_.tolist(),
        linear_weights=method.linear_weights_.tolist(),
        folds=method.fold_records_,
        tree=serialize(method.models_[2].root),
        top_n=int(top.n_selected_),
        ranking=top.ranking_.tolist(),
        prefix_brier=top.prefix_scores_.tolist(),
        fit_seconds=elapsed,
    )
    rows = []
    tests = [("iid", source, 1)]
    if name in SHIFTS:
        tests.append(("shift", SCENARIOS[name](**SHIFTS[name]), 2))
    for split, scenario, stream in tests:
        assert scenario.model_ids() == source.model_ids()
        z, labels, _ = scenario_block(scenario, seed, stream, 4096)
        comp = method.component_predictions(z)
        predictions = {
            "mean": z.mean(1),
            "top_n": top.predict(z),
            "logit": comp[:, 0],
            "prob_lr": comp[:, 1],
            "gp": comp[:, 2],
            "linear_blend": np.einsum("nmc,m->nc", comp[:, :2], method.linear_weights_),
            "hybrid": method.predict(z),
        }
        for label, p in predictions.items():
            if (
                not np.isfinite(p).all()
                or p.min() < -1e-12
                or not np.allclose(p.sum(1), 1, atol=2e-12, rtol=0)
            ):
                raise ValueError("Invalid class probabilities")
            brier = np.square(p - np.eye(classes)[labels]).sum(1).mean()
            rows.append(
                dict(
                    **key,
                    split=split,
                    method=label,
                    brier=brier,
                    prediction_sha256=fingerprint(p),
                )
            )
    return rows, info, method.oof_


def summarize(df):
    wide = df.pivot(
        index=["scenario", "replicate", "split"], columns="method", values="brier"
    )
    means = wide.groupby(["scenario", "split"]).mean()
    means.to_csv(OUT / "families.csv")
    out = []
    for reference in REFERENCES:
        gap = wide.hybrid - wide[reference]
        for split, v in gap.groupby("split"):
            banks = v.groupby("replicate").mean()
            by_family = v.groupby("scenario").mean()
            half = t.ppf(0.975, len(banks) - 1) * banks.sem()
            loo = (by_family.sum() - by_family) / (len(by_family) - 1)
            out.append(
                {
                    "split": split,
                    "reference": reference,
                    "delta": banks.mean(),
                    "low": banks.mean() - half,
                    "high": banks.mean() + half,
                    "wins": int((by_family < 0).sum()),
                    "families": len(by_family),
                    "loo_min": loo.min(),
                    "loo_max": loo.max(),
                    "worst": by_family.idxmax(),
                    "worst_delta": by_family.max(),
                }
            )
    pd.DataFrame(out).to_csv(OUT / "contrasts.csv", index=False)


def main(workers):
    start = perf_counter()
    jobs = [(i, n, r) for i, n in enumerate(SCENARIOS) for r in range(12)]
    scores, manifests, oof = [], [], {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, (rows, meta, p) in enumerate(pool.map(panel, jobs), 1):
            scores.extend(rows)
            manifests.append(meta)
            oof["b" + str(meta["data_seed"])] = p
            if i % 48 == 0:
                print(f"{i}/{len(jobs)} banks finished", flush=True)
    np.savez_compressed(OUT / "oof.npz", **oof)
    with gzip.open(OUT / "banks.jsonl.gz", "wt") as out:
        for meta in manifests:
            print(json.dumps(meta, separators=(",", ":")), file=out)
    df = pd.DataFrame(scores)
    df.to_csv(OUT / "scores.csv.gz", index=False)
    summarize(df)
    print(
        f"{len(manifests)} banks, {len(scores)} held-out scores, {perf_counter() - start:.1f} wall seconds"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3)
    main(parser.parse_args().workers)

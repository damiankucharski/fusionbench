"""Run the committed default scenario suite as regression cases for point mutation."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

from fusionbench.eval.metrics import score
from fusionbench.eval.native_switch import serialize
from fusionbench.methods.averaging import TopN
from fusionbench.methods.stacking import StackingLogit
from fusionbench.scenarios.registry import SCENARIOS

SHIFTS = {
    "two_pathways": {"accuracy": 0.78},
    "parity": {"accuracy": 0.78},
    "weakest_link": {"accuracy": 0.76},
    "correlated": {"corr": 0.1},
    "calibrated_suppressor": {"corr": 0.0},
    "polarity_fault": {"correlation": 0.0},
    "native_switch": {"mixing": 1.0},
    "native_close_switch": {"mixing": 1.0},
    "compositional_switch": {"corner_probability": 0.0},
}


def scenario_block(scenario, seed, index, n):
    preds, labels, latent = scenario._generate(n, np.random.default_rng([seed, index]))
    q = scenario.bayes_oracle(preds, latent)
    if q is not None:
        q = np.asarray(q)
        assert q.shape == (len(labels), scenario.n_classes)
    return preds, labels, q


def data_for(name, seed):
    scenario = SCENARIOS[name]()
    sub = {
        s: scenario_block(scenario, seed, i, n)
        for i, (s, n) in enumerate([("fit", 384), ("validation", 256), ("iid", 2048)])
    }
    if name in SHIFTS:
        alternative = SCENARIOS[name](**SHIFTS[name])
        if (
            scenario.model_ids() != alternative.model_ids()
            or scenario.n_classes != alternative.n_classes
        ):
            raise ValueError(f"{name} shift changed provider mapping")
        sub["shift"] = scenario_block(alternative, seed, 3, 2048)
    return scenario, sub


def fingerprint(array):
    return hashlib.sha256(np.asarray(array, dtype="<f8").tobytes()).hexdigest()


def panel(pair):
    from loguru import logger

    from fusionbench.eval.suite_point_gp import fit_suite_point, predict_tree

    logger.disable("okapi")
    name, seed = pair
    scenario, sub = data_for(name, seed)
    C = scenario.n_classes
    x, y, _ = sub["fit"]
    rows = []

    def record(method, predictions, **meta):
        for split, proba in predictions.items():
            p = np.asarray(proba)
            _, labels, q = sub[split]
            assert p.shape == (len(labels), C)
            assert np.all(np.isfinite(p)) and np.all((p >= -1e-14) & (p <= 1 + 1e-14))
            np.testing.assert_allclose(p.sum(1), 1, rtol=0, atol=2e-12)
            standard = score(labels, p, C, metrics=["macro_f1", "log_loss", "accuracy"])
            joined = {
                **standard,
                "brier": np.mean(np.sum((p - np.eye(C)[labels]) ** 2, axis=1)),
                "conditional_excess": np.nan
                if q is None
                else np.mean(np.sum((p - q) ** 2, axis=1)),
            }
            rows.append(
                {
                    "scenario": name,
                    "data_seed": seed,
                    "method": method,
                    "split": split,
                    "classes": C,
                    "providers": x.shape[1],
                    "rows": len(labels),
                    "prediction_sha256": fingerprint(p),
                    "oracle_sha256": None if q is None else fingerprint(q),
                    **joined,
                    **meta,
                }
            )

    top = TopN("brier").fit(x, y, task=scenario.task, n_classes=C)
    stack = StackingLogit().fit(x, y, task=scenario.task, n_classes=C)
    basic = {"selection": "fit", "checkpoint": "fixed"}
    record("mean", {s: pp.mean(1) for s, (pp, _, _) in sub.items()}, **basic)
    record(
        "top_n",
        {s: top.predict(pp) for s, (pp, _, _) in sub.items()},
        **basic,
        n_selected=top.n_selected_,
        ranking=json.dumps(top.ranking_.tolist()),
        prefix_scores=json.dumps(top.prefix_scores_.tolist()),
    )
    record(
        "stack_logit", {s: stack.predict(pp) for s, (pp, _, _) in sub.items()}, **basic
    )
    for g in (541, 547):
        for enabled in (False, True):
            checkpoints = fit_suite_point(
                x,
                y,
                sub["validation"][0],
                sub["validation"][1],
                seed=g,
                point_mutation=enabled,
            )
            for calls, cp in checkpoints.items():
                for selection, tree in (
                    ("fit", cp.fit_tree),
                    ("validation", cp.validation_tree),
                ):
                    model = {s: predict_tree(tree, pp) for s, (pp, _, _) in sub.items()}
                    record(
                        "point_on" if enabled else "point_off",
                        model,
                        selection=selection,
                        checkpoint=str(calls),
                        calls=cp.calls,
                        gp_seed=g,
                        nodes=tree.nodes_count,
                        archive_trees=cp.n_trees,
                        generation=cp.generation,
                        search_seconds=cp.elapsed_seconds,
                        path_digest=cp.path_digest,
                        program=json.dumps(serialize(tree.root)),
                    )
    return rows


def summarize(frame):
    keys = ["scenario", "checkpoint", "selection", "split"]
    metrics = [
        "brier",
        "conditional_excess",
        "macro_f1",
        "accuracy",
        "log_loss",
        "nodes",
    ]
    means = (
        frame.groupby([*keys, "method", "data_seed"])[metrics]
        .mean()
        .groupby([*keys, "method"])
        .mean()
        .reset_index()
    )
    extra = []
    for key, g in frame[frame.gp_seed.notna()].groupby(keys):
        for met in metrics:
            table = g.pivot(
                index=["data_seed", "gp_seed"], columns="method", values=met
            )
            bank = (table.point_on - table.point_off).groupby("data_seed").mean()
            ci = t.ppf(0.975, len(bank) - 1) * bank.sem()
            extra.append(
                dict(zip(keys, key, strict=True))
                | {
                    "metric": met,
                    "banks": len(bank),
                    "runs": len(table),
                    "difference": bank.mean(),
                    "sd_across_banks": bank.std(),
                    "low_95": bank.mean() - ci,
                    "high_95": bank.mean() + ci,
                    "banks_on_better": int((bank < 0).sum())
                    if met not in ("macro_f1", "accuracy")
                    else int((bank > 0).sum()),
                }
            )
    return means, pd.DataFrame(extra)


def score_spread(frame):
    """Spread of each frozen method across data draws, averaging GP restarts first."""
    keys = ["scenario", "split", "checkpoint", "selection", "method"]
    metrics = ["brier", "conditional_excess", "log_loss", "accuracy", "macro_f1"]
    bank = frame.groupby([*keys, "data_seed"])[metrics].mean().reset_index()
    long = bank.melt(
        id_vars=[*keys, "data_seed"],
        value_vars=metrics,
        var_name="metric",
        value_name="score",
    )
    spread = (
        long.groupby([*keys, "metric"])
        .score.agg(["count", "mean", "std"])
        .reset_index()
    )
    spread = spread.rename(columns={"count": "banks", "std": "sd_across_banks"})
    half = (
        t.ppf(0.975, spread.banks - 1) * spread.sd_across_banks / np.sqrt(spread.banks)
    )
    return spread.assign(low_95=spread["mean"] - half, high_95=spread["mean"] + half)


def run(out, workers):
    out.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        df = pd.DataFrame(
            [
                row
                for batch in pool.map(panel, product(SCENARIOS, range(4600, 4608)))
                for row in batch
            ]
        )
    df.to_csv(out / "runs.csv.gz", index=False)
    means, paired = summarize(df)
    means.to_csv(out / "means.csv", index=False)
    paired.to_csv(out / "paired.csv", index=False)
    score_spread(df).to_csv(out / "spread.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--workers", default=1, type=int)
    args = parser.parse_args()
    run(args.out, args.workers)

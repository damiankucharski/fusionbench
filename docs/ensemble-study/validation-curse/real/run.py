"""Run the locked small-development-sample experiment on existing held-out provider banks."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import runpy
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from fusionbench.methods.averaging import SimpleAverage, TopN
from fusionbench.methods.native_pattern import RegularizedNativePatternOKAPI
from fusionbench.methods.stacking import StackingLogit, StackingLR

STUDY = Path(__file__).resolve().parent
POOLS = ("digits_hetero_a", "digits_lr_bag", "digits_hetero_b", "wine", "iris", "seeds")
CANDIDATES = ("flat", "augmented", "gp")


@lru_cache(maxsize=1)
def archive():
    # Reuse the loader that defines and checks the published bank records.
    return runpy.run_path(str(STUDY.parents[1] / "class-capacity" / "run.py"))


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def row_brier(pred, labels):
    pred = np.asarray(pred, dtype=np.float64)
    labels = np.asarray(labels)
    if pred.ndim != 2 or len(pred) != len(labels) or not np.all(np.isfinite(pred)):
        raise ValueError("Invalid prediction array")
    if np.min(pred) < -1e-10 or not np.allclose(pred.sum(axis=1), 1, rtol=0, atol=2e-6):
        raise ValueError("Expected class probabilities")
    return np.square(pred - np.eye(pred.shape[1])[labels]).sum(axis=1)


def source_ids(pool, replicate):
    source, start, fit_dir, test_dir = archive()["REAL"][pool]
    key = f"{source}_{start + replicate}_rows"
    with np.load(fit_dir / "fit_banks.npz", allow_pickle=False) as data:
        fit_rows = data[key]
    with np.load(test_dir / "test_banks.npz", allow_pickle=False) as data:
        test_rows = data[key]
    if np.intersect1d(fit_rows, test_rows).size:
        raise AssertionError("Unseen test rows overlap fusion training rows")
    return fit_rows, test_rows


def run_bank(job):
    from fusionbench.methods.search_guard import SearchGuardedNativeOKAPI
    from fusionbench.methods.validation_guard import GuardedNativeOKAPI

    pool, rep = job
    bank, seed = archive()["load_bank"](pool, rep)
    fit_ids, test_ids = source_ids(pool, rep)
    n = min(64, len(fit_ids))
    if n < len(fit_ids):
        pos = np.sort(
            np.random.default_rng([seed, 96301]).choice(len(fit_ids), n, replace=False)
        )
    else:
        pos = np.arange(n)
    x, y = bank.fit_preds[pos], bank.fit_y[pos]
    train = {"task": bank.task, "n_classes": bank.n_classes}
    constructors = {
        "mean": SimpleAverage,
        "top_n": lambda: TopN("brier"),
        "prob_lr": StackingLR,
        "logit_lr": StackingLogit,
        "incumbent": lambda: RegularizedNativePatternOKAPI(seed=seed, split_seed=seed),
        "search_z0": lambda: SearchGuardedNativeOKAPI(
            seed=seed, split_seed=seed, z=0.0
        ),
        "search_z1": lambda: SearchGuardedNativeOKAPI(
            seed=seed, split_seed=seed, z=1.0
        ),
        "guarded": lambda: GuardedNativeOKAPI(seed=seed, split_seed=seed),
    }
    estimators, times = {}, {}
    for name, make in constructors.items():
        start = perf_counter()
        estimators[name] = make().fit(x, y, **train)
        times[name] = perf_counter() - start

    # No held-out model prediction is evaluated until every estimator is already frozen.
    xt, yt = bank.test_preds, bank.test_y
    predictions, scores, loss_arrays = {}, [], {}
    info = {
        "pool": pool,
        "dataset": "digits" if pool.startswith("digits_") else pool,
        "replicate": rep,
        "search_seed": seed,
        "n_available": len(fit_ids),
        "n_fit": n,
        "n_test": len(yt),
        "n_classes": bank.n_classes,
    }

    def record(name, pred, *, choice="", evaluations=None, seconds=None):
        p = np.asarray(pred)
        losses = row_brier(p, yt)
        predictions[name] = p
        loss_arrays[name] = losses
        scores.append(
            {
                **info,
                "method": name,
                "test_brier": losses.mean(),
                "test_accuracy": np.mean(p.argmax(axis=1) == yt),
                "choice": choice,
                "fit_seconds": seconds,
                "score_calls": evaluations,
                "prediction_sha256": digest(p),
            }
        )

    for name in ("mean", "top_n", "prob_lr", "logit_lr"):
        model = estimators[name]
        record(
            name,
            model.predict(xt),
            choice=str(model.n_selected_) if name == "top_n" else "",
            seconds=times[name],
        )
    audit = {
        **info,
        "provider_ids": bank.model_ids,
        "fit_rows": fit_ids[pos].tolist(),
        "test_rows_sha256": digest(test_ids),
        "topn_ranking": estimators["top_n"].ranking_.tolist(),
        "topn_prefix_brier": estimators["top_n"].prefix_scores_.tolist(),
    }
    finders = []
    for name in ("incumbent", "search_z0", "search_z1"):
        model = estimators[name]
        component = model.component_predictions(xt)
        for key in CANDIDATES:
            record(f"{name}_{key}", component[key])
        record(
            name,
            component[model.chosen_],
            choice=model.chosen_,
            seconds=times[name],
            evaluations=model.counts_["total_score_calls"],
        )
        if not np.array_equal(model.predict(xt), predictions[name]):
            raise AssertionError("Publicly deployed and archived program disagree")
        selected_risk = loss_arrays[name].mean()
        oracle = min(CANDIDATES, key=lambda key: loss_arrays[f"{name}_{key}"].mean())
        oracle_loss = loss_arrays[f"{name}_{oracle}"].mean()
        finders.append(
            {
                **info,
                "search_method": name,
                "selected": model.chosen_,
                "test_brier": selected_risk,
                "oracle_among_three": oracle,
                "test_three_oracle_brier": oracle_loss,
                "selection_regret": selected_risk - oracle_loss,
                "test_oracle_if_mean_allowed": min(
                    oracle_loss, loss_arrays["mean"].mean()
                ),
            }
        )
        audit[name] = {
            "selected": model.chosen_,
            "selection_row_ids": fit_ids[pos][model.validation_indices_].tolist(),
            "score_calls": model.counts_["total_score_calls"],
        }
    guard = estimators["guarded"]
    outer = np.asarray(guard.outer_indices_, dtype=int)
    inner = np.asarray(guard.inner_indices_, dtype=int)
    if not np.array_equal(np.sort(np.r_[inner, outer]), np.arange(n)):
        raise AssertionError(
            "Guard fit and gate indices do not partition the development data"
        )
    gt = guard.candidate_predictions(xt)
    gv = guard.candidate_predictions(x[outer])
    gate_mean = row_brier(gv["mean"], y[outer]).mean()
    test_mean = row_brier(gt["mean"], yt).mean()
    if not np.allclose(gt["mean"], predictions["mean"], atol=2e-7, rtol=0):
        raise AssertionError("Native mean differs from the arithmetic comparator")
    for key in ("mean", *CANDIDATES):
        record("outer_" + key, gt[key])
    gate_rows = []
    choices = {
        "guarded": guard.chosen_,
        "raw_plus_mean": guard.raw_choice_,
        "inner_incumbent": guard.estimator_.chosen_,
    }
    if not np.array_equal(guard.predict(xt), gt[guard.chosen_]):
        raise AssertionError(
            "Guarded public predictions changed after candidate evaluation"
        )
    for name, key in choices.items():
        gate_risk = row_brier(gv[key], y[outer]).mean()
        test_risk = row_brier(gt[key], yt).mean()
        record(
            name,
            gt[key],
            choice=key,
            seconds=times["guarded"],
            evaluations=guard.estimator_.counts_["total_score_calls"],
        )
        gate_rows.append(
            {
                **info,
                "method": name,
                "selected": key,
                "outer_n": len(outer),
                "gate_brier": gate_risk,
                "test_brier": test_risk,
                "gate_minus_mean": gate_risk - gate_mean,
                "test_minus_mean": test_risk - test_mean,
                "relative_optimism": (test_risk - test_mean) - (gate_risk - gate_mean),
            }
        )
    audit.update(
        inner_ids=fit_ids[pos][inner].tolist(),
        outer_ids=fit_ids[pos][outer].tolist(),
        outer_choices=choices,
        outer_gate_brier={
            key: float(row_brier(p, y[outer]).mean()) for key, p in gv.items()
        },
    )
    return scores, gate_rows, finders, audit, loss_arrays


def worker(job):
    from loguru import logger

    logger.disable("okapi")
    with threadpool_limits(limits=1):
        return run_bank(job)


def run(out, workers):
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(pool, rep) for pool in POOLS for rep in range(8)]
    scores, gates, finders, audits, losses = [], [], [], [], {}
    with ProcessPoolExecutor(max_workers=workers) as proc:
        for (pool, rep), (rows, gate, ref, meta, vecs) in zip(
            jobs, proc.map(worker, jobs), strict=True
        ):
            scores.extend(rows)
            gates.extend(gate)
            finders.extend(ref)
            audits.append(meta)
            for name, v in vecs.items():
                losses[f"{pool}_{rep}_{name}"] = v
    pd.DataFrame(scores).to_csv(out / "scores.csv.gz", index=False)
    pd.DataFrame(gates).to_csv(out / "gate.csv", index=False)
    pd.DataFrame(finders).to_csv(out / "final-family-oracle.csv", index=False)
    np.savez_compressed(out / "test-row-losses.npz", **losses)
    with gzip.open(out / "banks.jsonl.gz", "wt") as f:
        for a in audits:
            f.write(json.dumps(a, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=STUDY / "results")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    run(args.out, args.workers)

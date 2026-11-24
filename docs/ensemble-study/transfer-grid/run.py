"""Evaluate frozen selectors and their discovery candidates on a paired transfer grid."""

import argparse
import csv
import gzip
import json
import os
import runpy
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
from loguru import logger
from threadpoolctl import threadpool_limits

from fusionbench.eval.transfer_grid import (
    SOURCE_NOISE,
    development,
    iter_test_conditions,
)
from fusionbench.methods.averaging import TopN
from fusionbench.methods.stacking import StackingLogit, StackingLR
from fusionbench.transfer import load_preset

HERE = Path(__file__).resolve().parent
CANDIDATES = ("flat", "augmented", "gp", "half")
IDENTITY = ("scenario", "replicate", "data_seed", "noise")
CONDITION = (*IDENTITY, "mode", "severity")
RESULTS = (*CONDITION, "method", "brier")
DIAGNOSTICS = (
    *CONDITION,
    "old_choice",
    "robust_choice",
    *(f"val_{k}" for k in CANDIDATES),
    *(f"test_{k}" for k in CANDIDATES),
)
COSTS = (
    *IDENTITY,
    "old_choice",
    "robust_choice",
    "development_rows",
    "input_providers",
    "classes",
    "training_rows_corrupted",
    "native_fit_seconds",
    "prob_lr_fit_seconds",
    "logit_fit_seconds",
    "top_n_fit_seconds",
    "fit_seconds",
    "evaluation_seconds",
    "total_seconds",
    "conditions",
    "test_rows_per_condition",
    "gp_discovery_calls",
    "base_score_calls",
    "top_n_prefixes",
    "top_n_selected",
    "additional_score_calls",
    "total_score_calls",
    "probe_scores",
    "base_counts",
    "transfer_counts",
)
SEED_BASE = 1_700_000


def plain(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(type(obj).__name__)


@lru_cache(maxsize=1)
def serializer():
    return runpy.run_path(str(HERE.parent / "native-pattern" / "run.py"))[
        "native_program"
    ]


def full_brier(p, target):
    p = np.asarray(p, dtype=np.float64)
    if (
        p.shape != target.shape
        or not np.all(np.isfinite(p))
        or np.any(p < 0)
        or np.any(p > 1)
        or not np.allclose(p.sum(axis=1), 1, atol=1e-6, rtol=0)
    ):
        raise ValueError("Expected one normalized probability vector per test label")
    return float(np.mean(np.sum(np.square(p - target), axis=1)))


@dataclass
class Fit:
    identity: dict
    native: object
    references: dict
    validation: dict
    cost: dict


def fit_one(identity, x, y, clean_y):
    from fusionbench.methods.transfer_pattern import TransferPatternOKAPI

    c = x.shape[2]
    options = {"task": "binary" if c == 2 else "multiclass", "n_classes": c}
    start = perf_counter()
    native = TransferPatternOKAPI(seed=identity["data_seed"]).fit(x, y, **options)
    native_seconds = perf_counter() - start
    base = native.base_
    if set(native.discovery_trees_) != set(CANDIDATES) or set(native.trees_) != set(
        CANDIDATES
    ):
        raise ValueError("Expected four discovery and four fitted candidates")
    if (
        base.chosen_ not in ("flat", "augmented", "gp")
        or native.chosen_ not in CANDIDATES
    ):
        raise ValueError("A selector returned an unknown candidate")
    if set(native.clean_scores_) != set(CANDIDATES):
        raise ValueError("Missing discovery validation score")
    validation = {f"val_{k}": 2.0 * (1.0 - native.clean_scores_[k]) for k in CANDIDATES}
    if not all(
        np.isfinite(v) and -1e-12 <= v <= 2 + 1e-12 for v in validation.values()
    ):
        raise ValueError("Invalid discovery Brier on the validation rows")
    references = {}
    durations = {}
    for name, model in (
        ("prob_lr", StackingLR()),
        ("logit", StackingLogit()),
        ("top_n", TopN("brier")),
    ):
        t0 = perf_counter()
        references[name] = model.fit(x, y, **options)
        durations[f"{name}_fit_seconds"] = perf_counter() - t0
    top = references["top_n"]
    if (
        len(top.prefix_scores_) != x.shape[1]
        or int(np.argmin(top.prefix_scores_)) + 1 != top.n_selected_
    ):
        raise AssertionError("Top-N must evaluate every training-ranked prefix")
    cost = dict(
        **identity,
        old_choice=base.chosen_,
        robust_choice=native.chosen_,
        development_rows=len(y),
        input_providers=x.shape[1],
        classes=c,
        training_rows_corrupted=int(np.count_nonzero(y != clean_y)),
        native_fit_seconds=native_seconds,
        **durations,
        fit_seconds=perf_counter() - start,
        evaluation_seconds=0.0,
        conditions=0,
        gp_discovery_calls=base.counts_["score_calls"]["gp_discovery"],
        base_score_calls=base.counts_["total_score_calls"],
        top_n_prefixes=len(top.prefix_scores_),
        top_n_selected=int(top.n_selected_),
        additional_score_calls=native.counts_["additional_score_calls"],
        total_score_calls=native.counts_["total_score_calls"],
        probe_scores=json.dumps(native.scores_, default=plain),
        base_counts=json.dumps(base.counts_, default=plain),
        transfer_counts=json.dumps(native.counts_, default=plain),
    )
    return Fit(identity, native, references, validation, cost)


def evaluate(fit, mode, severity, x, target):
    from fusionbench.eval.suite_point_gp import predict_tree

    start = perf_counter()
    method = fit.native
    old_choice, robust_choice = method.base_.chosen_, method.chosen_
    tree_scores = {}
    predictions = {}
    for name in CANDIDATES:
        tree = method.trees_[name]
        predictions[name] = predict_tree(tree, x)
        tree_scores[id(tree)] = full_brier(predictions[name], target)
    deployed = method.base_.predict(x)
    robust = method.predict(x)
    if not np.array_equal(deployed, predictions[old_choice]) or not np.array_equal(
        robust, predictions[robust_choice]
    ):
        raise AssertionError(
            "Deployed probabilities must equal the corresponding single fitted tree"
        )
    scores = {
        "deployed": tree_scores[id(method.trees_[old_choice])],
        "robust": tree_scores[id(method.trees_[robust_choice])],
    }
    scores.update({name: tree_scores[id(method.trees_[name])] for name in CANDIDATES})
    scores.update(
        {
            name: full_brier(model.predict(x), target)
            for name, model in fit.references.items()
        }
    )
    scores["mean"] = full_brier(x.mean(axis=1), target)
    row = dict(**fit.identity, mode=mode, severity=severity)
    diagnostics = dict(
        **row, old_choice=old_choice, robust_choice=robust_choice, **fit.validation
    )
    for name in CANDIDATES:
        tree = method.discovery_trees_[name]
        if id(tree) not in tree_scores:
            tree_scores[id(tree)] = full_brier(predict_tree(tree, x), target)
        diagnostics[f"test_{name}"] = tree_scores[id(tree)]
    fit.cost["evaluation_seconds"] += perf_counter() - start
    fit.cost["conditions"] += 1
    return [
        dict(**row, method=name, brier=b) for name, b in scores.items()
    ], diagnostics


def sample_record(fit):
    native_program = serializer()
    return dict(
        **fit.identity,
        old_choice=fit.native.base_.chosen_,
        robust_choice=fit.native.chosen_,
        trees={
            name: native_program(fit.native.trees_[name].root) for name in CANDIDATES
        },
        discovery_trees={
            name: native_program(fit.native.discovery_trees_[name].root)
            for name in CANDIDATES
        },
    )


def panel(job):
    logger.disable("okapi")
    idx, name, rep, n_development, n_test = job
    seed = SEED_BASE + 1000 * idx + rep
    fitted, clean_x, clean_y = [], None, None
    with threadpool_limits(1):
        for noise in SOURCE_NOISE:
            x, y = development(name, seed, n_development, noise)
            y = np.asarray(y).ravel()
            if clean_x is None:
                if noise != 0.0:
                    raise ValueError("Clean annotations must precede noisy ones")
                clean_x, clean_y = x, y
            elif not np.array_equal(clean_x, x):
                raise AssertionError(
                    "Development inputs must be shared across annotation noise levels"
                )
            identity = {
                "scenario": name,
                "replicate": rep,
                "data_seed": seed,
                "noise": noise,
            }
            fitted.append(fit_one(identity, x, y, clean_y))
        samples = [sample_record(f) for f in fitted] if rep == 0 else []
        rows, diagnostics = [], []
        for mode, severity, x, y in iter_test_conditions(name, seed, n_test):
            target = np.eye(x.shape[2])[np.asarray(y).ravel()]
            for f in fitted:
                scores, diag = evaluate(f, mode, severity, x, target)
                rows.extend(scores)
                diagnostics.append(diag)
        for f in fitted:
            f.cost["test_rows_per_condition"] = n_test
            f.cost["total_seconds"] = (
                f.cost["fit_seconds"] + f.cost["evaluation_seconds"]
            )
    return rows, diagnostics, [f.cost for f in fitted], samples


def run(output, workers):
    if not 1 <= workers <= 8:
        raise ValueError("The benchmark supports one to eight concurrent bank workers")
    output.mkdir(parents=True, exist_ok=True)
    suite = load_preset("standard")
    jobs = [
        (spec["seed_slot"], name, r, suite["fit_rows"], suite["test_rows"])
        for name, spec in suite["scenarios"].items()
        for r in range(suite["repeats"])
    ]
    files = (
        "results.csv.gz",
        "diagnostics.csv.gz",
        "costs.csv",
        "sample_trees.jsonl.gz",
    )
    with TemporaryDirectory(dir=output) as tmp:
        tmp = Path(tmp)
        with (
            gzip.open(tmp / files[0], "wt", newline="") as rf,
            gzip.open(tmp / files[1], "wt", newline="") as df,
            (tmp / files[2]).open("w", newline="") as cf,
            gzip.open(tmp / files[3], "wt") as sf,
            ProcessPoolExecutor(max_workers=workers) as pool,
        ):
            writers = [
                csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
                for f, columns in zip(
                    (rf, df, cf), (RESULTS, DIAGNOSTICS, COSTS), strict=True
                )
            ]
            for writer in writers:
                writer.writeheader()
            for i, (rows, diagnostics, costs, samples) in enumerate(
                pool.map(panel, jobs), 1
            ):
                for writer, block in zip(
                    writers, (rows, diagnostics, costs), strict=True
                ):
                    writer.writerows(block)
                for sample in samples:
                    sf.write(
                        json.dumps(sample, default=plain, separators=(",", ":")) + "\n"
                    )
                if i % 8 == 0:
                    print(f"Finished {i}/{len(jobs)} banks", flush=True)
        for name in files:
            os.replace(tmp / name, output / name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", "--out", type=Path, default=HERE)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    run(args.output, args.workers)

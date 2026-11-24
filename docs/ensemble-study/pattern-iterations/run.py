"""Compare two frozen native configurations on independent version-1 scenario banks."""

import argparse
import csv
import gzip
import json
import os
import runpy
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
from loguru import logger
from threadpoolctl import threadpool_limits

from fusionbench.eval.full_suite_point import fingerprint
from fusionbench.eval.transfer_grid import (
    development,
    iter_input_stresses,
    iter_test_conditions,
)
from fusionbench.methods.averaging import TopN
from fusionbench.methods.native_pattern import NativePatternOKAPI
from fusionbench.methods.stacking import StackingLogit, StackingLR
from fusionbench.transfer import load_preset

HERE = Path(__file__).resolve().parent
COMPONENTS = ("flat", "augmented", "gp")
ID = ("round", "scenario", "replicate", "data_seed", "gp_seed")
CONDITION = (*ID, "mode", "severity", "method")
RESULT = (*CONDITION, "brier")
DIAG = (
    *CONDITION,
    "chosen",
    *(f"val_{c}" for c in COMPONENTS),
    *(f"discovery_{c}" for c in COMPONENTS),
    *(f"refit_{c}" for c in COMPONENTS),
    "oracle_discovery_brier",
    "oracle_refit_brier",
    "chosen_discovery_regret",
    "chosen_refit_regret",
    "chosen_ties_discovery_oracle",
    "chosen_ties_refit_oracle",
    "coherent_providers",
)
COST = (
    *ID,
    "method",
    "chosen",
    "fit_seconds",
    "test_seconds",
    "development_rows",
    "discovery_rows",
    "validation_rows",
    "refit_rows",
    "validation_candidates",
    "total_score_calls",
    "gp_discovery_calls",
    "gp_unique_programs",
    "gp_path_digest",
    "selected_features",
    "pair_programs",
    "pair_ranking_calls",
    "pair_screen_calls",
    "selected_pair_count",
    "selected_pairs",
    "top_n_prefixes",
    "top_n_selected",
    "reference_reused_from_seed",
    "counts_json",
)
ROOT_SEEDS = {1: 2_010_000, 2: 2_110_000, 3: 2_210_000}
REPLICATES = {1: 6, 2: 6, 3: 10}
EXTRA_GP_OFFSET = 10_000_000


@cache
def common_functions():
    old = runpy.run_path(str(HERE.parent / "transfer-grid" / "run.py"))
    return old["full_brier"], old["serializer"](), old["plain"]


@cache
def coherent_pair():
    return runpy.run_path(str(HERE / "coherent.py"))["coherent_pair"]


@dataclass(frozen=True)
class Job:
    round_id: int
    scenario: str
    slot: int
    replicate: int
    incumbent_kwargs: dict
    challenger_kwargs: dict
    n_dev: int = 640
    n_test: int = 2048

    @property
    def data_seed(self):
        return ROOT_SEEDS[self.round_id] + 1000 * self.slot + self.replicate

    def identity(self, gp_seed):
        return {
            "round": self.round_id,
            "scenario": self.scenario,
            "replicate": self.replicate,
            "data_seed": self.data_seed,
            "gp_seed": gp_seed,
        }

    def search_seeds(self):
        if self.round_id == 3 and self.replicate < 2:
            return (self.data_seed, self.data_seed + EXTRA_GP_OFFSET)
        return (self.data_seed,)


@dataclass
class Fitted:
    name: str
    model: NativePatternOKAPI
    val: dict
    cost: dict
    sample: dict | None


def fit_native(job, name, kwargs, seed, x, y):
    from fusionbench.eval.suite_point_gp import predict_tree

    brier, serialize, plain = common_functions()
    c = x.shape[2]
    options = {"task": "binary" if c == 2 else "multiclass", "n_classes": c}
    start = perf_counter()
    model = NativePatternOKAPI(seed=seed, split_seed=job.data_seed, **kwargs).fit(
        x, y, **options
    )
    elapsed = perf_counter() - start
    stats = model.counts_
    validation = np.asarray(model.validation_indices_, dtype=np.int64)
    # Reconstruct the original ordered discovery/validation split; save both for replay.
    ordered = np.random.default_rng(
        np.random.SeedSequence([job.data_seed, 51])
    ).permutation(len(y))
    d, v = np.split(ordered, [stats["discovery_rows"]])
    if (
        not np.array_equal(v, validation)
        or stats["discovery_sha256"] != fingerprint(d)
        or stats["validation_sha256"] != fingerprint(v)
    ):
        raise AssertionError(
            "Archived discovery/validation split disagrees with the fitted learner"
        )
    if (
        len(v) != stats["validation_rows"]
        or set(model.trees_) != set(COMPONENTS)
        or set(model.discovery_trees_) != set(COMPONENTS)
    ):
        raise AssertionError("Incomplete native candidate or split ledger")
    val_predictions = {
        kind: predict_tree(model.discovery_trees_[kind], x[v]) for kind in COMPONENTS
    }
    target = np.eye(c)[y[v]]
    val = {kind: brier(p, target) for kind, p in val_predictions.items()}
    for kind in COMPONENTS:
        old = 2 * (1.0 - stats["validation_scores"][kind])
        np.testing.assert_allclose(val[kind], old, rtol=0, atol=1e-12)
    if model.chosen_ not in COMPONENTS:
        raise AssertionError(
            "Deployed choice must be frozen before evaluating test conditions"
        )
    pair = stats.get("pair_basis", {})
    cost = dict(
        **job.identity(seed),
        method=name,
        chosen=model.chosen_,
        fit_seconds=elapsed,
        test_seconds=0.0,
        development_rows=len(x),
        discovery_rows=len(d),
        validation_rows=len(v),
        refit_rows=stats["refit_rows"],
        validation_candidates=len(val),
        total_score_calls=stats["total_score_calls"],
        gp_discovery_calls=stats["score_calls"]["gp_discovery"],
        gp_unique_programs=stats["gp_unique_programs"],
        gp_path_digest=stats["gp_path_digest"],
        selected_features=len(stats["screen"]["selected"]),
        pair_programs=pair.get("programs", 0),
        pair_ranking_calls=pair.get("evaluations", 0),
        pair_screen_calls=pair.get("screen_calls", 0),
        selected_pair_count=pair.get("selected_pair_count", 0),
        selected_pairs=json.dumps(pair.get("selected_pairs", []), default=plain),
        counts_json=json.dumps(stats, default=plain),
    )
    sample = None
    if job.replicate == 0:
        sample = dict(
            **job.identity(seed),
            method=name,
            chosen=model.chosen_,
            kwargs=kwargs,
            selected_pairs=pair.get("selected_pairs", []),
            discovery_indices=d.tolist(),
            validation_indices=v.tolist(),
            val_brier=val,
            val_sha256={kind: fingerprint(p) for kind, p in val_predictions.items()},
            trees={kind: serialize(tree.root) for kind, tree in model.trees_.items()},
            discovery_trees={
                kind: serialize(tree.root)
                for kind, tree in model.discovery_trees_.items()
            },
        )
    return Fitted(name, model, val, cost, sample)


def fit_controls(x, y):
    options = {
        "task": "binary" if x.shape[-1] == 2 else "multiclass",
        "n_classes": x.shape[-1],
    }
    fitted = {}
    for label, model in (
        ("mean", None),
        ("top_n", TopN("brier")),
        ("prob_lr", StackingLR(c_value=1)),
        ("logit", StackingLogit(c_value=1)),
    ):
        elapsed = 0.0
        if model is not None:
            start = perf_counter()
            model.fit(x, y, **options)
            elapsed = perf_counter() - start
        fitted[label] = model, elapsed
    top = fitted["top_n"][0]
    if (
        len(top.prefix_scores_) != x.shape[1]
        or top.n_selected_ != np.argmin(top.prefix_scores_) + 1
    ):
        raise AssertionError(
            "The ranked-prefix baseline must score all available prefix lengths"
        )
    return fitted


def conditions(job, dev_x):
    # Request just the canonical endpoint. The source iterator also supplies the clean
    # bank, which both mild perturbations and the optional confirmation shift reuse.
    for mode, severity, x, y in iter_test_conditions(
        job.scenario,
        job.data_seed,
        job.n_test,
        modes=("canonical_shift",),
        severities=(1.0,),
    ):
        yield mode, severity, x, y, ()
        if mode == "clean":
            for kind, level, z in iter_input_stresses(
                x,
                job.data_seed,
                modes=("provider_swap", "log_jitter"),
                severities=(0.5,),
            ):
                yield kind, level, z, y, ()
            if job.round_id == 3:
                stressed, providers = coherent_pair()(dev_x, x, job.data_seed)
                yield "coherent_pair", 0.5, stressed, y, providers


def native_scores(fit, key, x, target, coherent):
    from fusionbench.eval.suite_point_gp import predict_tree

    brier = common_functions()[0]
    model = fit.model
    start = perf_counter()
    predictions = {kind: predict_tree(model.trees_[kind], x) for kind in COMPONENTS}
    deployed = model.predict(x)
    if not np.array_equal(predictions[model.chosen_], deployed):
        raise AssertionError("Deployed output differs from the frozen chosen program")
    refit = {kind: brier(p, target) for kind, p in predictions.items()}
    discovery = {}
    for kind in COMPONENTS:
        if model.discovery_trees_[kind] is model.trees_[kind]:
            discovery[kind] = refit[kind]
        else:
            discovery[kind] = brier(
                predict_tree(model.discovery_trees_[kind], x), target
            )
    best_d, best_r = min(discovery.values()), min(refit.values())
    selected_d, selected_r = discovery[model.chosen_], refit[model.chosen_]
    diag = dict(
        **key,
        method=fit.name,
        chosen=model.chosen_,
        **{f"val_{kind}": score for kind, score in fit.val.items()},
        **{f"discovery_{kind}": score for kind, score in discovery.items()},
        **{f"refit_{kind}": score for kind, score in refit.items()},
        oracle_discovery_brier=best_d,
        oracle_refit_brier=best_r,
        chosen_discovery_regret=selected_d - best_d,
        chosen_refit_regret=selected_r - best_r,
        chosen_ties_discovery_oracle=selected_d - best_d <= 1e-12,
        chosen_ties_refit_oracle=selected_r - best_r <= 1e-12,
        coherent_providers=",".join(str(i) for i in coherent),
    )
    fit.cost["test_seconds"] += perf_counter() - start
    return dict(**key, method=fit.name, brier=selected_r), diag


def panel(job):
    with threadpool_limits(1):
        return _panel(job)


def _panel(job):
    logger.disable("okapi")
    brier = common_functions()[0]
    x, y = development(job.scenario, job.data_seed, job.n_dev, 0.0)
    y = np.asarray(y).ravel()
    controls = fit_controls(x, y)
    ref_elapsed = dict.fromkeys(controls, 0.0)
    native = {}
    expected_partition = None
    for seed in job.search_seeds():
        a = fit_native(job, "native", job.incumbent_kwargs, seed, x, y)
        b = fit_native(job, "challenger", job.challenger_kwargs, seed, x, y)
        for fit in (a, b):
            partition = (
                fit.model.counts_["discovery_sha256"],
                fit.model.counts_["validation_sha256"],
            )
            if expected_partition is None:
                expected_partition = partition
            elif partition != expected_partition:
                raise AssertionError("Data split changed between policies or GP seeds")
        if job.round_id == 1 and a.cost["gp_path_digest"] != b.cost["gp_path_digest"]:
            raise AssertionError(
                "R1 pair screening changed the shared ordinary GP trajectory"
            )
        native[seed] = (a, b)
    # All fits and deployment choices exist before requesting the first test bank.
    rows, diag, costs, samples = [], [], [], []
    for mode, severity, test, labels, providers in conditions(job, x):
        target = np.eye(test.shape[2])[np.asarray(labels).ravel()]
        ref_scores = {}
        for name, (model, _) in controls.items():
            start = perf_counter()
            p = test.mean(axis=1) if model is None else model.predict(test)
            ref_scores[name] = brier(p, target)
            ref_elapsed[name] += perf_counter() - start
        for seed, models in native.items():
            key = dict(**job.identity(seed), mode=mode, severity=severity)
            rows.extend(
                dict(**key, method=name, brier=val) for name, val in ref_scores.items()
            )
            for fit in models:
                row, diagnostics = native_scores(fit, key, test, target, providers)
                rows.append(row)
                diag.append(diagnostics)
    for seed, models in native.items():
        for fit in models:
            costs.append(fit.cost)
            if fit.sample is not None:
                samples.append(fit.sample)
        for name, (model, duration) in controls.items():
            reuse = seed != job.data_seed
            cost = dict(
                **job.identity(seed),
                method=name,
                fit_seconds=0.0 if reuse else duration,
                test_seconds=0.0 if reuse else ref_elapsed[name],
                development_rows=len(y),
                reference_reused_from_seed=job.data_seed if reuse else "",
            )
            if name == "top_n":
                cost.update(
                    top_n_prefixes=len(model.prefix_scores_),
                    top_n_selected=int(model.n_selected_),
                )
            costs.append(cost)
    return rows, diag, costs, samples


def run(jobs, output, workers, settings):
    if not 1 <= workers <= 8:
        raise ValueError("Use one to eight CPU bank workers")
    output.mkdir(exist_ok=True, parents=True)
    files = (
        "results.csv.gz",
        "diagnostics.csv.gz",
        "costs.csv",
        "sample_trees.jsonl.gz",
        "settings.json",
    )
    prior = output / "settings.json"
    if prior.exists() and json.loads(prior.read_text()) != settings:
        raise ValueError(
            "Use a different output directory for a different frozen round or configuration"
        )
    plain = common_functions()[2]
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
                for f, columns in zip((rf, df, cf), (RESULT, DIAG, COST), strict=True)
            ]
            for writer in writers:
                writer.writeheader()
            for i, (rr, dd, cc, ss) in enumerate(pool.map(panel, jobs), 1):
                for writer, batch in zip(writers, (rr, dd, cc), strict=True):
                    writer.writerows(batch)
                for record in ss:
                    sf.write(
                        json.dumps(record, default=plain, separators=(",", ":")) + "\n"
                    )
                if i % 6 == 0:
                    print(f"Completed {i}/{len(jobs)} banks", flush=True)
        (tmp / files[4]).write_text(
            json.dumps(settings, indent=2, sort_keys=True) + "\n"
        )
        for name in files:
            os.replace(tmp / name, output / name)


def options(value):
    obj = json.loads(value)
    if not isinstance(obj, dict) or set(obj).intersection(("seed", "split_seed")):
        raise argparse.ArgumentTypeError(
            "Supply a JSON object of NativePatternOKAPI options, excluding seed and split_seed"
        )
    return obj


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--round", dest="round_id", required=True, type=int, choices=tuple(ROOT_SEEDS)
    )
    parser.add_argument("--incumbent-kwargs", type=options, default={})
    parser.add_argument("--challenger-kwargs", type=options)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    r = args.round_id
    if r == 1:
        if args.incumbent_kwargs or args.challenger_kwargs not in (
            None,
            {"pair_basis": True},
        ):
            parser.error("Round 1 compares the default fit with pair_basis=true")
        args.challenger_kwargs = {"pair_basis": True}
    if args.challenger_kwargs is None:
        parser.error(
            "Supply the prospectively selected configuration using --challenger-kwargs"
        )
    if r == 3 and args.incumbent_kwargs:
        parser.error("Round 3 uses the original native model as its comparator")
    versioned = load_preset("standard")["scenarios"]
    slots = {name: definition["seed_slot"] for name, definition in versioned.items()}
    if sorted(slots.values()) != list(range(19)):
        raise ValueError(
            "Expected the fixed 19 scenario slots in the version-1 standard preset"
        )
    jobs = [
        Job(r, name, slot, rep, args.incumbent_kwargs, args.challenger_kwargs)
        for name, slot in sorted(slots.items(), key=lambda item: item[1])
        for rep in range(REPLICATES[r])
    ]
    settings = {
        "round": r,
        "seed_root": ROOT_SEEDS[r],
        "scenarios": slots,
        "replicates": REPLICATES[r],
        "development_rows": 640,
        "test_rows": 2048,
        "source_label_noise": 0.0,
        "native_kwargs": args.incumbent_kwargs,
        "challenger_kwargs": args.challenger_kwargs,
        "primary_weights": {"clean": 0.5, "provider_swap": 0.25, "log_jitter": 0.25},
        "split_seed": "data_seed",
        "second_gp_seed_offset": EXTRA_GP_OFFSET if r == 3 else None,
        "second_gp_seed_replicates": [0, 1] if r == 3 else [],
        "coherent_pair": r == 3,
    }
    run(jobs, args.out or HERE / f"round-{r}", args.workers, settings)


if __name__ == "__main__":
    main()

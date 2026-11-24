"""Run or resume the frozen real and synthetic multiclass head comparison."""

import argparse
import csv
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from fusionbench import Dataset, get_scenario
from fusionbench.eval.transfer_grid import brier
from fusionbench.methods.averaging import SimpleAverage, TopN
from fusionbench.methods.native_pattern import NativePatternOKAPI
from fusionbench.methods.stacking import StackingLogit, StackingLR

STUDY = Path(__file__).resolve().parent
PREVIOUS = STUDY.parent / "external-coverage" / "external-results"
RECENT = STUDY.parent / "runner-up-transfer"
sys.path.insert(0, str(PREVIOUS.parent))
from symmetric import PooledScaleStackingLR, SymmetricStackingLR

REAL = {
    "digits_hetero_a": ("digits", 3140000, PREVIOUS, PREVIOUS),
    "digits_lr_bag": ("digits_lr_bag", 3140000, PREVIOUS, PREVIOUS),
    "wine": ("wine", 3150000, PREVIOUS, PREVIOUS),
    "digits_hetero_b": (
        "digits",
        3300000,
        RECENT / "external-fit",
        RECENT / "external-results",
    ),
    "iris": ("iris", 3310000, RECENT / "external-fit", RECENT / "external-results"),
    "seeds": ("seeds", 3320000, RECENT / "external-fit", RECENT / "external-results"),
}
SYNTHETIC = {
    "original": (0, 0),
    "shared_38": (1, 0),
    "mixed": (1, 0.5),
    "provider_specific": (1, 1),
}
ARCHIVED = ("mean", "top_n", "prob_lr", "logit", "native", "symmetric_lr", "pooled_lr")
METHODS = (
    *ARCHIVED,
    "stacking_shared_bias",
    "stacking_diagonal",
    "okapi_class_diagonal",
    "okapi_class_full",
)
BUDGET = 4096
INPUT_SHA256 = {
    PREVIOUS
    / "fit_banks.npz": "163b70904a03c314348de2652f81ac61909d62fa246c13ba760cd20df0fae886",
    PREVIOUS
    / "test_banks.npz": "835bce8241dd773683174e60d7882c4d7ebd94df04fa3a0757f788df793aac83",
    PREVIOUS
    / "scores.csv": "f2e080e7ae92b67729fe82a3a6df50febf95689bea4fbbb0e4f98aca08e16cc3",
    PREVIOUS
    / "fit_manifest.json": "bd273d64e36fabb14c0276cfccdaaef60180353015aa1cb8e3441ecfb2f701c5",
    RECENT
    / "external-fit/fit_banks.npz": "7caa774d7bb2022173838d12f9d45222fcbc9da196e511e6df4e185cb9134d3e",
    RECENT
    / "external-fit/fit_manifest.json": "c1d8343af41822188ccbb11c227a095e013a0278d2381bc4a029a87d332970a6",
    RECENT
    / "external-results/test_banks.npz": "c01b4cb376af61c740997c859afe556905d4e628ec733c6cdde1439c7783f97c",
    RECENT
    / "external-results/scores.csv": "83044da2f4a7b1302fb27a1ecac84c197fb05ffeb8b16bddd2c8ad9392db0d59",
}
COLUMNS = (
    "source",
    "cohort",
    "replicate",
    "seed",
    "method",
    "bank_sha256",
    "origin",
    "n_fit",
    "n_test",
    "brier",
    "accuracy",
    "fit_seconds",
    "affected_n",
    "affected_brier_sum",
    "affected_brier",
    "other_n",
    "other_brier_sum",
    "other_brier",
    "native_choice",
    "top_n",
    "evaluations",
    "native_metadata",
)


def sha(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def synthetic_dataset(cohort, seed):
    strength, deviation = SYNTHETIC[cohort]
    params = {"n_classes": 10, "separation": 6.0, "votes": [31, 7]}
    shared = np.zeros((10, 10))
    shared[[3, 8], [8, 3]] = 1
    individual = np.zeros((3, 10, 10))
    for j in range(3):
        individual[j, [3, 8], j] = 1
    base = get_scenario("multiclass_provider_panel", **params)
    scenario = (
        get_scenario(
            "shared_confusion_panel",
            **params,
            shared_matrix=shared,
            provider_matrices=individual,
            strength=strength,
            deviation_mix=deviation,
        )
        if strength
        else base
    )
    data = []
    for stream, size in enumerate((449, 2048)):
        p, y, _ = scenario.sample(size, np.random.default_rng([seed, stream]))
        p = p.astype(np.float32)
        if strength:
            original, labels, _ = base.sample(
                size, np.random.default_rng([seed, stream])
            )
            original = original.astype(np.float32)
            true_idx = y[:, None, None]
            np.testing.assert_array_equal(y, labels)
            np.testing.assert_array_equal(
                np.take_along_axis(p, true_idx, axis=2),
                np.take_along_axis(original, true_idx, axis=2),
            )
            np.testing.assert_array_equal(np.sort(p, axis=2), np.sort(original, axis=2))
            unchanged = ~np.isin(y, [3, 8])
            np.testing.assert_array_equal(p[unchanged], original[unchanged])
        data.extend((p, y))
    fit, y, test, t = data
    return Dataset(
        name=cohort,
        task="multiclass",
        n_classes=10,
        model_ids=scenario.model_ids(),
        fit_preds=fit,
        fit_y=y,
        test_preds=test,
        test_y=t,
    )


def load_bank(cohort, replicate):
    """Return an ordinary Dataset and its frozen fit/search seed."""
    if cohort in SYNTHETIC and 0 <= replicate < 6:
        seed = 3500000 + replicate
        return synthetic_dataset(cohort, seed), seed
    if cohort not in REAL or not 0 <= replicate < 8:
        raise ValueError(f"Unknown bank {cohort!r} / {replicate}")
    original, root, fit_dir, test_dir = REAL[cohort]
    seed = root + replicate
    key = original + "_" + str(seed)
    (record,) = [
        r
        for r in json.loads((fit_dir / "fit_manifest.json").read_text())["banks"]
        if r["suite"] == original and r["seed"] == seed
    ]

    def read(directory, split):
        with np.load(directory / (split + "_banks.npz"), allow_pickle=False) as bank:
            p, y, indices = (
                bank[key + "_" + part] for part in ("inputs", "labels", "rows")
            )
        if sha(indices) != record[split + "_rows_sha256"]:
            raise ValueError(f"Split row mismatch in {key}, {split}")
        return p, y

    x, y = read(fit_dir, "fit")
    z, t = read(test_dir, "test")
    if sha(x) != record["fit_inputs_sha256"] or sha(y) != record["fit_labels_sha256"]:
        raise ValueError(f"Frozen input or label mismatch in {key}")
    return Dataset(
        name=cohort,
        task="multiclass",
        n_classes=x.shape[2],
        model_ids=record["providers"],
        fit_preds=x,
        fit_y=y,
        test_preds=z,
        test_y=t,
    ), seed


def new_method(method, seed):
    factories = {
        "mean": SimpleAverage,
        "top_n": lambda: TopN("brier"),
        "logit": StackingLogit,
        "prob_lr": StackingLR,
        "native": lambda: NativePatternOKAPI(seed=seed, l2_strength=0.25),
        "symmetric_lr": SymmetricStackingLR,
        "pooled_lr": PooledScaleStackingLR,
    }
    if method in factories:
        return factories[method]()
    if method.startswith("stacking_"):
        from fusionbench.methods.class_stacking import (
            DiagonalStackingLR,
            SharedBiasStackingLR,
        )

        return {
            "stacking_shared_bias": SharedBiasStackingLR,
            "stacking_diagonal": DiagonalStackingLR,
        }[method](c_value=1.0)
    from fusionbench.methods.class_head import OKAPIClassDiagonal, OKAPIClassFull

    return {
        "okapi_class_diagonal": OKAPIClassDiagonal,
        "okapi_class_full": OKAPIClassFull,
    }[method](l2_strength=0.25, max_evaluations=BUDGET)


def native_meta_info(model):
    info = getattr(model, "fit_info_", None)
    if info is not None:
        return info.get("evaluations"), json.dumps(
            info, default=lambda a: np.asarray(a).tolist(), sort_keys=True
        )
    return None, None


def evaluate_method(method, ds, seed, synthetic):
    model = new_method(method, seed)
    tick = perf_counter()
    model.fit(ds.fit_preds, ds.fit_y, task=ds.task, n_classes=ds.n_classes)
    elapsed = perf_counter() - tick
    p = np.asarray(model.predict(ds.test_preds))
    y = ds.test_y
    score = brier(p, y)
    calls, meta = native_meta_info(model)
    row = {
        "origin": "fit",
        "fit_seconds": elapsed,
        "brier": score,
        "accuracy": float(np.mean(p.argmax(axis=1) == y)),
        "native_choice": getattr(model, "chosen_", None),
        "top_n": getattr(model, "n_selected_", None),
        "evaluations": calls,
        "native_metadata": meta,
    }
    if synthetic:
        errors = np.square(p.astype(float) - np.eye(ds.n_classes)[y]).sum(axis=1)
        affected = np.isin(y, [3, 8])
        for name, mask in (("affected", affected), ("other", ~affected)):
            loss = float(errors[mask].sum())
            row.update(
                {
                    name + "_n": int(mask.sum()),
                    name + "_brier_sum": loss,
                    name + "_brier": loss / mask.sum(),
                }
            )
        np.testing.assert_allclose(score, errors.mean(), rtol=0, atol=1e-14)
    return row


def run_bank(job):
    cohort, replicate, methods, previous_hash = job
    with threadpool_limits(limits=1):
        ds, seed = load_bank(cohort, replicate)
        bank_hash = hashlib.sha256(
            "".join(
                sha(a) for a in (ds.fit_preds, ds.fit_y, ds.test_preds, ds.test_y)
            ).encode()
        ).hexdigest()
        if previous_hash and previous_hash != {bank_hash}:
            raise ValueError(
                f"Resume inputs differ for {cohort}, replicate {replicate}"
            )
        syn = cohort in SYNTHETIC
        cached = {}
        if not syn:
            suite, _, _, directory = REAL[cohort]
            with (directory / "scores.csv").open() as handle:
                cached = {
                    r["method"]: r
                    for r in csv.DictReader(handle)
                    if r["suite"] == suite and int(r["seed"]) == seed
                }
        result = []
        for method in methods:
            if not syn and method in ARCHIVED:
                r = cached[method]
                stats = {
                    "origin": "archive",
                    "brier": r["brier"],
                    "accuracy": r["accuracy"],
                    "fit_seconds": r["fit_seconds"],
                    "native_choice": r["native_choice"],
                    "top_n": r["top_n"],
                }
            else:
                stats = evaluate_method(method, ds, seed, syn)
            result.append(
                {
                    "source": "synthetic" if syn else "real",
                    "cohort": cohort,
                    "replicate": replicate,
                    "seed": seed,
                    "n_fit": len(ds.fit_y),
                    "n_test": len(ds.test_y),
                    "bank_sha256": bank_hash,
                    "method": method,
                    **stats,
                }
            )
        return result


def write_scores(path, rows):
    temp = path.with_suffix(".tmp")
    with temp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows[key] for key in sorted(rows))
    os.replace(temp, path)


def summarize(out, frame):
    groups = frame.groupby(["source", "cohort", "method"])
    means = groups[
        [
            "brier",
            "accuracy",
            "fit_seconds",
            "evaluations",
            "affected_brier",
            "other_brier",
        ]
    ].mean()
    totals = groups[
        ["affected_n", "affected_brier_sum", "other_n", "other_brier_sum"]
    ].sum(min_count=1)
    means.join(totals).to_csv(out / "means.csv")
    comparisons = [
        ("stacking_shared_bias", "symmetric_lr"),
        ("stacking_diagonal", "stacking_shared_bias"),
        ("pooled_lr", "stacking_diagonal"),
        ("okapi_class_diagonal", "native"),
        ("okapi_class_full", "native"),
        ("okapi_class_full", "okapi_class_diagonal"),
    ]
    contrasts = []
    for (source, cohort), group in frame.groupby(["source", "cohort"]):
        metrics = (
            ("brier", "affected_brier", "other_brier")
            if source == "synthetic"
            else ("brier",)
        )
        for metric in metrics:
            values = group.pivot(index="replicate", columns="method", values=metric)
            for first, second in comparisons:
                if first not in values or second not in values:
                    continue
                delta = (values[first] - values[second]).dropna()
                if len(delta):
                    contrasts.append(
                        {
                            "source": source,
                            "cohort": cohort,
                            "endpoint": metric,
                            "left": first,
                            "right": second,
                            "replicates": len(delta),
                            "left_minus_right": delta.mean(),
                            "left_wins": (delta < -1e-12).sum(),
                            "ties": (delta.abs() <= 1e-12).sum(),
                            "left_losses": (delta > 1e-12).sum(),
                        }
                    )
    pd.DataFrame(contrasts).to_csv(out / "contrasts.csv", index=False)
    metadata = {
        "sources": {
            str(path.relative_to(STUDY.parent)): value
            for path, value in INPUT_SHA256.items()
        },
        "real_reference_method_ids": list(ARCHIVED),
        "real_seeds": {key: spec[1] for key, spec in REAL.items()},
        "real_replicates": 8,
        "synthetic_seed_root": 3500000,
        "synthetic_replicates": 6,
        "fit_stream": 0,
        "test_stream": 1,
        "synthetic_rows": {"fit": 449, "test": 2048},
        "synthetic_affected_labels": [3, 8],
        "brier": "mean row sum of squared class errors; lower is better",
        "pairing": "All methods on identical stored provider tensors. Synthetic variants use common random numbers and unchanged true-label probability and class-probability multiset per provider per row.",
        "fit_budget": {
            "native_class_head_l2_strength": 0.25,
            "native_class_head_max_evaluations": BUDGET,
            "logistic_C": 1.0,
        },
        "tie_tolerance": 1e-12,
        "n_scores": len(frame),
        "n_scores_by_origin": frame.groupby("origin").size().to_dict(),
    }
    (out / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=STUDY / "results")
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--source", choices=("real", "synthetic", "all"), default="all")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.source != "synthetic":
        for file, expected in INPUT_SHA256.items():
            if hashlib.sha256(file.read_bytes()).hexdigest() != expected:
                raise ValueError(
                    f"Input checksum differs from the frozen source: {file}"
                )
    dest = args.out / "scores.csv"
    rows = {}
    if dest.exists():
        with dest.open() as f:
            for r in csv.DictReader(f):
                rows[r["cohort"], int(r["replicate"]), r["method"]] = r
    cohorts = {
        **(REAL if args.source != "synthetic" else {}),
        **(SYNTHETIC if args.source != "real" else {}),
    }
    jobs = []
    for cohort in cohorts:
        for rep in range(6 if cohort in SYNTHETIC else 8):
            required = [
                m for m in dict.fromkeys(args.methods) if (cohort, rep, m) not in rows
            ]
            previous_hash = {
                r["bank_sha256"]
                for (c, i, _), r in rows.items()
                if (c, i) == (cohort, rep)
            }
            if required:
                jobs.append((cohort, rep, required, previous_hash))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_bank, j) for j in jobs]
        for job in as_completed(futures):
            for row in job.result():
                rows[row["cohort"], row["replicate"], row["method"]] = row
            write_scores(dest, rows)
    if rows:
        summarize(args.out, pd.read_csv(dest, float_precision="round_trip"))
    print(f"{len(rows)} unique scores, {len(jobs)} bank jobs", flush=True)


if __name__ == "__main__":
    main()

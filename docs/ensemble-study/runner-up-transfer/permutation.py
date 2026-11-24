"""Replay two historical panels with a common per-row change of class coordinates."""

import argparse
import hashlib
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t
from threadpoolctl import threadpool_limits

from fusionbench import Dataset, get_scenario, load_dataset

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "external-coverage"
sys.path.insert(0, str(SOURCE))

from common import evaluate_methods
from export_bank import export_bank
from synthetic import SETTINGS, STRATA

COMPARISONS = {
    "native_minus_prob_lr": ("native", "prob_lr"),
    "pooled_lr_minus_symmetric_lr": ("pooled_lr", "symmetric_lr"),
}
KEY = ["cohort", "replicate", "seed"]


def digest(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def relabel(x, y, seed, split_id):
    """Return scores and labels in new coordinates, and an audit of the mapping."""
    n, _, classes = x.shape
    rng = np.random.default_rng(np.random.SeedSequence([seed, 330, split_id]))
    order = rng.permuted(
        np.broadcast_to(np.arange(classes, dtype=np.int64), (n, classes)), axis=1
    )
    transformed = np.take_along_axis(x, order[:, None, :], axis=2)
    labels = (order == y[:, None]).argmax(axis=1).astype(y.dtype)
    rows = np.arange(n)
    np.testing.assert_array_equal(order[rows, labels], y)
    np.testing.assert_array_equal(transformed[rows, :, labels], x[rows, :, y])
    np.testing.assert_array_equal(np.sort(transformed, axis=2), np.sort(x, axis=2))

    mean_original = x.mean(axis=1, dtype=np.float64)
    mean_changed = transformed.mean(axis=1, dtype=np.float64)
    np.testing.assert_array_equal(
        mean_changed, np.take_along_axis(mean_original, order, axis=1)
    )
    eye = np.eye(classes)
    original_brier = np.square(mean_original - eye[y]).sum(axis=1)
    changed_brier = np.square(mean_changed - eye[labels]).sum(axis=1)
    error = float(np.max(np.abs(original_brier - changed_brier)))
    if error > 1e-12:
        raise AssertionError(f"Mean-pool Brier changed on a row: {error}")
    checks = {
        "rows": n,
        "providers": x.shape[1],
        "classes": classes,
        "changed_labels": int(np.count_nonzero(labels != y)),
        "row_mean_brier_max_abs_change": error,
        "raw_x_sha256": digest(x),
        "raw_y_sha256": digest(y),
        "permuted_x_sha256": digest(transformed),
        "permuted_y_sha256": digest(labels),
        "order_sha256": digest(order),
    }
    return transformed, labels, checks


def historical_bank(cohort, seed):
    if cohort == "digits":
        with tempfile.TemporaryDirectory() as directory:
            return load_dataset(export_bank(cohort, seed, directory))
    classes, n_fit, _ = STRATA["digits"]
    name, kwargs = SETTINGS[cohort]
    scenario = get_scenario(name, n_classes=classes, **kwargs)
    fit, test = [
        scenario.sample(n, np.random.default_rng([seed, split_id]))
        for split_id, n in enumerate((n_fit, 2048))
    ]
    return Dataset(
        name=cohort,
        task="multiclass",
        n_classes=classes,
        model_ids=scenario.model_ids(),
        fit_preds=fit[0].astype(np.float32),
        fit_y=fit[1],
        test_preds=test[0].astype(np.float32),
        test_y=test[1],
    )


def run_bank(job):
    from loguru import logger

    logger.disable("okapi")
    cohort, _replicate, seed = job
    key = dict(zip(KEY, job))
    original = historical_bank(cohort, seed)
    checks, arrays = [], {}
    for split_id, split in enumerate(("fit", "test")):
        x = getattr(original, f"{split}_preds")
        y = getattr(original, f"{split}_y")
        changed_x, changed_y, audit = relabel(x, y, seed, split_id)
        arrays[f"{split}_preds"], arrays[f"{split}_y"] = changed_x, changed_y
        checks.append({**key, "split": split, **audit})
    permuted = replace(original, **arrays)
    scores = []
    with threadpool_limits(limits=1):
        for condition, bank in (("raw", original), ("permuted", permuted)):
            scores.extend(
                {**key, "condition": condition, **r}
                for r in evaluate_methods(bank, seed=seed)
            )
    return scores, checks


def match_archive(methods):
    digits = pd.read_csv(
        SOURCE / "external-results/scores.csv", float_precision="round_trip"
    ).query("suite == 'digits'")
    panel = pd.read_csv(
        SOURCE / "synthetic-results/scores.csv", float_precision="round_trip"
    ).query("stratum == 'digits' and config == 'panel_E'")
    archived = pd.concat(
        [digits.assign(cohort="digits"), panel.assign(cohort="panel_E")]
    )
    keys = ["cohort", "seed", "method"]
    replay = methods.query("condition == 'raw'").merge(
        archived[keys + ["brier"]].rename(columns={"brier": "published_brier"}),
        on=keys,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not replay["_merge"].eq("both").all() or len(replay) != 16 * 9:
        raise AssertionError("Partial historical score replay")
    replay["raw_minus_published"] = replay.brier - replay.published_brier
    np.testing.assert_allclose(replay.brier, replay.published_brier, rtol=0, atol=1e-8)
    return replay[keys + ["published_brier", "raw_minus_published"]]


def summarize(methods):
    losses = methods.pivot(
        index=KEY + ["method"], columns="condition", values="brier"
    ).reset_index()
    losses["change"] = losses.permuted - losses.raw
    for control in ("mean", "symmetric_lr"):
        np.testing.assert_allclose(
            losses.loc[losses.method.eq(control), "change"], 0.0, rtol=0, atol=1e-7
        )
    means = losses.groupby(["cohort", "method"], as_index=False)[
        ["raw", "permuted", "change"]
    ].mean()
    paired = []
    for key, bank in losses.groupby(KEY):
        bank = bank.set_index("method")
        for comparison, (left, right) in COMPARISONS.items():
            raw, permuted, delta = (
                bank.loc[left, ["raw", "permuted", "change"]]
                - bank.loc[right, ["raw", "permuted", "change"]]
            )
            paired.append(
                {
                    **dict(zip(KEY, key)),
                    "comparison": comparison,
                    "raw_gap": raw,
                    "permuted_gap": permuted,
                    "change": delta,
                }
            )
    pairs = pd.DataFrame(paired)
    contrasts = []
    for (cohort, comparison), group in pairs.groupby(["cohort", "comparison"]):
        for quantity in ("raw_gap", "permuted_gap", "change"):
            values = group[quantity]
            center = values.mean()
            width = t.ppf(0.975, len(values) - 1) * values.sem()
            contrasts.append(
                {
                    "cohort": cohort,
                    "comparison": comparison,
                    "quantity": quantity,
                    "n": len(values),
                    "mean": center,
                    "low": center - width,
                    "high": center + width,
                }
            )
    return means, pairs, pd.DataFrame(contrasts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--out", type=Path, default=HERE / "permutation")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    jobs = [
        (cohort, rep, base + rep)
        for cohort, base in (("digits", 3_140_000), ("panel_E", STRATA["digits"][2]))
        for rep in range(8)
    ]
    scores, checks = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for bank_scores, bank_checks in pool.map(run_bank, jobs):
            scores.extend(bank_scores)
            checks.extend(bank_checks)
    methods = pd.DataFrame(scores)
    if len(methods) != 288 or not np.isfinite(methods.brier).all():
        raise AssertionError("Incomplete method results")
    archive = match_archive(methods)
    means, pairs, contrasts = summarize(methods)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, frame in (
        ("methods", methods),
        ("checks", pd.DataFrame(checks)),
        ("archive", archive),
        ("means", means),
        ("pairs", pairs),
        ("contrasts", contrasts),
    ):
        frame.to_csv(args.out / f"{name}.csv", index=False, lineterminator="\n")


if __name__ == "__main__":
    main()

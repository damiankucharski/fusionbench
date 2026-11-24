"""Replay only the historical class-permutation banks and independently check the reported contrasts."""

import csv
import hashlib
import json
from math import sqrt
from pathlib import Path
from statistics import fmean, stdev

import numpy as np

from fusionbench import get_scenario

HERE = Path(__file__).resolve().parent
HISTORY = HERE.parents[1] / "external-coverage"
METHODS = {
    "mean",
    "top_n",
    "prob_lr",
    "logit",
    "native",
    "pooled_lr",
    "symmetric_lr",
    "temperature",
    "sharpener",
}
CONTRASTS = {
    "native_minus_prob_lr": ("native", "prob_lr"),
    "pooled_lr_minus_symmetric_lr": ("pooled_lr", "symmetric_lr"),
}


def read(path):
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def keyed(rows, fields):
    records = {tuple(row[f] for f in fields): row for row in rows}
    assert len(records) == len(rows), "duplicate archive keys"
    return records


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def historical_splits():
    historical = HISTORY / "external-results"
    with (
        np.load(historical / "fit_banks.npz") as fit,
        np.load(historical / "test_banks.npz") as test,
    ):
        for i in range(8):
            seed = 3_140_000 + i
            key = f"digits_{seed}_"
            assert not np.intersect1d(fit[key + "rows"], test[key + "rows"]).size
            for split, bank in enumerate((fit, test)):
                yield (
                    "digits",
                    i,
                    seed,
                    split,
                    bank[key + "inputs"],
                    bank[key + "labels"],
                )
    law = get_scenario(
        "multiclass_provider_panel",
        n_classes=10,
        separation=6.0,
        votes=[31, 7],
        confusion_bias=3.0,
    )
    for i in range(8):
        seed = 3_160_000 + i
        for split, n in enumerate((449, 2048)):
            x, y, _ = law.sample(n, np.random.default_rng([seed, split]))
            yield "panel_E", i, seed, split, x.astype(np.float32), y


def mean_brier(x, labels):
    means = np.maximum(x.mean(axis=1).astype(np.float64), 1e-12)
    means /= means.sum(axis=1, keepdims=True)
    residuals = means.copy()
    residuals[np.arange(len(x)), labels] -= 1
    return means, np.square(residuals).sum(axis=1)


def transform_check(x, y, seed, split, row):
    n, providers, classes = x.shape
    rows = np.arange(n)
    rng = np.random.default_rng([seed, 330, split])
    order = np.array([rng.permutation(classes) for _ in rows], dtype=np.int64)
    inverse = np.argsort(order, axis=1)
    yy = inverse[rows, y].astype(y.dtype)
    xx = np.empty_like(x)
    for old_class in range(classes):
        xx[rows, :, inverse[:, old_class]] = x[:, :, old_class]
    np.testing.assert_array_equal(order[rows, yy], y)
    np.testing.assert_array_equal(
        np.take_along_axis(xx, inverse[:, None, :], axis=2), x
    )
    np.testing.assert_array_equal(np.sort(x, axis=2), np.sort(xx, axis=2))
    np.testing.assert_array_equal(x[rows, :, y], xx[rows, :, yy])
    for name, value in (
        ("raw_x", x),
        ("raw_y", y),
        ("permuted_x", xx),
        ("permuted_y", yy),
        ("order", order),
    ):
        assert row[name + "_sha256"] == sha(value), (seed, split, name)

    means, loss = mean_brier(x, y)
    alt_means, alt_loss = mean_brier(xx, yy)
    mean_error = np.max(np.abs(alt_means[rows[:, None], inverse] - means))
    brier_error = np.max(np.abs(alt_loss - loss))
    assert mean_error < 2e-15 and brier_error < 3e-15
    new_top_old = order[rows[:, None], xx.argmax(axis=2)]
    switches = x.argmax(axis=2) != new_top_old
    ties = (x == x.max(axis=2, keepdims=True)).sum(axis=2) > 1
    assert not np.any(switches & ~ties)
    return (
        n,
        n * providers,
        int(switches.sum()),
        float(mean_error),
        float(brier_error),
        loss.mean(),
        alt_loss.mean(),
    )


def verify():
    methods = keyed(
        read(HERE / "methods.csv"), ("cohort", "seed", "condition", "method")
    )
    pairs = keyed(read(HERE / "pairs.csv"), ("cohort", "seed", "comparison"))
    contrasts = keyed(
        read(HERE / "contrasts.csv"), ("cohort", "comparison", "quantity")
    )
    checks = keyed(read(HERE / "checks.csv"), ("cohort", "seed", "split"))
    archive = keyed(read(HERE / "archive.csv"), ("cohort", "seed", "method"))
    means = keyed(read(HERE / "means.csv"), ("cohort", "method"))
    assert (len(methods), len(pairs), len(contrasts), len(checks)) == (288, 32, 12, 32)
    assert len(archive) == 144 and len(means) == 18
    sources = [
        ("digits", r)
        for r in read(HISTORY / "external-results/scores.csv")
        if r["suite"] == "digits"
    ]
    sources += [
        ("panel_E", r)
        for r in read(HISTORY / "synthetic-results/scores.csv")
        if r["stratum"] == "digits" and r["config"] == "panel_E"
    ]
    assert len(sources) == 144
    archive_error = 0.0
    for cohort, source in sources:
        raw = methods[cohort, source["seed"], "raw", source["method"]]
        saved = archive[cohort, source["seed"], source["method"]]
        published = float(source["brier"])
        difference = float(raw["brier"]) - published
        archive_error = max(archive_error, abs(difference))
        assert abs(float(saved["published_brier"]) - published) < 2e-15
        assert abs(float(saved["raw_minus_published"]) - difference) < 2e-15
        assert all(
            abs(float(raw[k]) - float(source[k])) < 1e-12 for k in ("brier", "accuracy")
        )

    rows = provider_rows = switches = 0
    max_mean_error = max_brier_error = max_symmetric_error = 0.0
    for cohort, replicate, seed, split, x, y in historical_splits():
        record = checks[cohort, str(seed), ("fit", "test")[split]]
        assert int(record["replicate"]) == replicate
        n, nx, ties, mean_error, brier_error, raw_brier, perm_brier = transform_check(
            x, y, seed, split, record
        )
        rows += n
        provider_rows += nx
        switches += ties
        max_mean_error, max_brier_error = (
            max(max_mean_error, mean_error),
            max(max_brier_error, brier_error),
        )
        if split == 1:
            for condition, brier in [("raw", raw_brier), ("permuted", perm_brier)]:
                assert (
                    abs(
                        float(methods[cohort, str(seed), condition, "mean"]["brier"])
                        - brier
                    )
                    < 1e-12
                )
                for method in METHODS:
                    assert (
                        int(methods[cohort, str(seed), condition, method]["replicate"])
                        == replicate
                    )
            symmetric = [
                float(methods[cohort, str(seed), c, "symmetric_lr"]["brier"])
                for c in ("raw", "permuted")
            ]
            max_symmetric_error = max(
                max_symmetric_error, abs(symmetric[0] - symmetric[1])
            )
    assert max_symmetric_error < 1e-7

    max_pair = max_summary = 0.0
    for cohort, first in [("digits", 3_140_000), ("panel_E", 3_160_000)]:
        for method in METHODS:
            entry = means[cohort, method]
            scores = {}
            for condition in ("raw", "permuted"):
                scores[condition] = [
                    float(methods[cohort, str(first + i), condition, method]["brier"])
                    for i in range(8)
                ]
            scores["change"] = [
                p - r for p, r in zip(scores["permuted"], scores["raw"], strict=True)
            ]
            for quantity, values in scores.items():
                assert abs(float(entry[quantity]) - fmean(values)) < 2e-15
        for comparison, (a, b) in CONTRASTS.items():
            values = {"raw_gap": [], "permuted_gap": [], "change": []}
            for i in range(8):
                key = (cohort, str(first + i))
                row = pairs[*key, comparison]
                assert int(row["replicate"]) == i
                for c in ("raw", "permuted"):
                    values[c + "_gap"].append(
                        float(methods[*key, c, a]["brier"])
                        - float(methods[*key, c, b]["brier"])
                    )
                values["change"].append(
                    values["permuted_gap"][-1] - values["raw_gap"][-1]
                )
                for name, series in values.items():
                    max_pair = max(max_pair, abs(series[-1] - float(row[name])))
            for name, series in values.items():
                row = contrasts[cohort, comparison, name]
                assert int(row["n"]) == 8
                avg = fmean(series)
                half = (
                    2.3646242515927844 * stdev(series) / sqrt(8)
                )  # two-sided .95, 7 df
                max_summary = max(
                    max_summary,
                    *(
                        abs(v - float(row[k]))
                        for k, v in [
                            ("mean", avg),
                            ("low", avg - half),
                            ("high", avg + half),
                        ]
                    ),
                )
    assert max_pair < 3e-15 and max_summary < 1e-12
    return {
        "splits": len(checks),
        "rows": rows,
        "provider_rows": provider_rows,
        "tie_changed_top1": switches,
        "historical_method_scores": len(sources),
        "historical_max_brier_change": archive_error,
        "matched_cohort_method_summaries": 3 * len(means),
        "array_hashes": 5 * len(checks),
        "individual_paired_gaps": len(pairs) * 3,
        "interval_summaries": len(contrasts),
        "max_mean_probability_error": max_mean_error,
        "max_row_brier_error": max_brier_error,
        "max_symmetric_brier_change": max_symmetric_error,
        "max_paired_gap_error": max_pair,
        "max_summary_error": max_summary,
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))

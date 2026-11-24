"""Paired capacity contrasts and the matched synthetic treatment effect."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

from run import METHODS, REAL, SYNTHETIC

PAIRS = [
    ("stacking_shared_bias", "symmetric_lr"),
    ("stacking_diagonal", "stacking_shared_bias"),
    ("pooled_lr", "stacking_diagonal"),
    ("okapi_class_full", "okapi_class_diagonal"),
    ("okapi_class_diagonal", "native"),
    ("okapi_class_full", "native"),
    ("okapi_class_full", "pooled_lr"),
]


def summary(delta, synthetic):
    n = len(delta)
    center = delta.mean()
    width = t.ppf(0.975, n - 1) * delta.std(ddof=1) / np.sqrt(n)
    return dict(
        n=n,
        delta=center,
        first_lower=int((delta < -1e-12).sum()),
        tied=int((delta.abs() <= 1e-12).sum()),
        minimum=delta.min(),
        maximum=delta.max(),
        ci_low=center - width if synthetic else np.nan,
        ci_high=center + width if synthetic else np.nan,
    )


def analyze(folder):
    scores = pd.read_csv(folder / "scores.csv")
    means, contrasts, changes = [], [], []
    for endpoint in ["brier", "affected_brier", "other_brier"]:
        eligible = (
            scores if endpoint == "brier" else scores[scores.cohort.isin(SYNTHETIC)]
        )
        wide = eligible.pivot(
            index=["cohort", "replicate"], columns="method", values=endpoint
        )
        if set(wide.columns) != set(METHODS) or wide.isna().any().any():
            raise ValueError(f"Incomplete score panel for {endpoint}")
        for cohort, group in wide.groupby("cohort"):
            group = group.droplevel("cohort")
            synthetic = cohort in SYNTHETIC
            assert set(group.index) == set(range(6 if synthetic else 8))
            means.append(dict(cohort=cohort, endpoint=endpoint, **group.mean()))
            for first, second in PAIRS:
                contrasts.append(
                    dict(
                        cohort=cohort,
                        endpoint=endpoint,
                        first=first,
                        second=second,
                        **summary(group[first] - group[second], synthetic),
                    )
                )
            if synthetic and cohort != "original":
                control = wide.xs("original")
                for method in METHODS:
                    changes.append(
                        dict(
                            cohort=cohort,
                            endpoint=endpoint,
                            method=method,
                            **summary(group[method] - control[method], True),
                        )
                    )
    assert set(scores.cohort) == set(REAL) | set(SYNTHETIC)
    pd.DataFrame(means).to_csv(folder / "panel-means.csv", index=False)
    pd.DataFrame(contrasts).to_csv(folder / "paired-capacity.csv", index=False)
    pd.DataFrame(changes).to_csv(folder / "synthetic-changes.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", type=Path, default=Path(__file__).with_name("results")
    )
    analyze(parser.parse_args().results)

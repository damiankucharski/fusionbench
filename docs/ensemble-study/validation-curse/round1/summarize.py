"""Produce paired, bank-level intervals for menu selection and fitted comparators."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

HERE = Path(__file__).resolve().parent
METRICS = (
    "validation",
    "audit",
    "iid",
    "shift",
    "source_risk",
    "optimism",
    "extra_optimism",
    "exact_optimism",
    "exact_extra_optimism",
    "source_selection_regret",
    "source_menu_penalty",
    "source_best_hit",
    "shift_selection_regret",
    "shift_menu_penalty",
)
SPLITS = ("fit", "validation", "audit", "iid", "shift")


def interval(x):
    x = np.asarray(x, dtype=float)
    mean = x.mean()
    h = t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return {"banks": len(x), "mean": mean, "low": mean - h, "high": mean + h}


def summarize(out):
    raw = pd.read_csv(out / "selections.csv")
    assert set(raw.law) == {"mean_null", "information"}
    assert (
        len(raw) == 2 * 32 * 12
        and not raw.duplicated(["law", "seed", "validation_rows", "menu_size"]).any()
    )
    rows = []
    paired = []
    for (law, n, k), cell in raw.groupby(["law", "validation_rows", "menu_size"]):
        assert cell.seed.nunique() == 32
        key = {"law": law, "validation_rows": n, "menu_size": k}
        for metric in METRICS:
            rows.append({**key, "metric": metric, **interval(cell[metric])})
    for (law, n), frame in raw.groupby(["law", "validation_rows"]):
        for metric in METRICS:
            pivot = frame.pivot(
                index="seed", columns="menu_size", values=metric
            ).astype(float)
            for k in (4, 16, 64):
                paired.append(
                    {
                        "law": law,
                        "validation_rows": n,
                        "menu_size": k,
                        "versus_menu": 1,
                        "metric": metric,
                        **interval(pivot[k] - pivot[1]),
                    }
                )
    for law, frame in raw[raw.menu_size == 64].groupby("law"):
        for metric in METRICS:
            pivot = frame.pivot(
                index="seed", columns="validation_rows", values=metric
            ).astype(float)
            for n in (128, 512):
                paired.append(
                    {
                        "law": law,
                        "validation_rows": n,
                        "menu_size": 64,
                        "versus_menu": 64,
                        "versus_validation_rows": 32,
                        "metric": metric,
                        **interval(pivot[n] - pivot[32]),
                    }
                )
    pd.DataFrame(rows).to_csv(out / "selection-means.csv", index=False)
    pd.DataFrame(paired).to_csv(out / "selection-contrasts.csv", index=False)
    controls = pd.read_csv(out / "references.csv")
    assert not controls.duplicated(["law", "seed", "method"]).any()
    ref_rows = []
    for cohort, subset in [
        ("common_8", controls[controls.replicate < 8]),
        ("all_32_cheap", controls[controls.method != "native"]),
    ]:
        for (law, method), frame in subset.groupby(["law", "method"]):
            for split in SPLITS:
                base = subset[
                    (subset.law == law) & (subset.method == "mean")
                ].set_index("seed")[split]
                scores = frame.set_index("seed")[split]
                diff = scores - base
                assert diff.notna().all() and len(diff) == (
                    8 if cohort == "common_8" else 32
                )
                ref_rows.append(
                    {
                        "cohort": cohort,
                        "law": law,
                        "method": method,
                        "split": split,
                        "mean_brier": scores.mean(),
                        "contrast": "method minus mean",
                        **interval(diff),
                    }
                )
    pd.DataFrame(ref_rows).to_csv(out / "reference-summary.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "results")
    summarize(parser.parse_args().out)

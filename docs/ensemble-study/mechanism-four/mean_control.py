"""Score the registered ancillary calibrated-mean control on the same confirmation draws."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from banks_study import banks, score
from calibrate_gp import fit_calibrated
from run_four import digest, fit_model, forecast
from stats_four import summarize
from threadpoolctl import threadpool_limits

from fusionbench.methods.averaging import SimpleAverage

OUT = Path(__file__).resolve().parent / "results"
KEY = ["cohort", "family", "seed"]
NAME = "calibration-control"


def replay():
    rows = []
    with threadpool_limits(limits=1):
        for b in banks(4):
            base = fit_model(SimpleAverage(), b.x, b.y)
            model = fit_calibrated(base, b.x, b.y, groups=b.groups)
            f, p = forecast(model, b.x), forecast(model, b.audit_x)
            f0, p0 = forecast(base, b.x), forecast(base, b.audit_x)
            rows.append(
                {
                    **b.context,
                    "round": 4,
                    "method": "cal_mean",
                    **score(b, p),
                    "fit_brier": np.square(f - np.eye(f.shape[1])[b.y]).sum(1).mean(),
                    "providers": len(model.best_tree.unique_value_node_ids),
                    "nodes": model.best_tree.nodes_count,
                    "fit_hash": digest(f),
                    "test_hash": digest(p),
                    "beta": model.calibration_["beta"],
                    "mean_fit_hash": digest(f0),
                    "mean_test_hash": digest(p0),
                    "changed_decisions": int(
                        np.count_nonzero(p.argmax(1) != p0.argmax(1))
                    ),
                }
            )
    assert len(rows) == 408
    pd.DataFrame(rows).to_csv(OUT / f"{NAME}-scores.csv", index=False)


def compare():
    primary = pd.read_csv(OUT / "r4-scores.csv", float_precision="round_trip")
    control = pd.read_csv(OUT / f"{NAME}-scores.csv", float_precision="round_trip")
    indexed = control.set_index(KEY, verify_integrity=True).sort_index()
    means = (
        primary.query("method=='mean'")
        .set_index(KEY, verify_integrity=True)
        .sort_index()
    )
    assert len(indexed) == 408 and indexed.index.equals(means.index)
    for k in ("fit_hash", "test_hash"):
        assert np.array_equal(indexed["mean_" + k], means[k])
        identity = indexed.beta == 1
        assert np.array_equal(indexed.loc[identity, k], means.loc[identity, k])
    assert np.array_equal(indexed.providers, means.providers)
    levels, contrasts, _ = summarize(
        pd.concat((primary, control), ignore_index=True), 4
    )
    levels[levels.method == "cal_mean"].to_csv(OUT / f"{NAME}-levels.csv", index=False)
    keep = (contrasts.method == "cal_mean") & contrasts.reference.isin(
        ["mean", "cal_pull", "lr"]
    )
    contrasts[keep].to_csv(OUT / f"{NAME}-contrasts.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", nargs="?", choices=("replay", "compare", "both"), default="both"
    )
    mode = parser.parse_args().mode
    if mode in ("replay", "both"):
        replay()
    if mode in ("compare", "both"):
        compare()

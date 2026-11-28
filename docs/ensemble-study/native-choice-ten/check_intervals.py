"""Check round summaries using direct bootstrap indexing, independent of the count matrix."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
EXCLUDED = {
    "grouped_rare_event",
    "shared_error_bloc",
    "rare_scene_specialist",
    "cloned_shortcut",
    "parity",
    "similar_classes_38",
}


def check(r):
    stem = f"r{r:02d}"
    raw = pd.read_csv(OUT / f"{stem}.csv", float_precision="round_trip")
    published = pd.read_csv(OUT / f"{stem}-contrasts.csv", float_precision="round_trip")
    raw = raw[raw.variant.eq("primary")]
    standard = raw[raw.cohort.eq("standard")]
    groups = {
        "standard": standard,
        "seen24": standard[~standard.family.isin(EXCLUDED)],
        "held6": standard[standard.family.isin(EXCLUDED)],
        **dict(tuple(raw[raw.cohort.eq("control")].groupby("family"))),
    }
    checked = 0
    for group, frame in groups.items():
        wide = frame.pivot(
            index=["family", "seed"], columns="method", values="brier"
        ).sort_index()
        n_families = frame.family.nunique()
        n = len(wide) // n_families
        draw = np.random.default_rng(81000 + r).integers(
            n, size=(10_000, n_families, n)
        )
        draw += (n * np.arange(n_families))[None, :, None]
        for rule in wide:
            if rule == "native":
                continue
            loss = (wide[rule] - wide["native"]).to_numpy()
            estimate = np.mean(loss[draw], axis=(1, 2))
            limits = np.quantile(estimate, [0.025, 0.975])
            row = published[
                (published.group == group)
                & (published.method == rule)
                & (published.reference == "native")
            ]
            np.testing.assert_allclose(
                row[["delta", "lo", "hi"]], [[loss.mean(), *limits]], atol=2e-13, rtol=0
            )
            checked += 1
    return {
        "round": r,
        "fixed_groups": len(groups),
        "independent_paired_intervals": checked,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("round", type=int, choices=range(1, 11), nargs="+")
    rounds = parser.parse_args().round
    for r in rounds:
        validation = check(r)
        (OUT / f"r{r:02d}-interval-check.json").write_text(
            json.dumps(validation, indent=2) + "\n"
        )

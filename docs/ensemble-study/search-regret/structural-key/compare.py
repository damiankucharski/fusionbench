"""Independently validate the two-checkout replay against the observed run."""

import argparse
import csv
import hashlib
from pathlib import Path


def records(file):
    with file.open() as stream:
        return list(csv.DictReader(stream))


def key(row):
    return int(row["data_seed"]), int(row["gp_seed"]), int(row["calls"]), row["policy"]


def compare(native, observed):
    native_rows = records(native)
    fingerprints = {key(row): row for row in native_rows}
    assert len(native_rows) == len(fingerprints) == 192
    fit_champions = [row for row in records(observed) if row["champion"] == "fit"]
    assert len(fit_champions) == 192
    largest_fit_delta = 0.0
    for row in fit_champions:
        policy = "structural" if row["policy"] == "fixed" else "legacy"
        panel = int(row["data_seed"]), int(row["gp_seed"]), int(row["budget"]), policy
        check = fingerprints.pop(panel)
        single_tree = hashlib.sha256((row["tree"] + "\n").encode("utf-8"))
        assert row["path_sha256"] == check["sequence_sha256"], (panel, "search path")
        assert single_tree.hexdigest() == check["fit_champion_sha256"], (
            panel,
            "fit champion",
        )
        difference = abs(float(check["fit"]) - float(row["fit"]))
        assert difference < 2e-15, (panel, "Brier", difference)
        largest_fit_delta = max(largest_fit_delta, difference)
    assert not fingerprints
    print(
        f"{len(fit_champions)} matches for both path and fit champion. Largest Brier discrepancy {largest_fit_delta:.3g}."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("native", type=Path)
    parser.add_argument("observed", type=Path)
    args = parser.parse_args()
    compare(args.native, args.observed)

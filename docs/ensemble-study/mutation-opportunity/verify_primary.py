"""Independently calculate paired Brier estimates directly from the four raw files."""

import argparse
import csv
import gzip
import itertools
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from scipy.stats import t


def rows(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        yield from csv.DictReader(handle)


def verify(folder):
    cells = defaultdict(dict)
    sizes = {}
    for src in (
        "runs.csv.gz",
        "refresh-independent-runs.csv.gz",
        "wine-runs.csv.gz",
        "wine-refresh-independent-runs.csv.gz",
    ):
        for r in rows(folder / src):
            if r["checkpoint"] == "fixed" or r["split"] != "iid":
                continue
            cell = (
                r["scenario"],
                r["checkpoint"],
                r["selection"],
                int(r["data_seed"]),
                r["method"],
            )
            seed = int(float(r["gp_seed"]))
            assert seed not in cells[cell]
            cells[cell][seed] = float(r["brier"])
            sizes[(r["scenario"], int(r["data_seed"]))] = int(r["rows"])
    families = sorted({k[0] for k in cells} - {"wine"})
    pairs = (
        ("retype", "identity"),
        ("retype", "refresh"),
        ("refresh", "identity"),
        ("identity", "stock"),
        ("retype", "stock"),
        ("refresh_independent", "refresh"),
        ("refresh_independent", "identity"),
        ("retype", "refresh_independent"),
    )
    assert len(families) == 19 and all(set(v) == {661, 673} for v in cells.values())

    def sample(family, checkpoint, selection, bank, a, b):
        parts = [
            cells[(family, checkpoint, selection, bank, method)] for method in (a, b)
        ]
        return statistics.mean(parts[0][s] - parts[1][s] for s in (661, 673))

    def ci(d):
        m = statistics.mean(d)
        margin = t.ppf(0.975, len(d) - 1) * statistics.stdev(d) / math.sqrt(len(d))
        return m, m - margin, m + margin

    def match(x, y):
        assert math.isclose(x, float(y), rel_tol=1e-12, abs_tol=2e-14), (x, y)

    lookup = {
        (r["split"], r["checkpoint"], r["selection"], r["a"], r["b"]): r
        for r in rows(folder / "suite-contrasts.csv")
        if r["metric"] == "brier"
    }
    per_family = {
        (r["scenario"], r["split"], r["checkpoint"], r["selection"], r["a"], r["b"]): r
        for r in rows(folder / "paired.csv.gz")
        if r["metric"] == "brier"
    }
    external = {
        (r["checkpoint"], r["selection"], r["a"], r["b"]): r
        for r in rows(folder / "wine-contrasts.csv")
    }
    counts = defaultdict(int)
    for cp, selection, (a, b) in itertools.product(
        ("calls", "generation"), ("fit", "validation"), pairs
    ):
        by_bank = defaultdict(list)
        for f in families:
            d = []
            for bank in range(5100, 5106):
                effect = sample(f, cp, selection, bank, a, b)
                d.append(effect)
                by_bank[bank].append(effect)
            stored = per_family[(f, "iid", cp, selection, a, b)]
            for measured, key in zip(ci(d), ("mean", "low_95", "high_95"), strict=True):
                match(measured, stored[key])
            counts["scenario_intervals"] += 1
        whole = [statistics.mean(x) for x in by_bank.values()]
        stored = lookup[("iid", cp, selection, a, b)]
        for measured, key in zip(ci(whole), ("mean", "low_95", "high_95"), strict=True):
            match(measured, stored[key])
        counts["fixed_suite_intervals"] += 1
        wine = external[(cp, selection, a, b)]
        pool = 0.0
        for fold in range(5):
            d = sample("wine", cp, selection, fold, a, b)
            match(d, wine[f"fold{fold}"])
            pool += d * sizes["wine", fold]
        match(pool / 178, wine["pooled_difference"])
        counts["external_pooled_differences"] += 1
    (folder / "agreement.json").write_text(json.dumps(counts, indent=2) + "\n")
    print(json.dumps(counts))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    verify(parser.parse_args().directory)

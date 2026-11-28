"""Freeze a screening winner, then evaluate prespecified limits on new synthetic draws."""

import argparse
import json
import subprocess
from hashlib import sha256
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
LOCK = HERE / "selection.json"
CANDIDATES = ((2, "sparse_gp"), (3, "cal_pull"), (3, "cal_sparse"))
LIMITS = {"standard": 0.005, "scene16": 0.005, "strong": 0.010}


def comparisons(r, policy):
    tab = pd.read_csv(OUT / f"r{r}-contrasts.csv", float_precision="round_trip")
    return tab[(tab.method == policy) & (tab.reference == "pull_mean")].set_index(
        ["group", "metric"]
    )


def pick():
    rows = []
    for r, policy in CANDIDATES:
        c = comparisons(r, policy)
        breadth = {k: float(c.at[(k, "brier"), "delta"]) for k in LIMITS}
        rows.append(
            {
                "method": policy,
                "round": r,
                "breadth": breadth,
                "factorial_expected_delta": float(
                    c.at[("factorial", "expected_brier"), "delta"]
                ),
                "max_violation": max(
                    [0, *(breadth[k] / limit - 1 for k, limit in LIMITS.items())]
                ),
            }
        )
    best = min(
        rows, key=lambda row: (row["max_violation"], row["factorial_expected_delta"])
    )
    files = [
        OUT / f"r{r}-{kind}.csv" for r in (1, 2, 3) for kind in ("scores", "contrasts")
    ]
    return {
        "chosen": best["method"],
        "candidates": rows,
        "sha256": {file.name: sha256(file.read_bytes()).hexdigest() for file in files},
    }


def locked():
    root = Path(
        subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], cwd=HERE, text=True
        ).strip()
    )
    stored = subprocess.check_output(
        ["git", "show", f"HEAD:{LOCK.relative_to(root)}"], cwd=root
    )
    if stored != LOCK.read_bytes() or json.loads(stored) != pick():
        raise RuntimeError("The committed policy and its screening evidence must match")
    return json.loads(stored)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("select", "confirm"))
    if parser.parse_args().action == "select":
        result = pick()
        path = LOCK
        if path.exists() and json.loads(path.read_text()) != result:
            raise RuntimeError("A different policy is already locked")
    else:
        chosen = locked()["chosen"]
        c = comparisons(4, chosen)
        primary = bool(c.at[("factorial", "expected_brier"), "hi"] < 0)
        checks = {
            k: bool(c.at[(k, "brier"), "hi"] <= limit) for k, limit in LIMITS.items()
        }
        result = {
            "chosen": chosen,
            "primary_pass": primary,
            "breadth_checks": checks,
            "broad_pass": primary and all(checks.values()),
        }
        path = OUT / "confirmation.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(result["chosen"])


if __name__ == "__main__":
    main()

"""Lock one policy after four screening rounds and test the prespecified comparisons."""

import argparse
import json
import subprocess
from hashlib import sha256
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
LOCK = HERE / "selection.json"
LIMITS = {"standard": 0.005, "strong": 0.010, "digits_449": 0.004}
POLICIES = {2: "seeded_mean", 3: "pull_mean", 4: "paired_se"}


def contrasts(r, method):
    rows = pd.read_csv(OUT / f"round{r}-contrasts.csv", float_precision="round_trip")
    return rows.query(
        "metric == 'brier' and method == @method and reference == 'full'"
    ).set_index("group")


def select():
    checksums = {}
    for r in range(1, 5):
        for table in ("scores", "contrasts"):
            file = OUT / f"round{r}-{table}.csv"
            checksums[file.name] = sha256(file.read_bytes()).hexdigest()
    choices = []
    for r, method in POLICIES.items():
        delta = contrasts(r, method).delta
        violation = max(
            [0, *(float(delta[g]) / limit - 1 for g, limit in LIMITS.items())]
        )
        choices.append(
            {
                "round": r,
                "method": method,
                "deltas": delta.to_dict(),
                "violation": violation,
                "target_average": float((delta.weak + delta.scene16) / 2),
            }
        )
    best = min(choices, key=lambda row: (row["violation"], row["target_average"]))
    return {"chosen": best["method"], "candidates": choices, "sha256": checksums}


def read_locked():
    raw = LOCK.read_bytes()
    root = Path(
        subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], cwd=HERE, text=True
        ).strip()
    )
    rel = LOCK.relative_to(root)
    if subprocess.check_output(["git", "show", f"HEAD:{rel}"], cwd=root) != raw:
        raise RuntimeError("The committed policy differs from the working record")
    locked = json.loads(raw)
    if select() != locked:
        raise RuntimeError("Screening data differ from the locked selection")
    return locked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("select", "confirm"))
    action = parser.parse_args().action
    if action == "select":
        record = select()
        if LOCK.exists() and json.loads(LOCK.read_text()) != record:
            raise RuntimeError("Selection already exists and differs from this screen")
        dest = LOCK
    else:
        chosen = read_locked()["chosen"]
        data = contrasts(5, chosen)
        checks = {g: bool(data.at[g, "hi"] < 0) for g in ("weak", "scene16")}
        checks.update(
            {
                g: bool(data.at[g, "delta" if g == "digits_449" else "hi"] <= cap)
                for g, cap in LIMITS.items()
            }
        )
        record = {
            "chosen": chosen,
            "criteria": checks,
            "both_targets": checks["weak"] and checks["scene16"],
            "all_criteria": all(checks.values()),
        }
        dest = OUT / "confirmation.json"
    dest.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(dest, record["chosen"])


if __name__ == "__main__":
    main()

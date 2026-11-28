"""Apply the preregistered screening and, separately, held-out confirmation rules."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
ENTRIES = [
    (1, "gp_top8"),
    (2, "simplex"),
    (3, "bootstrap_medoid"),
    (4, "cf_gain"),
    (4, "cf_margin"),
]
LIMITS = {"standard": 0.005, "digits_449": 0.004, "strong": 0.010}


def screen():
    rows = []
    hashes = {}
    for r, method in ENTRIES:
        path = OUT / f"round{r}-contrasts.csv"
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        tab = pd.read_csv(path)
        delta = (
            tab[(tab.method == method) & (tab.reference == "regularized")]
            .set_index("group")
            .delta
        )
        assert {"weak", *LIMITS} <= set(delta.index)
        violation = max(
            [0.0, *((delta[k] - bound) / bound for k, bound in LIMITS.items())]
        )
        rows.append(
            {
                "round": r,
                "candidate": method,
                **delta.to_dict(),
                "passes_screen": bool(violation == 0),
                "largest_scaled_violation": float(violation),
            }
        )
    winner = min(rows, key=lambda row: (row["largest_scaled_violation"], row["weak"]))
    return {
        "round": winner["round"],
        "candidate": winner["candidate"],
        "passed_screen": winner["passes_screen"],
        "limits": LIMITS,
        "screen": rows,
        "screen_table_sha256": hashes,
    }


def confirmation(pick):
    tab = pd.read_csv(OUT / "round5-contrasts.csv")
    data = tab[(tab.method == pick) & (tab.reference == "regularized")].set_index(
        "group"
    )
    checks = {
        "weak_upper_below_zero": bool(data.loc["weak", "hi"] < 0),
        "standard_upper_within_005": bool(
            data.loc["standard", "hi"] <= LIMITS["standard"]
        ),
        "strong_upper_within_010": bool(data.loc["strong", "hi"] <= LIMITS["strong"]),
        "digits449_mean_within_004": bool(
            data.loc["digits_449", "delta"] <= LIMITS["digits_449"]
        ),
    }
    per_family = (
        pd.read_csv(OUT / "round5-families.csv")
        .query("method == @pick")
        .set_index("family")
        .delta_reg
    )
    return {
        "candidate": pick,
        "primary_confirmed": checks["weak_upper_below_zero"],
        "broad_criteria": all(checks.values()),
        "checks": checks,
        "improved_families": int((per_family < -1e-12).sum()),
        "tied_families": int((np.abs(per_family) <= 1e-12).sum()),
        "worsened_families": int((per_family > 1e-12).sum()),
        "largest_gains": per_family.nsmallest(3).to_dict(),
        "largest_regressions": per_family.nlargest(3).to_dict(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("select", "confirm"))
    action = parser.parse_args().action
    locked = HERE / "selection.json"
    result = screen()
    if action == "select":
        if locked.exists() and result != json.loads(locked.read_text()):
            raise ValueError("Screen differs from existing selection record")
        locked.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(result["candidate"], result["round"])
    else:
        assert result == json.loads(locked.read_text()), "Selection changed since lock"
        report = confirmation(result["candidate"])
        (OUT / "confirmation.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))

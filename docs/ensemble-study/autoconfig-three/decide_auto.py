"""Lock one configuration rule on development families and confirm on fresh source banks."""

import argparse
import json
import subprocess
from hashlib import sha256
from pathlib import Path

import pandas as pd

from summarize_auto import HELDOUT, OUT

HERE = Path(__file__).resolve().parent
LOCK = HERE / "selection.json"
CANDIDATES = ("tree_rule", "cv_rule")


def select():
    # Derive the decision from bank-level records, before any R3 scoring.
    eval_set = pd.read_csv(OUT / "r2-scores.csv", float_precision="round_trip")
    seen = eval_set[(eval_set.cohort == "standard") & ~eval_set.family.isin(HELDOUT)]
    means = seen.groupby(["family", "method"]).brier.mean().unstack()
    if len(means) != 24 or means[list(CANDIDATES)].isna().any().any():
        raise ValueError("The registered 24-family screen is incomplete")
    risks = {c: float(means[c].mean()) for c in CANDIDATES}
    chosen = min(CANDIDATES, key=risks.get)
    files = [OUT / "r1-policy.json", OUT / "r2-scores.csv", OUT / "r2-levels.csv"]
    return {
        "chosen": chosen,
        "trained24_risks": risks,
        "sha256": {p.name: sha256(p.read_bytes()).hexdigest() for p in files},
    }


def locked():
    root = Path(
        subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], cwd=HERE, text=True
        ).strip()
    )
    raw = subprocess.check_output(
        ["git", "show", f"HEAD:{LOCK.relative_to(root)}"], cwd=root
    )
    record = json.loads(raw)
    if raw != LOCK.read_bytes() or record != select():
        raise RuntimeError("The evidence differs from the committed rule selection")
    return record


def confirm():
    chosen = locked()["chosen"]
    frame = pd.read_csv(OUT / "r3-contrasts.csv", float_precision="round_trip")
    c = frame[
        (frame.method == chosen)
        & (frame.reference == "constant")
        & (frame.metric == "brier")
    ].set_index("group")
    levels = pd.read_csv(OUT / "r3-levels.csv", float_precision="round_trip")
    wide = levels.pivot(index="group", columns="method", values="brier")
    oracle = {}
    for group in ("standard", "heldout6", "scene16", "factorial"):
        available = wide.at[group, "constant"] - wide.at[group, "portfolio_oracle"]
        gain = wide.at[group, "constant"] - wide.at[group, chosen]
        oracle[group] = {
            "available_vs_constant": float(available),
            "gain_vs_constant": float(gain),
            "fraction_captured": float(gain / available) if available > 1e-12 else None,
        }
    primary = bool(c.at["standard", "hi"] < 0)
    holdout = bool(c.at["heldout6", "hi"] < 0)
    scarce = bool(c.at["scene16", "hi"] <= 0.005)
    return {
        "chosen": chosen,
        "standard_pass": primary,
        "heldout_pass": holdout,
        "scene16_within_limit": scarce,
        "generalization_pass": primary and holdout and scarce,
        "test_oracle_diagnostic": oracle,
        "changes_vs_constant": {
            g: {k: float(c.at[g, k]) for k in ("delta", "lo", "hi")}
            for g in ("standard", "trained24", "heldout6", "scene16")
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("lock", "confirm"))
    if parser.parse_args().action == "lock":
        data, path = select(), LOCK
        if path.exists() and json.loads(path.read_text()) != data:
            raise RuntimeError("A different rule was already selected")
    else:
        data, path = confirm(), OUT / "confirmation.json"
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    print(data["chosen"])


if __name__ == "__main__":
    main()

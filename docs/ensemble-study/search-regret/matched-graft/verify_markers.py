"""Compare a full native replay with lineage observation enabled and disabled."""

import argparse
import json
from pathlib import Path

import pandas as pd


def verify(marked, unmarked):
    a = pd.read_csv(marked / "runs.csv.gz", float_precision="round_trip")
    b = pd.read_csv(unmarked / "runs.csv.gz", float_precision="round_trip")
    excluded = {"proposal_seconds", "search_seconds", "winner_occurrences"}
    invariant = [c for c in a if c not in excluded and not c.startswith("lineage_")]
    pd.testing.assert_frame_equal(a[invariant], b[invariant], check_exact=True)
    assert (b.winner_occurrences.dropna() == "[]").all()
    return {
        "identical_rows": len(a),
        "identical_columns": len(invariant),
        "distinct_restarts": len(
            a[["data_seed", "fit_regime", "gp_seed"]].dropna().drop_duplicates()
        ),
        "compared_columns": invariant,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("marked", type=Path)
    parser.add_argument("unmarked", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.marked, args.unmarked), indent=2))

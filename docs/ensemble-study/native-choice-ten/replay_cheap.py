"""Independently reconstruct the three cheap comparators on fixed representative banks."""

import argparse
import json
from hashlib import sha256
from pathlib import Path
from runpy import run_path

import numpy as np
from threadpoolctl import threadpool_limits

from banks_ten import banks
from summarize import CHEAP, OUT, table, validate_rows

HERE = Path(__file__).resolve().parent
AUDIT = run_path(str(HERE.parent / "mechanism-four" / "verify_four.py"))
BASE = AUDIT["OLD"]
FAMILIES = {
    "correlated",
    "gated_experts",
    "multiclass_committee",
    "grouped_rare_event",
    "weak",
    "scene16",
    "shortcut_000",
}


def replay(r):
    raw = table(r)
    checks = validate_rows(raw, r)
    panels = raw.set_index(["variant", "family", "seed"])
    checked = []
    identities = raw.groupby(["variant", "family", "seed"])[
        ["source_x_hash", "source_y_hash", "test_x_hash", "target_hash"]
    ].first()
    seen = set()
    first = raw.groupby(["variant", "family"]).seed.min().to_dict()
    for b in banks(r):
        key = (b.variant, b.family, b.seed)
        if key in seen:
            raise ValueError("Duplicate reconstructed bank")
        seen.add(key)
        target = b.audit_y if b.audit_y is not None else np.column_stack((b.mass, b.q))
        expect = [
            sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
            for a in (b.x, b.y, b.audit_x, target)
        ]
        if identities.loc[key].tolist() != expect:
            raise ValueError(f"Regenerated source/target data differ for {key}")
        if b.family not in FAMILIES or b.seed != first[(b.variant, b.family)]:
            continue
        rows = panels.loc[[(b.variant, b.family, b.seed)]].set_index("method")
        fit, audit = BASE["controls"](b.x, b.y, (b.x, b.audit_x))
        for method in CHEAP:
            # Derive mean, ranked prefix and sklearn logistic fits independently of
            # deployment code, then reconstruct empirical or exact support scores.
            metrics = AUDIT["checked_measures"](b, audit[method])
            want = [BASE["empirical"](fit[method], b.y), metrics[0], metrics[2]]
            np.testing.assert_allclose(
                rows.loc[method, ["fit_brier", "brier", "macro_f1"]].astype(float),
                want,
                atol=4e-12,
                rtol=0,
            )
        checked.append([b.variant, b.family, int(b.seed)])
    if seen != set(identities.index):
        raise ValueError("Reconstructed bank set differs from score table")
    if {x[1] for x in checked} != set(raw.family) & FAMILIES:
        raise ValueError("A representative comparison was missed")
    return {
        **checks,
        "regenerated_bank_identities": len(seen),
        "independent_reference_fits": 3 * len(checked),
        "replayed_banks": checked,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("round", type=int, choices=range(1, 11))
    r = parser.parse_args().round
    with threadpool_limits(limits=1):
        check = replay(r)
    (OUT / f"r{r:02d}-cheap-replay.json").write_text(json.dumps(check, indent=2) + "\n")

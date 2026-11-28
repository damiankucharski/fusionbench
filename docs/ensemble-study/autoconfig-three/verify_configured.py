"""Check deployed full-data predictions against frozen round-two benchmark rows."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from banks_auto import banks
from threadpoolctl import threadpool_limits

from fusionbench.methods.auto_configuration import ConfiguredOKAPI

HERE = Path(__file__).resolve().parent
KEYS = {
    ("factorial", "h0c0s0", 612000),
    ("standard", "similar_classes_38", 612400),
    ("standard", "correlated", 612400),
}


def _read_scores():
    with (HERE / "results" / "r2-scores.csv").open(newline="") as stream:
        return {
            (r["cohort"], r["family"], int(r["seed"]), r["method"]): r
            for r in csv.DictReader(stream)
            if r["method"] in {"tree_rule", "cv_rule"}
        }


def _digest(matrix):
    return hashlib.sha256(
        np.ascontiguousarray(matrix, dtype=np.float64).tobytes()
    ).hexdigest()


def verify():
    expected = _read_scores()
    policy = json.loads((HERE / "results" / "r1-policy.json").read_text())["policy"]
    checks = []
    with threadpool_limits(limits=1):
        for bank in banks(2):
            key = tuple(bank.context[k] for k in ("cohort", "family", "seed"))
            if key not in KEYS:
                continue
            classes = bank.x.shape[-1]
            for method, strategy in (("tree_rule", policy), ("cv_rule", "source_cv")):
                ref = expected[(*key, method)]
                adapter = ConfiguredOKAPI(strategy).fit(
                    bank.x,
                    bank.y,
                    task="binary" if classes == 2 else "multiclass",
                    n_classes=classes,
                    groups=bank.groups,
                )
                fitted = _digest(adapter.predict(bank.x))
                deployed = _digest(adapter.predict(bank.audit_x))
                if (adapter.action_, fitted, deployed) != (
                    ref["action"],
                    ref["fit_hash"],
                    ref["test_hash"],
                ):
                    raise RuntimeError(
                        f"Action or prediction hash mismatch on {key}, {method}"
                    )
                checks.append(
                    {
                        "cohort": key[0],
                        "family": key[1],
                        "seed": key[2],
                        "classes": classes,
                        "groups": None
                        if bank.groups is None
                        else len(np.unique(bank.groups)),
                        "method": method,
                        "action": adapter.action_,
                        "fit_sha256": fitted,
                        "audit_sha256": deployed,
                    }
                )
            if len(checks) == 2 * len(KEYS):
                break
    if len(checks) != 2 * len(KEYS):
        raise RuntimeError("Some expected verification banks were absent")
    return checks


if __name__ == "__main__":
    checks = verify()
    path = HERE / "results" / "r2-public-replay.json"
    path.write_text(json.dumps(checks, indent=2) + "\n")
    print(f"{len(checks)} configured fits reproduced source and audit hashes")

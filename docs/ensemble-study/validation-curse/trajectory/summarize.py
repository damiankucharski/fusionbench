"""Summarize complete native traces without rerunning or changing evolution."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fusionbench.eval.population_trajectory import draw

OUT = Path(__file__).parent
METRICS = (
    "median_test",
    "champion_test",
    "audit_pick_test",
    "test_oracle_test",
    "behaviors",
    "genotypes",
    "median_nodes",
)
TOL = 1e-12


def compare(condition, label, after, before):
    after = after.set_index("bank").sort_index()
    before = before.set_index("bank").sort_index()
    assert after.index.equals(before.index)
    rows = []
    for metric in METRICS:
        delta = after[metric] - before[metric]
        rows.append(
            {
                "condition": condition,
                "contrast": label,
                "metric": metric,
                "n": len(delta),
                "median_delta": delta.median(),
                "lower": (delta < -TOL).sum(),
                "higher": (delta > TOL).sum(),
                "tie": (delta.abs() <= TOL).sum(),
            }
        )
    return rows


def main():
    stages = pd.read_csv(OUT / "stages.csv")
    deltas = []
    for condition, g in stages.groupby("condition", sort=False):

        def select(policy, pool, generation, g=g):
            return g[
                (g.objective == policy)
                & (g.pool == pool)
                & (g.generation == generation)
            ]

        for policy in ("brier", "paired_se"):
            final = select(policy, "active", 6)
            deltas.extend(
                compare(
                    condition,
                    f"{policy}_final_minus_initial",
                    final,
                    select(policy, "active", 0),
                )
            )
            deltas.extend(
                compare(
                    condition,
                    f"{policy}_final_active_minus_archive",
                    final,
                    select(policy, "archive", 6),
                )
            )
        for pool, generation in (("active", 6), ("archive", 6), ("archive240", -1)):
            deltas.extend(
                compare(
                    condition,
                    f"paired_se_minus_brier_{pool}",
                    select("paired_se", pool, generation),
                    select("brier", pool, generation),
                )
            )
    pd.DataFrame(deltas).to_csv(OUT / "paired.csv", index=False)
    stages.groupby(["condition", "objective", "pool", "generation"])[
        ["calls", *METRICS]
    ].median().to_csv(OUT / "medians.csv")

    retention = []
    split_medians = []
    with gzip.open(OUT / "traces.jsonl.gz", "rt") as f:
        for line in f:
            run = json.loads(line)
            c, b = run["condition"], run["bank"]
            scenario, (_, (x, _, latent), _) = draw(c, b)
            posterior = scenario.bayes_oracle(x, latent)[:, 1]
            signature = hashlib.sha256(
                np.round(posterior, 10).astype("<f8").tobytes()
            ).hexdigest()
            matching = {
                i for i, r in enumerate(run["programs"]) if r["behavior"] == signature
            }
            archive = set()
            for g, snap in enumerate(run["generations"]):
                archive.update(snap["active"])
                archive.update(snap["offered"])
                for pool_name, ids in (
                    ("active", snap["active"]),
                    ("archive", archive),
                ):
                    split_medians.append(
                        {
                            "condition": c,
                            "bank": b,
                            "objective": run["objective"],
                            "generation": g,
                            "pool": pool_name,
                            **{
                                split: np.median(
                                    [run["programs"][i][split] for i in ids]
                                )
                                for split in ("support", "audit", "test")
                            },
                        }
                    )
                retention.append(
                    {
                        "condition": c,
                        "bank": b,
                        "objective": run["objective"],
                        "generation": g,
                        "active_match": bool(matching & set(snap["active"])),
                        "archive_match": bool(matching & archive),
                    }
                )
    pd.DataFrame(split_medians).to_csv(OUT / "split-medians.csv", index=False)
    pd.DataFrame(retention).to_csv(OUT / "posterior-retention.csv", index=False)


if __name__ == "__main__":
    main()

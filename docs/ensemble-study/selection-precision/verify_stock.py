"""Independently replay the historical stock route at current data and GP seeds."""

import argparse
import json
from itertools import product
from pathlib import Path

import pandas as pd
from loguru import logger

from fusionbench.eval.full_suite_point import data_for
from fusionbench.eval.suite_point_gp import fit_suite_point
from fusionbench.scenarios.registry import SCENARIOS


def collect(path):
    logger.disable("okapi")
    written = (
        {tuple(json.loads(line)["job"]) for line in path.read_text().splitlines()}
        if path.exists()
        else set()
    )
    with path.open("a") as f:
        for name, bank in product(SCENARIOS, range(5900, 5906)):
            _, data = data_for(name, bank)
            for seed in (709, 719):
                job = (name, bank, seed)
                if job in written:
                    continue
                cp = fit_suite_point(
                    *data["fit"][:2],
                    *data["validation"][:2],
                    seed=seed,
                    point_mutation=False,
                )
                stats = {
                    calls: {
                        "path_digest": v.path_digest,
                        "generation": v.generation,
                        "archive_trees": v.n_trees,
                        "raw_fit": v.fit_brier,
                        "raw_validation": v.validation_brier,
                    }
                    for calls, v in cp.items()
                }
                f.write(json.dumps({"job": job, "points": stats}) + "\n")
                f.flush()


def verify(path, folder):
    df = pd.read_csv(folder / "runs.csv.gz", float_precision="round_trip")
    sample = df[df.method.eq("round3") & df.selection.eq("fit") & df.split.eq("fit")]
    q = sample.set_index(
        ["scenario", "data_seed", "gp_seed", "calls"], verify_integrity=True
    )
    count = 0
    for line in path.read_text().splitlines():
        r = json.loads(line)
        for n, v in r["points"].items():
            row = q.loc[(*r["job"], int(n))]
            for field in ("generation", "archive_trees", "path_digest"):
                assert row[field] == v[field], (r["job"], field)
            assert abs(row.brier - 2 * v["raw_fit"]) < 4e-13
            count += 1
    assert len(q) == count == 456
    out = {
        "stock_trajectories": count // 2,
        "ordered_fitness_calls": 228 * 1200,
        "checkpoints": count,
    }
    (folder / "stock-verified.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("journal", type=Path)
    p.add_argument("--compare", type=Path)
    opts = p.parse_args()
    if opts.compare:
        verify(opts.journal, opts.compare)
    else:
        collect(opts.journal)

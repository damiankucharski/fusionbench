"""Replay eight preregistered finite-panel paths without the donor/population observer."""

import argparse
import gzip
import json
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from fusionbench.eval.precision_lattice import (
    draw_bank,
    exact_family,
    integer_losses,
    native_search,
)
from fusionbench.eval.search_graded import SCORES, counts

PIN = "3eae0e5480cba0471cc5fda71b0c90f35b6a73f2"


def check(value, message):
    if not value:
        raise ValueError(message)


def nodes(tree):
    return 1 + sum(nodes(c) for c in tree["children"])


def verify(filename):
    import okapi
    from loguru import logger

    logger.disable("okapi")
    sdk = Path(okapi.__file__).resolve().parent
    check(
        subprocess.check_output(
            ["git", "-C", sdk, "rev-parse", "HEAD"], text=True
        ).strip()
        == PIN,
        "Unexpected native checkout",
    )
    seen = set()
    all_rules = total_calls = 0
    with gzip.open(filename, "rt") as stream, TemporaryDirectory() as temp:
        directory = Path(temp)
        inputs = [directory / f"p{i}.npy" for i in range(4)]
        for i, path in enumerate(inputs):
            np.save(path, SCORES[:, i : i + 1])
        gt = directory / "labels.npy"
        for line in stream:
            t = json.loads(line)
            if t["data_seed"] not in (6600, 6615):
                continue
            key = t["law"], t["data_seed"], t["gp_seed"]
            check(key not in seen, f"Repeated trace {key}")
            seen.add(key)
            bank = draw_bank(*key[:2])
            observed = counts(bank.fit_cells, bank.fit_labels)
            units = integer_losses(exact_family()[0], observed)
            np.save(
                gt,
                observed.pos[:, None]
                / np.maximum((observed.pos + observed.neg)[:, None], 1),
            )
            # This pre-existing sampler calls native run_iteration and observes its
            # actual fitness callback; it has no crossover/mutation/survivor hooks.
            checkpoints, _ = native_search(
                inputs, gt, units, 256, key[2], 3, False, (400,)
            )
            archive, digest, _, _ = checkpoints[400]
            expected = {}
            for program in t["archive"]:
                rule = t["rule_ids"][program]
                expected.setdefault(rule, program)
            check(
                digest == t["raw_digest"],
                f"Different tree order, rule or raw training score: {key}",
            )
            check(
                list(archive) == list(expected),
                f"Different chronological rule archive: {key}",
            )
            for rule, (size, text) in archive.items():
                tree = t["programs"][expected[rule]]
                check(
                    json.loads(text) == tree and size == nodes(tree),
                    f"First genotype mismatch: {key}/{rule}",
                )
            all_rules += len(archive)
            total_calls += 400
    expect = {
        (law, bank, gp)
        for law in ("matched", "additive")
        for bank in (6600, 6615)
        for gp in (809, 821)
    }
    check(seen == expect, "Missing predeclared path")
    return {
        "okapi_sha": PIN,
        "laws": sorted({k[0] for k in seen}),
        "data_seeds": [6600, 6615],
        "gp_seeds": [809, 821],
        "matched_path_digests": len(seen),
        "exact_fitness_calls": total_calls,
        "first_visited_rule_programs": all_rules,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "trace",
        type=Path,
        nargs="?",
        default=Path(__file__).with_name("traces.jsonl.gz"),
    )
    args = p.parse_args()
    print(json.dumps(verify(args.trace), indent=2))

"""Re-evaluate saved survivor choices using only objective scores and candidate sizes."""

import argparse
import json
import math
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PIN = "3eae0e5480cba0471cc5fda71b0c90f35b6a73f2"
POLICIES = {"raw": None, "round3": 3, "round6": 6}
GROUP = ["scenario", "data_seed", "gp_seed", "method"]


@dataclass(frozen=True)
class Candidate:
    index: int
    nodes_count: int


def check(predicate, description):
    if not predicate:
        raise ValueError(description)


def verify(folder):
    import okapi
    from loguru import logger
    from okapi.pareto import maximize
    from okapi.population import choose_pareto, choose_pareto_then_proximity

    logger.disable("okapi")
    source = Path(okapi.__file__).resolve().parent
    sha = subprocess.check_output(
        ["git", "-C", source, "rev-parse", "HEAD"], text=True
    ).strip()
    check(sha == PIN, "Use the pinned OKAPI checkout")
    frame = pd.read_csv(folder / "cohorts.csv.gz")
    raw = pd.read_csv(folder / "runs.csv.gz")
    selected = raw.query(
        'method in ["round3", "round6", "raw"] and split == "fit" and selection == "fit"'
    )
    runs = {k: g.sort_values("call_end") for k, g in frame.groupby(GROUP)}
    check(
        len(runs) == 684 and len(selected) == 1368,
        "Expected 684 runs and two scored checkpoints",
    )
    for row in selected.itertuples(index=False):
        key = tuple(getattr(row, c) for c in GROUP)
        f = runs[key]
        group = f[f.call_end <= row.calls]
        check(
            row.fit_calls == row.calls and row.validation_calls == row.archive_trees,
            key,
        )
        check(
            len(group) == row.complete_batches
            and group.n.sum() == row.complete_scored_rows,
            key,
        )
        check(np.all(f.n.cumsum().to_numpy() == f.call_end.to_numpy()), key)
    check(
        (frame.distinct_3 <= frame.distinct_raw).all()
        and (frame.distinct_6 <= frame.distinct_raw).all()
        and (frame.distinct_raw <= frame.n).all()
        and (frame.distinct_compressed_3 == 1).all(),
        "Reported resolution outside possible range",
    )
    check(
        (frame.tied_pairs_raw <= frame.tied_pairs_3).all()
        and (frame.tied_pairs_raw <= frame.tied_pairs_6).all(),
        "Reported tie-pair counts outside possible range",
    )
    wants = (
        (frame.method == "round3")
        & (frame.data_seed == 5900)
        & (frame.gp_seed == 709)
        & (frame.kind == "survival")
    )
    check(
        np.array_equal(wants, frame.candidate_fitness.notna()),
        "Unexpected same-cohort sampling set",
    )
    audit = frame[wants]
    check(audit.scenario.nunique() == 19, "Incomplete scenario coverage")
    changes = {name: Counter() for name in POLICIES}
    raw_vs_six = Counter()
    for row in audit.itertuples(index=False):
        fitness = np.asarray(json.loads(row.candidate_fitness), dtype=float)
        sizes = json.loads(row.candidate_nodes)
        native = json.loads(row.native_indices)
        check(len(fitness) == len(sizes) == row.n, "Candidate count")
        candidates = [Candidate(i, size) for i, size in enumerate(sizes)]
        for name, digits in POLICIES.items():
            scores = fitness if digits is None else np.round(fitness, digits)
            suffix = "raw" if digits is None else str(digits)
            frequencies = Counter(scores.tolist())
            check(
                len(frequencies) == getattr(row, f"distinct_{suffix}"),
                "Distinct score count",
            )
            check(
                sum(math.comb(n, 2) for n in frequencies.values())
                == getattr(row, f"tied_pairs_{suffix}"),
                "Score tie pairs",
            )
            front, _ = choose_pareto(
                candidates, scores[:, None], len(candidates), (maximize,), True
            )
            winners, _ = choose_pareto_then_proximity(
                candidates, scores[:, None], (maximize,), len(native), True
            )
            idx = [obj.index for obj in winners]
            check(
                idx == json.loads(getattr(row, name + "_indices")),
                "Different ordered survivor list",
            )
            if name == "round3":
                check(idx == native, "Default native replay differs")
            displaced = len(set(native) - set(idx))
            moved = sum(a != b for a, b in zip(native, idx, strict=True))
            check(len(front) == getattr(row, name + "_front_size"), "Front size")
            check(
                displaced == getattr(row, name + "_membership_replacements"),
                "Displaced members",
            )
            check(
                moved == getattr(row, name + "_positions_changed"), "Changed positions"
            )
            metrics = {
                "mean_nodes": np.mean(np.asarray(sizes)[idx]),
                "mean_raw_fitness": np.mean(fitness[idx]),
                "max_raw_fitness": np.max(fitness[idx]),
            }
            for metric, value in metrics.items():
                check(
                    abs(value - getattr(row, f"{name}_{metric}")) < 1e-12,
                    (name, metric),
                )
            changes[name].update(
                cohorts=1,
                different_set=int(displaced > 0),
                order_only=int(moved > 0 and displaced == 0),
                identical_order=int(moved == 0),
                swaps=displaced,
            )
        full, six = json.loads(row.raw_indices), json.loads(row.round6_indices)
        if set(full) != set(six):
            raw_vs_six["different_set"] += 1
        elif full != six:
            raw_vs_six["order_only"] += 1
        else:
            raw_vs_six["identical_order"] += 1
    resolution_summary = {}
    for policy, table in frame.groupby("method"):
        summary = {}
        for budget in (600, 1200):
            before = table[table.call_end <= budget]
            info = {
                "complete_scoring_cohorts": len(before),
                "candidate_rows": int(before.n.sum()),
            }
            for suffix in ("raw", "3", "6", "compressed_3"):
                info[f"sum_distinct_{suffix}"] = int(before[f"distinct_{suffix}"].sum())
                info[f"cohorts_collapsing_raw_values_{suffix}"] = int(
                    (before[f"distinct_{suffix}"] < before.distinct_raw).sum()
                )
            for suffix in ("3", "6"):
                info[f"additional_tied_pairs_{suffix}"] = int(
                    (before[f"tied_pairs_{suffix}"] - before.tied_pairs_raw).sum()
                )
            summary[str(budget)] = info
        resolution_summary[policy] = summary
    return {
        "okapi_sha": sha,
        "trajectories": len(runs),
        "checkpoints": len(selected),
        "complete_scoring_cohorts": len(frame),
        "resolution_by_trajectory_and_budget": resolution_summary,
        "audited_scenarios": audit.scenario.nunique(),
        "audited_cohorts": len(audit),
        "audited_candidate_rows": int(audit.n.sum()),
        "raw_vs_round6": dict(raw_vs_six),
        "changes_against_actual_round3": {k: dict(v) for k, v in changes.items()},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", nargs="?", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    print(json.dumps(verify(args.folder), indent=2))

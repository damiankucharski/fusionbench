"""Run the locked matched-cube experiment in the shared native replacement search."""

import argparse
import gzip
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from inputs import (
    POLICIES,
    POOLS,
    SEARCH_BASE,
    SUPPORT,
    generate,
    score,
    support_targets,
)
from loguru import logger
from threadpoolctl import threadpool_limits

from fusionbench.eval.native_switch import serialize
from fusionbench.eval.suite_point_gp import predict_tree

OUT = Path(__file__).resolve().parent / "results"


def law_run(bank, law, policy, fit, blocks, test, targets, components):
    from fusionbench.eval.replacement_search import run_replacement_search

    out = run_replacement_search(
        fit.predictions,
        fit.labels[law],
        None
        if policy == "ordinary"
        else [(b.predictions, b.labels[law]) for b in blocks],
        seed=SEARCH_BASE + bank,
        policy=policy,
        incumbent="mean",
    )
    key = {"bank": bank, "law": law, "policy": policy}
    births, stages, ancestor_components = [], [], {}
    q = targets[law]
    origin_excess = float(np.mean(2 * (targets["null"] - q) ** 2))
    for rec in out.records:
        p = predict_tree(rec.tree, SUPPORT)[:, 1]
        parts = tuple(k for k, v in components.items() if np.all(np.abs(p - v) < 1e-10))
        parents = list(rec.parent_ids)
        earlier = set().union(*(ancestor_components[i] for i in parents))
        ancestor_components[rec.birth_id] = earlier.union(parts)
        excess = float(np.mean(2 * (p - q) ** 2))
        births.append(
            {
                **key,
                "id": rec.birth_id,
                "gen": rec.generation,
                "parents": parents,
                "fit_brier": rec.discovery_brier,
                "n_ops": len(rec.tree.nodes["op_nodes"]),
                "provider_ids": list(rec.provider_ids),
                "confirmed": rec.confirmed,
                "unconfirmed_ancestor": rec.has_unconfirmed_ancestor,
                "p1": p.tolist(),
                "function": serialize(rec.tree.root),
                "partial": parts,
                "partial_ancestors": sorted(earlier),
                "exact_excess": excess,
                "hits_oracle": bool(np.max(np.abs(p - q)) < 1e-10),
                "beats_mean": excess < origin_excess - 1e-12,
            }
        )
    numbered = {row["id"]: row for row in births}
    for i, gen in enumerate(out.generations):
        prev, new = numbered[gen.incumbent_before], numbered[gen.incumbent_after]
        stages.append(
            {
                **key,
                "gen": i + 1,
                "proposals": list(gen.proposals),
                "eligible": list(gen.eligible),
                "survivors": list(gen.survivors),
                "gate_comparisons": gen.gate_comparisons,
                "gate_rows": gen.gate_rows,
                "before": gen.incumbent_before,
                "after": gen.incumbent_after,
                "promoted": gen.promoted,
                "gains": list(gen.gains),
                "lower_bounds": list(gen.lower_bounds),
                "before_excess": prev["exact_excess"],
                "after_excess": new["exact_excess"],
                "after_oracle": new["hits_oracle"],
            }
        )
    p = predict_tree(out.tree, SUPPORT)[:, 1]
    final = numbered[stages[-1]["after"]]
    if not np.allclose(p, final["p1"], rtol=0, atol=1e-12):
        raise AssertionError("Incumbent record differs from the deployed native tree")
    submitted = [numbered[i] for g in stages for i in g["proposals"]]
    promoted = [numbered[g["promoted"]] for g in stages if g["promoted"] is not None]
    final_parents = set()

    def add_ancestors(node):
        for i in node["parents"]:
            if i not in final_parents:
                final_parents.add(i)
                add_ancestors(numbered[i])

    add_ancestors(final)
    summary = {
        **key,
        **score(p, q, test, law),
        "initial_excess": origin_excess,
        "deploy_id": final["id"],
        "deploy_ops": final["n_ops"],
        "n_providers": len(final["provider_ids"]),
        "discovery_evaluations": out.discovery_evaluations,
        "proposal_count": out.proposal_count,
        "path_digest": out.path_digest,
        "unique_genotypes": len(
            {json.dumps(r["function"], sort_keys=True) for r in births}
        ),
        "unique_functions": len({tuple(np.round(r["p1"], 12)) for r in births}),
        "initial_oracle": any(r["hits_oracle"] for r in births if r["gen"] == 0),
        "offer_oracle": any(r["hits_oracle"] for r in submitted),
        "ever_oracle": any(r["hits_oracle"] for r in births),
        "earliest_oracle_gen": min(
            (r["gen"] for r in births if r["hits_oracle"]), default=None
        ),
        "ever_oracle_incumbent": numbered[stages[0]["before"]]["hits_oracle"]
        or any(g["after_oracle"] for g in stages),
        "promoted_oracle": any(r["hits_oracle"] for r in promoted),
        "final_oracle": final["hits_oracle"],
        "offered_better_than_mean": any(r["beats_mean"] for r in submitted),
        "final_better_than_mean": final["beats_mean"],
        "n_promotions": len(promoted),
        "ever_harmful_move": any(
            g["after_excess"] > g["before_excess"] + 1e-12 for g in stages
        ),
        "ever_nonoptimal_after": any(g["after_excess"] > 1e-12 for g in stages),
        "n_true_improving_moves": sum(
            g["after_excess"] < g["before_excess"] - 1e-12 for g in stages
        ),
        "n_adverse_moves": sum(
            g["after_excess"] > g["before_excess"] + 1e-12 for g in stages
        ),
        "best_offered_excess": min(r["exact_excess"] for r in submitted),
        "ever_below_one_update_bound": any(
            r["exact_excess"] < 3 / 32 - 1e-12 for r in submitted
        )
        if law == "signal"
        else None,
        "deploy_below_one_update_bound": (final["exact_excess"] < 3 / 32 - 1e-12)
        if law == "signal"
        else None,
        "n_final_ancestors": len(final_parents),
        "final_has_partial_ancestor": bool(final["partial_ancestors"]),
        "offered_exact_via_partial": any(
            r["hits_oracle"] and r["partial_ancestors"] for r in submitted
        ),
        "final_unconfirmed_ancestor": final["unconfirmed_ancestor"],
    }
    return summary, births, stages


def run_bank(bank):
    fit = generate(bank, 21, 128)
    blocks = [generate(bank, i, 128) for i in range(31, 37)]
    test = generate(bank, 41, 2048)
    targets, components, _ = support_targets(bank)
    results = []
    logger.disable("okapi")
    with threadpool_limits(limits=1):
        for law in POOLS:
            paths = set()
            for policy in POLICIES:
                result = law_run(
                    bank, law, policy, fit, blocks, test, targets, components
                )
                results.append(result)
                if policy != "strict_kl":
                    paths.add(result[0]["path_digest"])
            if len(paths) != 1:
                raise AssertionError("Open breeding paths disagree across deploy rules")
    return results


def run(workers, out):
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    with (
        gzip.open(out / "births.jsonl.gz", "wt") as birthfile,
        gzip.open(out / "generations.jsonl.gz", "wt") as genfile,
        ProcessPoolExecutor(max_workers=workers) as pool,
    ):
        for bank in pool.map(run_bank, range(48)):
            for result, births, stages in bank:
                rows.append(result)
                for handle, part in ((birthfile, births), (genfile, stages)):
                    for r in part:
                        handle.write(
                            json.dumps(
                                r, separators=(",", ":"), default=lambda x: x.item()
                            )
                            + "\n"
                        )
    pd.DataFrame(rows).to_csv(out / "search.csv", index=False)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("--workers", type=int, default=2)
    args.add_argument("--out", type=Path, default=OUT)
    opt = args.parse_args()
    run(opt.workers, opt.out)

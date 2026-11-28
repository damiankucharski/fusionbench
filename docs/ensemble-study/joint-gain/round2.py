"""Reproduce round two on matched singleton and two-input MIN laws."""

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from runpy import run_path
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from fusionbench.eval.suite_point_gp import predict_tree
from fusionbench.scenarios.native_switch import draw_native_switch

HERE = Path(__file__).resolve().parent
ENGINE = run_path(str(HERE.parent / "fold-promotion/search.py"))
BASELINES = run_path(str(HERE.parent / "seeded-checkpoint/references.py"))[
    "score_references"
]
SEEDS = tuple(range(288200, 288232))
POLICIES = ("fold_joint", "fold_fold", "fold_veto", "fold_pooled")
REFERENCES = ("mean", "top_n", "prob_lr", "regularized")
SIZES = (192, 1536)
LAWS = ("singleton", "min")
IDS = ["seed", "law", "n_rows"]


def target(draw, law):
    a, b = draw.parent, draw.child
    p = draw.preds[:, a, 1]
    q = p if law == "singleton" else np.minimum(p, draw.preds[:, b, 1])
    return q, (draw.uniform < q).astype(np.int64)


def native_witness(draw, law):
    from okapi.node import MinNode, ValueNode
    from okapi.tree import Tree

    a, b = draw.parent, draw.child
    ops = None
    if law == "min":
        child = ValueNode(None, draw.preds[:, b, 1:2], f"m{b}.npy")
        ops = [MinNode([child])]
    return Tree(ValueNode(ops, draw.preds[:, a, 1:2], f"m{a}.npy"))


def check_witness_grid(draw):
    # Exhaust all operand pairs; unused provider scores cannot affect this program.
    a, b = np.indices((63, 63)).reshape(2, -1) + 1
    grid = np.full((a.size, 6), 0.5)
    grid[:, draw.parent], grid[:, draw.child] = a / 64, b / 64
    np.testing.assert_array_equal(
        predict_tree(native_witness(draw, "min"), grid)[:, 1], np.minimum(a, b) / 64
    )


def fixed_menu(fitted, x, y, q, witness):
    from okapi.fold_gain import FoldGain

    observed = predict_tree(witness, x)
    np.testing.assert_array_equal(observed[:, 1], q)
    parent = predict_tree(fitted["archive"][fitted["anchor"]]["tree"], x)
    outcomes = np.eye(2)[y]
    losses = lambda p: ((p - outcomes) ** 2).sum(axis=1)
    gate = FoldGain(losses(parent), np.arange(len(y)), group_seed=fitted["group_seed"])
    assert ENGINE["array_digest"](gate.fold_ids) == fitted["fold_digest"]
    decision = gate.score(losses(observed))
    return {
        "anchor_is_witness": bool(np.array_equal(parent, observed)),
        "pooled_gain": decision.pooled_gain,
        "fold_gain": decision.robust_gain,
        "joint_gain": decision.joint_gain,
        "accept_pooled": decision.pooled_gain > 0,
        "accept_fold": decision.robust_gain > 0,
        "accept_joint": decision.supports_replacement,
        **{f"fold_gain_{i}": v for i, v in enumerate(decision.fold_gains)},
    }


def bank(seed):
    train, audit = (
        draw_native_switch(seed, split, n) for split, n in ((1, 1536), (2, 4096))
    )
    scores, diagnostics, witnesses = [], [], []
    check_witness_grid(audit)
    for law in LAWS:
        fit_q, fit_y = target(train, law)
        q, audit_y = target(audit, law)
        for n in SIZES:
            cell = {"seed": seed, "law": law, "n_rows": n}
            x, y = train.preds[:n], fit_y[:n]
            fitted = ENGINE["fit_search"](
                x,
                y,
                groups=np.arange(n),
                data_seed=seed,
                search_seed=seed + 1_000_000,
                path="fold",
                joint_study=True,
            )
            assert set(fitted["choices"]) == set(POLICIES)
            fitted["choices"] = {**fitted["choices"], "anchor": fitted["anchor"]}
            outputs, trace = ENGINE["score_search"](
                fitted,
                audit.preds,
                audit_y,
                audit_q=q,
                fit_q=fit_q[:n],
                signal_provider=train.parent if law == "singleton" else None,
            )
            by_name = {r["method"]: r for r in outputs}
            a, joint = by_name["anchor"], by_name["fold_joint"]
            ever = joint["best_ever_audit_excess"]
            available = min(joint["best_final_audit_excess"], a["audit_excess"])
            assert min(available - ever, joint["audit_excess"] - available) >= -1e-12
            assert trace[0]["exact_active"] == (law == "singleton")
            diagnostics.append(
                {
                    **cell,
                    "parent_index": train.parent,
                    "child_index": train.child,
                    "fit_x_sha256": ENGINE["array_digest"](x),
                    "audit_x_sha256": ENGINE["array_digest"](audit.preds),
                    **{
                        k: joint[k]
                        for k in (
                            "path_digest",
                            "fold_digest",
                            "anchor_digest",
                            "objective_evaluations",
                            "offspring_proposals",
                            "unique_evaluated",
                            "anchor_is_exact",
                            "exact_ever_visited",
                            "exact_in_final",
                            "n_exact_visited",
                            "n_exact_final",
                            "best_final_audit_excess",
                            "best_ever_audit_excess",
                        )
                    },
                    "first_exact_generation": next(
                        (r["generation"] for r in trace if r["exact_ever_visited"]),
                        None,
                    ),
                    "exact_in_final_plus_anchor": joint["exact_in_final"]
                    or a["selected_is_exact"],
                    "best_menu_audit_excess": available,
                    "retention_gap": available - ever,
                    "joint_choice_gap": joint["audit_excess"] - available,
                    "fold_choice_gap": by_name["fold_fold"]["audit_excess"] - available,
                }
            )
            scores.extend({**cell, **r} for r in outputs)
            witness = native_witness(train, law)
            np.testing.assert_array_equal(predict_tree(witness, audit.preds)[:, 1], q)
            witnesses.append({**cell, **fixed_menu(fitted, x, y, fit_q[:n], witness)})
            refs = BASELINES(
                x, y, audit.preds, audit_y, seed=seed + 1_000_000, audit_q=q
            )
            scores.extend({**cell, "method": name, **r} for name, r in refs.items())
    return scores, diagnostics, witnesses


def summarize(scores, diagnostics, witness):
    draws = np.random.default_rng(97827).integers(32, size=(10000, 32))
    methods = (*POLICIES, "anchor", *REFERENCES)
    assert len(diagnostics) == len(witness) == 4 * 32 and len(scores) == 4 * 32 * len(
        methods
    )
    for table in (scores, diagnostics, witness):
        keys = IDS + (["method"] if "method" in table else [])
        assert not table.duplicated(keys).any()
    assert (witness.accept_joint == (witness.accept_pooled & witness.accept_fold)).all()
    assert not (witness.anchor_is_witness & witness.accept_joint).any()
    for metric in ("fit_x_sha256", "audit_x_sha256"):
        assert diagnostics.groupby(["seed", "n_rows"])[metric].nunique().eq(1).all()
    result = []
    for (law, n), group in scores.groupby(["law", "n_rows"]):
        for metric in ("audit_excess", "audit_brier", "fit_brier"):
            wide = group.pivot(index="seed", columns="method", values=metric).reindex(
                SEEDS
            )
            assert set(wide) == set(methods) and np.isfinite(wide).all().all()
            for name in methods:
                for ref in (None, "fold_fold"):
                    v = (
                        wide[name].to_numpy()
                        if ref is None
                        else (wide[name] - wide[ref]).to_numpy()
                    )
                    lo, hi = np.quantile(v[draws].mean(axis=1), [0.025, 0.975])
                    result.append(
                        {
                            "law": law,
                            "n_rows": n,
                            "method": name,
                            "reference": ref or "level",
                            "metric": metric,
                            "estimate": v.mean(),
                            "ci_low": lo,
                            "ci_high": hi,
                        }
                    )
    return pd.DataFrame(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--out", type=Path, default=HERE / "results")
    args = parser.parse_args()
    started = datetime.now(UTC).isoformat()
    clock = perf_counter()
    with (
        threadpool_limits(limits=1),
        ProcessPoolExecutor(max_workers=args.workers) as pool,
    ):
        batches = list(pool.map(bank, SEEDS))
    tables = [
        pd.DataFrame(row for batch in batches for row in batch[i]) for i in range(3)
    ]
    summary = summarize(*tables)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, table in zip(
        ("scores", "diagnostics", "witness", "summary"), [*tables, summary], strict=True
    ):
        table.to_csv(args.out / f"r2-{name}.csv", index=False)
    manifest = {
        "protocol_revision": "377587c",
        "started_utc": started,
        "elapsed_seconds": perf_counter() - clock,
        "seeds": SEEDS,
        "workers": args.workers,
        "searches": len(tables[1]),
        "grid_pairs_per_bank": 3969,
        "sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__),
                HERE.parent / "fold-promotion/search.py",
                HERE / "protocol.md",
            )
        },
    }
    (args.out / "r2-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"Finished {len(tables[1])} real fold searches in {manifest['elapsed_seconds']:.1f} s"
    )


if __name__ == "__main__":
    main()

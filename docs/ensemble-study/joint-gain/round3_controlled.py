"""Measure sensitivity to three paired native search seeds on fixed strong and weak starts."""

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

from fusionbench.scenarios.checkpoint_bank import (
    generate_checkpoint_bank,
    generate_scene_pool,
    scene_sample_indices,
    select_checkpoints,
    signal_provider_index,
)
from fusionbench.scenarios.native_switch import draw_native_switch

ROOT = Path(__file__).resolve().parent
ENGINE = run_path(str(ROOT.parent / "fold-promotion" / "search.py"))
SOURCE = run_path(str(ROOT.parent / "scene-count" / "run.py"))
SEEDS = range(288300, 288324)
PATHS = ("pooled", "fold", "joint")
REFS = ("mean", "top_n", "prob_lr", "regularized")
METHODS = [
    "pooled_final",
    "anchor",
    *[f"{p}_{rule}" for p in PATHS for rule in ("pooled", "fold", "veto", "joint")],
    *REFS,
]


def data(seed, law):
    if law == "signal":
        slots = select_checkpoints(generate_checkpoint_bank(seed, law=law)["A"], 8)
        pool = generate_scene_pool(
            seed, law=law, n_fit_scenes=192, n_audit_scenes=512, scene_error_share=1.0
        )
        fit, test = pool["F"], pool["I"]
        ix = scene_sample_indices(24, 8)
        return (
            SOURCE["inputs"](fit, slots)[ix],
            fit.y[ix],
            fit.scene[ix],
            fit.posterior[ix],
            SOURCE["inputs"](test, slots),
            test.y,
            test.posterior,
            signal_provider_index(seed),
        )
    if law != "min":
        raise ValueError(law)
    fit, test = draw_native_switch(seed, 1, 1536), draw_native_switch(seed, 2, 4096)

    def conditional(draw):
        return np.minimum(draw.preds[:, draw.parent, 1], draw.preds[:, draw.child, 1])

    fq, tq = conditional(fit), conditional(test)
    return (
        fit.preds[:192],
        (fit.uniform[:192] < fq[:192]).astype(int),
        np.arange(192),
        fq[:192],
        test.preds,
        (test.uniform < tq).astype(int),
        tq,
        None,
    )


def fingerprint(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def run_bank(seed):
    scores, input_records = [], []
    for law in ("min", "signal"):
        x, y, groups, fq, xt, yt, tq, signal = data(seed, law)
        # These methods are fitted only once per bank, independent of the three GP seeds.
        refs = SOURCE["score_references"](
            x, y, xt, yt, seed=seed + 1_000_000, audit_q=tq, signal_provider=signal
        )
        ids = {
            name: fingerprint(a)
            for name, a in zip(
                ("fit_x", "fit_y", "group", "fit_q", "audit_x", "audit_y", "audit_q"),
                (x, y, groups, fq, xt, yt, tq),
                strict=True,
            )
        }
        input_records.append({"seed": seed, "law": law, **ids})
        bank_anchor = None
        for offset in (1, 2, 3):
            search_seed = seed + offset * 1_000_000
            cell = {"seed": seed, "law": law, "search_seed": search_seed}
            for path in PATHS:
                fit = ENGINE["fit_search"](
                    x,
                    y,
                    groups=groups,
                    data_seed=seed,
                    search_seed=search_seed,
                    path=path,
                    joint_study=True,
                )
                evidence = (fit["anchor"], fit["fold_digest"])
                if bank_anchor is None:
                    bank_anchor = evidence
                if evidence != bank_anchor:
                    raise ValueError(
                        "Initial anchor or fold assignments changed across matched paths"
                    )
                rows, _ = ENGINE["score_search"](
                    fit, xt, yt, audit_q=tq, fit_q=fq, signal_provider=signal
                )
                scores.extend(
                    {**cell, **r, "n_providers": r["unique_providers"]} for r in rows
                )
            for name, r in refs.items():
                scores.append(
                    {
                        **cell,
                        "method": name,
                        **r,
                        "selected_is_exact": r["selected_is_signal"]
                        if signal is not None
                        else None,
                    }
                )
    return scores, input_records


def summarize(scores):
    index = ["law", "seed", "search_seed", "method"]
    if scores.duplicated(index).any() or len(scores) != len(SEEDS) * 2 * 3 * len(
        METHODS
    ):
        raise ValueError("Missing or duplicate scores")
    records = []
    for (law, seed, method), r in scores.groupby(["law", "seed", "method"]):
        if sorted(r.search_seed) != [
            seed + 1_000_000,
            seed + 2_000_000,
            seed + 3_000_000,
        ]:
            raise ValueError("Incomplete search seed pairing")
        v = r.audit_excess.to_numpy()
        if not np.isfinite(v).all():
            raise ValueError("Missing audit scores")
        records.append(
            {
                "law": law,
                "seed": seed,
                "method": method,
                "mean": v.mean(),
                "range": np.ptp(v),
                "worst": v.max(),
                "mean_provider_count": r.n_providers.mean(),
                "n_exact": r.selected_is_exact.eq(True).sum()
                if r.selected_is_exact.notna().all()
                else np.nan,
            }
        )
    banks = pd.DataFrame(records)
    draws = np.random.default_rng(97828).integers(len(SEEDS), size=(10000, len(SEEDS)))
    pairs = [
        ("joint_joint", ref)
        for ref in (
            "fold_joint",
            "pooled_joint",
            "fold_fold",
            "pooled_final",
            "anchor",
            *REFS,
        )
    ]
    pairs += [("joint_pooled", "fold_pooled"), ("fold_joint", "fold_fold")]
    contrasts = []
    for law, frame in banks.groupby("law"):
        for metric in ("mean", "range", "worst"):
            wide = frame.pivot(index="seed", columns="method", values=metric).loc[
                list(SEEDS), METHODS
            ]
            for method, ref in pairs:
                delta = (wide[method] - wide[ref]).to_numpy()
                lo, hi = np.quantile(delta[draws].mean(axis=1), [0.025, 0.975])
                contrasts.append(
                    {
                        "law": law,
                        "metric": metric,
                        "method": method,
                        "reference": ref,
                        "estimate": delta.mean(),
                        "ci_low": lo,
                        "ci_high": hi,
                        "n_banks": len(delta),
                    }
                )
    return banks, pd.DataFrame(contrasts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2, choices=(1, 2, 3))
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    started = datetime.now(UTC).isoformat()
    tic = perf_counter()
    with (
        threadpool_limits(limits=1),
        ProcessPoolExecutor(max_workers=args.workers) as executor,
    ):
        batches = list(executor.map(run_bank, SEEDS))
    scores = pd.DataFrame(row for rows, _ in batches for row in rows)
    banks, contrasts = summarize(scores)
    args.out.mkdir(exist_ok=True, parents=True)
    for name, frame in (("scores", scores), ("banks", banks), ("contrasts", contrasts)):
        frame.to_csv(args.out / f"r3-controlled-{name}.csv", index=False)
    manifest = {
        "protocol": "377587c",
        "okapi": "697335b",
        "elapsed_seconds": perf_counter() - tic,
        "started_utc": started,
        "workers": args.workers,
        "numerical_threads_per_worker": 1,
        "native_searches": len(SEEDS) * 2 * 3 * 3,
        "scores": len(scores),
        "numpy": np.__version__,
        "reference_seed": "data seed + 1,000,000; references reused across all three GP seeds",
        "inputs": [row for _, inputs in batches for row in inputs],
    }
    (args.out / "r3-controlled-run.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(
        f"Finished {manifest['native_searches']} native paths, {len(scores)} scores in {manifest['elapsed_seconds']:.2f} s."
    )


if __name__ == "__main__":
    main()

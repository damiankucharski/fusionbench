"""Run a registered round. Make each policy decision on its source bank only."""

import argparse
import gzip
import json
import multiprocessing
from functools import partial
from hashlib import sha256
from pathlib import Path
from runpy import run_path
from time import perf_counter

import numpy as np
import pandas as pd
from banks import sources
from okapi.case_mix import case_mix_weights
from okapi.conditional_weighting import conditional_case_mix_weights
from threadpoolctl import threadpool_limits

from fusionbench.methods.averaging import SimpleAverage, TopN
from fusionbench.methods.stacking import StackingLR

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
prior = run_path(str(HERE.parent / "environment-ten" / "run_env.py"))
native, measures, digest = (prior[k] for k in ("native", "measures", "digest"))
CORE = ("native", "mean", "top_n", "lr", "guarded", "single", "half", "replicated")
EXTRA = {2: ("raw",), 3: ("gate_sham",), 8: ("random_gate",)}
CHEAP = ("mean", "top_n", "lr")


def decisions(b, r):
    ones = np.ones(len(b.y))
    full, info = case_mix_weights(b.env, b.ctx)
    cap, clipped = case_mix_weights(b.env, b.ctx, cap=2)
    unconditional, action = (ones, "native")
    if info["effective_n"] >= 0.65 * len(ones):
        unconditional, action = full, "full"
    elif clipped["effective_n"] >= 0.65 * len(ones):
        unconditional, action = cap, "cap2"
    neutral = {
        "gate_active": False,
        "gate_provider": None,
        "action": "native",
        "reason": "reference",
    }
    original = (ones, neutral)
    plans = {
        "guarded": (unconditional, dict(neutral, action=action, reason="ess_only"))
    }
    kwargs = {"seed": b.seed, "diagnostic_indices": b.screen_idx}
    for name in ("single", "half", "replicated", *EXTRA.get(r, ())):
        if name == "gate_sham":
            rng = np.random.default_rng([b.seed, 1143])
            env = b.env.copy()
            for c in np.unique(b.ctx):
                group = b.ctx == c
                env[group] = rng.permutation(env[group])
            _, ev = conditional_case_mix_weights(b.x, b.y, env, b.ctx, **kwargs)
            w = unconditional if ev["gate_active"] else ones
            ev = dict(
                ev,
                action=action if ev["gate_active"] else "native",
                weight_environment="original",
            )
        elif name == "random_gate":
            active = bool(np.random.default_rng([b.seed, 1148]).random() < 0.5)
            w = unconditional if active else ones
            ev = dict(
                neutral,
                gate_active=active,
                action=action if active else "native",
                reason="coin",
            )
        else:
            mode = {
                "single": "single",
                "half": "single",
                "replicated": "replicated",
                "raw": "raw",
            }[name]
            w, ev = conditional_case_mix_weights(
                b.x,
                b.y,
                b.env,
                b.ctx,
                mode=mode,
                strength=0.5 if name == "half" else 1.0,
                **kwargs,
            )
        plans[name] = (w, ev)
    return original, plans, info


def fit_model(method, b, w, queries):
    weighted = np.ptp(w) > 0
    model = (
        {
            "mean": SimpleAverage,
            "top_n": lambda: TopN(metric="brier"),
            "lr": lambda: StackingLR(c_value=1),
        }[method]()
        if method in CHEAP
        else native()
    )
    path = sha256()

    def visit(tree, _pred):
        path.update((repr(tree.structural_key()) + "\n").encode())

    kw = {} if method in CHEAP else {"on_gp_candidate": visit}
    if weighted:
        kw["sample_weight"] = w
    model.fit(
        b.x,
        b.y,
        task="binary" if b.x.shape[2] == 2 else "multiclass",
        n_classes=b.x.shape[2],
        **kw,
    )
    chosen = getattr(model, "chosen_", method)
    tree = getattr(model, "trees_", {}).get(chosen)
    meta = {
        "chosen": chosen,
        "path_hash": path.hexdigest() if tree is not None else method,
        "tree_hash": sha256(repr(tree.structural_key()).encode()).hexdigest()
        if tree is not None
        else method,
        "fit_hash": digest(model.predict(b.x)),
        "providers": len(tree.unique_value_node_ids)
        if tree is not None
        else getattr(model, "n_selected_", b.x.shape[1]),
    }
    pred = {h: np.asarray(model.predict(x)) for h, x in queries.items()}
    for p in pred.values():
        if (
            not np.isfinite(p).all()
            or p.min() < -1e-8
            or not np.allclose(p.sum(1), 1, atol=1e-6, rtol=0)
        ):
            raise ValueError("Invalid forecast")
    return meta, pred


def evaluate(b, r):
    start = perf_counter()
    uniform, plans, info = decisions(b, r)
    meta = {
        "round": r,
        "cohort": b.cohort,
        "family": b.family,
        "seed": b.seed,
        "n_fit": len(b.y),
        "n_screen": len(b.screen_idx),
        "source_x_hash": digest(b.x),
        "source_y_hash": digest(b.y),
        "true_shortcut": b.shortcut,
        "true_specialist": b.specialist,
    }
    queries = {digest(x): x for _, x, _, _, _ in b.worlds}
    rows = []
    traces = {}
    cache = {}
    for method in (*CORE, *EXTRA.get(r, ())):
        w, ev = plans.get(method, uniform)
        wh = digest(w)
        applied = bool(np.ptp(w) > 0)
        key = method if method in CHEAP else wh if applied else "native"
        if key not in cache:
            cache[key] = fit_model(method, b, w, queries)
        fields, pred = cache[key]
        decision = {
            "method": method,
            "gate_active": ev["gate_active"],
            "gate_provider": ev["gate_provider"],
            "applied": applied,
            "policy": ev["action"],
            "reason": ev["reason"],
            "weight_hash": wh,
            "ess": float(w.sum() ** 2 / (w @ w)),
            "weight_max": float(w.max()),
            "stage": "discovery+validation+refit" if applied else "uniform",
        }
        traces[method] = {**decision, **fields, "evidence": ev}
        for world, x, y, mass, q in b.worlds:
            h = digest(x)
            p = pred[h]
            rows.append(
                {
                    **meta,
                    **decision,
                    **fields,
                    "world": world,
                    **measures(p, y, mass, q),
                    "oracle_brier": None
                    if q is None
                    else float(mass @ (2 * q * (1 - q))),
                    "test_hash": digest(p),
                    "test_x_hash": h,
                    "target_hash": digest(
                        y if q is None else np.column_stack((mass, q))
                    ),
                }
            )
    trace = {
        **meta,
        "env_hash": digest(b.env),
        "ctx_hash": digest(b.ctx),
        "screen_hash": digest(b.screen_idx),
        "full_weights": info,
        "methods": traces,
        "native_fits": len(cache) - 3,
        "seconds": perf_counter() - start,
    }
    return rows, trace


def measure(b, r):
    with threadpool_limits(limits=1):
        return evaluate(b, r)


def main(r, workers):
    if r == 10:
        from analyze import committed_selection

        committed_selection()
    start = perf_counter()
    OUT.mkdir(exist_ok=True)
    records = []
    logs = []
    with multiprocessing.get_context("fork").Pool(workers) as pool:
        for rows, trace in pool.imap(partial(measure, r=r), sources(r), chunksize=2):
            records.extend(rows)
            logs.append(trace)
    stem = f"r{r:02}"
    pd.DataFrame(records).to_csv(OUT / f"{stem}.csv", index=False)
    with (
        (OUT / f"{stem}-source.jsonl.gz").open("wb") as handle,
        gzip.GzipFile(fileobj=handle, mode="wb", mtime=0) as gz,
    ):
        for row in logs:
            gz.write((json.dumps(row, allow_nan=False) + "\n").encode())
    counts = {
        "round": r,
        "sources": len(logs),
        "target_conditions": len(records) // (len(CORE) + len(EXTRA.get(r, ()))),
        "rows": len(records),
        "native_fits": sum(t["native_fits"] for t in logs),
        "seconds": perf_counter() - start,
        "workers": workers,
    }
    (OUT / f"{stem}-time.json").write_text(json.dumps(counts, indent=2) + "\n")
    print(counts, flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("round", type=int, choices=range(1, 11))
    cli.add_argument("--workers", type=int, choices=range(1, 5), default=4)
    args = cli.parse_args()
    main(args.round, args.workers)

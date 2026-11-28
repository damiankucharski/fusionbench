"""Fit one preregistered round, reusing identical native fits within each source."""

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
from threadpoolctl import threadpool_limits

from fusionbench.methods.averaging import SimpleAverage, TopN
from fusionbench.methods.stacking import StackingLR

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
prior = run_path(str(HERE.parent / "environment-ten" / "run_env.py"))
native, measures, digest = (prior[k] for k in ("native", "measures", "digest"))
ALL = ("discovery", "validation", "refit")
CORE = ("native", "mean", "top_n", "lr", "full", "cap2", "guarded")
EXTRAS = {
    2: ("shuffled",),
    3: ("discovery_only",),
    4: ("selection_only", "refit_only"),
    5: ("cap15",),
    8: ("permuted_weights",),
}
STAGES = {
    "discovery_only": ("discovery",),
    "selection_only": ("validation",),
    "refit_only": ("refit",),
}


def plans(b, r):
    ones = np.ones(len(b.y))
    full, report = case_mix_weights(b.env, b.ctx)
    cap, clipped = case_mix_weights(b.env, b.ctx, cap=2)
    uniform = (ones, (), "native", "uniform")
    full_plan = (full, ALL, "full", report["reason"])
    cap_plan = (cap, ALL, "cap2", clipped["reason"])
    guarded = uniform
    if report["effective_n"] >= 0.65 * len(ones):
        guarded = full_plan
    elif clipped["effective_n"] >= 0.65 * len(ones):
        guarded = cap_plan
    policies = {"full": full_plan, "cap2": cap_plan, "guarded": guarded}
    for name in EXTRAS.get(r, ()):
        if name in STAGES:
            policies[name] = (full, STAGES[name], "full", report["reason"])
        elif name == "shuffled":
            w, info = case_mix_weights(
                np.random.default_rng([b.seed, 1042]).permutation(b.env), b.ctx
            )
            policies[name] = (w, ALL, name, info["reason"])
        elif name == "permuted_weights":
            policies[name] = (
                np.random.default_rng([b.seed, 1048]).permutation(full),
                ALL,
                name,
                report["reason"],
            )
        else:
            w, info = case_mix_weights(b.env, b.ctx, cap=1.5)
            policies[name] = (w, ALL, name, info["reason"])
    return uniform, policies, report


def evaluate_bank(b, r):
    start = perf_counter()
    common = {
        "cohort": b.cohort,
        "family": b.family,
        "seed": b.seed,
        "round": r,
        "n_fit": len(b.x),
        "source_x_hash": digest(b.x),
        "source_y_hash": digest(b.y),
    }
    opts = {
        "task": "binary" if b.x.shape[2] == 2 else "multiclass",
        "n_classes": b.x.shape[2],
    }
    uniform, policies, report = plans(b, r)
    weights_hash = digest(uniform[0])
    tests = {digest(x): x for _, x, _, _, _ in b.worlds}
    _, groupid = np.unique(b.groups, return_inverse=True)
    n_groups = int(groupid.max() + 1)
    rows = []
    fits = {}
    traces = {}
    exact_forecasts = {}
    for method in (*CORE, *EXTRAS.get(r, ())):
        w, st, policy, reason = policies.get(method, uniform)
        effective = st if np.ptp(w) > 0 else ()
        wh = digest(w)
        is_native = method not in ("mean", "top_n", "lr")
        key = (wh if effective else weights_hash, effective) if is_native else (method,)
        if key not in fits:
            model = (
                native()
                if is_native
                else {
                    "mean": SimpleAverage,
                    "top_n": lambda: TopN(metric="brier"),
                    "lr": lambda: StackingLR(c_value=1),
                }[method]()
            )
            path = sha256()

            def visited(tree, _pred, current=path):
                current.update((repr(tree.structural_key()) + "\n").encode())

            kw = {"on_gp_candidate": visited} if is_native else {}
            if effective:
                kw.update(sample_weight=w, weight_stages=effective)
            model.fit(b.x, b.y, **opts, **kw)
            chosen = getattr(model, "chosen_", method)
            tree = getattr(model, "trees_", {}).get(chosen)
            ph = path.hexdigest() if is_native else method
            th = (
                sha256(repr(tree.structural_key()).encode()).hexdigest()
                if tree is not None
                else method
            )
            fields = {
                "chosen": chosen,
                "path_hash": ph,
                "tree_hash": th,
                "fit_hash": digest(model.predict(b.x)),
                "providers": len(tree.unique_value_node_ids)
                if tree is not None
                else getattr(model, "n_selected_", b.x.shape[1]),
            }
            predictions = {h: np.asarray(model.predict(x)) for h, x in tests.items()}
            for pred in predictions.values():
                if (
                    not np.isfinite(pred).all()
                    or pred.min() < -1e-8
                    or not np.allclose(pred.sum(1), 1, atol=1e-6, rtol=0)
                ):
                    raise ValueError("Invalid class probabilities")
            fits[key] = (fields, predictions)
        fields, predictions = fits[key]
        gw = np.bincount(groupid, weights=w)
        meta = {
            "ess": w.sum() ** 2 / (w @ w),
            "group_ess": gw.sum() ** 2 / (gw @ gw),
            "groups": n_groups,
            "weight_max": w.max(),
            "weight_min": w.min(),
            "weight_hash": wh,
            "policy": policy,
            "reason": reason,
            "stage": "+".join(effective) if effective else "uniform",
            "method": method,
        }
        traces[method] = {**meta, **fields}
        if r == 10 and b.cohort == "focus":
            if len(predictions) != 1:
                raise ValueError("Focal worlds must share the same support")
            exact_forecasts[method] = next(iter(predictions.values())).tolist()
        for world, x, y, mass, q in b.worlds:
            h = digest(x)
            p = predictions[h]
            rows.append(
                {
                    **common,
                    **meta,
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
    return rows, {
        **common,
        "env_hash": digest(b.env),
        "ctx_hash": digest(b.ctx),
        "full_weights": report,
        "methods": traces,
        **({"exact_forecasts": exact_forecasts} if exact_forecasts else {}),
        "native_fits": len(fits) - 3,
        "seconds": perf_counter() - start,
    }


def measure(b, r):
    with threadpool_limits(limits=1):
        return evaluate_bank(b, r)


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
        for entry in logs:
            gz.write((json.dumps(entry, allow_nan=False) + "\n").encode())
    times = {
        "round": r,
        "sources": len(logs),
        "target_conditions": len(records) // (len(CORE) + len(EXTRAS.get(r, ()))),
        "rows": len(records),
        "native_fits": sum(x["native_fits"] for x in logs),
        "seconds": perf_counter() - start,
        "workers": workers,
    }
    (OUT / f"{stem}-time.json").write_text(json.dumps(times, indent=2) + "\n")
    print(times, flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("round", type=int, choices=range(1, 11))
    cli.add_argument("--workers", type=int, choices=range(1, 5), default=4)
    args = cli.parse_args()
    main(args.round, args.workers)

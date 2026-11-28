"""Fit one registered round, then score frozen forecasts on matched targets."""

import argparse
import gzip
import json
import multiprocessing
from functools import partial
from pathlib import Path
from runpy import run_path
from time import perf_counter

import numpy as np
import pandas as pd
from banks import sources
from okapi.environment_stability import environment_screen
from okapi.provider_use import provider_use_evidence
from threadpoolctl import threadpool_limits

from fusionbench.methods.averaging import SimpleAverage, TopN
from fusionbench.methods.provider_reliance import (
    limit_reliance,
    marginal_prediction,
    source_donors,
)
from fusionbench.methods.stacking import StackingLR

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
prior = run_path(str(HERE.parent / "environment-ten" / "run_env.py"))
digest, native, measures = (prior[n] for n in ("digest", "native", "measures"))
EXTRAS = {
    2: ("permuted_env",),
    4: ("quarter", "three_quarter", "full"),
    5: ("half_d2", "half_d32"),
    8: ("direct_only", "routing_only"),
}
COMMON = (
    "native",
    "mean",
    "top_n",
    "lr",
    "hard",
    "half",
    "protect",
    "adaptive",
    "shuffled_evidence",
)


def folds(groups, seed):
    order = np.random.default_rng([seed, 817]).permutation(np.unique(groups))
    return np.isin(groups, order[::2]).astype(int)


def strength(method, evidence):
    direct = evidence["direct_gain"] > 0.02
    routing = evidence["routing_gain"] > 0.02
    if method in {"adaptive", "shuffled_evidence", "permuted_env"}:
        return 0.0 if routing else 0.25 if direct else 0.75
    if method == "protect":
        return 0.0 if direct or routing else 0.5
    if method == "direct_only":
        return 0.0 if direct else 0.5
    if method == "routing_only":
        return 0.0 if routing else 0.5
    return {
        "quarter": 0.25,
        "half": 0.5,
        "three_quarter": 0.75,
        "full": 1.0,
        "half_d2": 0.5,
        "half_d32": 0.5,
    }[method]


def used(model, default):
    tree = getattr(model, "trees_", {}).get(getattr(model, "chosen_", None))
    return (
        len(tree.unique_value_node_ids)
        if tree is not None
        else getattr(model, "n_selected_", default)
    )


def forecast_bank(r, b, queries):
    """Fit from b.x, y and source metadata only. queries contains unlabelled features."""
    x, y, idx = b.x, b.y, b.idx
    options = dict(
        task="binary" if x.shape[2] == 2 else "multiclass", n_classes=x.shape[2]
    )
    models = {
        "native": native(),
        "mean": SimpleAverage(),
        "top_n": TopN(metric="brier"),
        "lr": StackingLR(c_value=1),
    }
    forecasts, metadata = {}, {}
    for method, model in models.items():
        model.fit(x, y, **options)
        forecasts[method] = np.asarray(model.predict(queries), dtype=float)
        metadata[method] = dict(
            drop=None,
            strength=0.0,
            direct_gain=None,
            routing_gain=None,
            providers=used(model, x.shape[1]),
        )
    fit_native = models["native"]
    half_ids = folds(b.groups[idx], b.seed)
    real_env = environment_screen(x[idx], y[idx], b.env, contexts=b.ctx)
    flags = {"original": real_env["drop"]}
    if r == 2:
        fake = np.random.default_rng([b.seed, 819]).permutation(b.env)
        flags["permuted"] = environment_screen(x[idx], y[idx], fake, contexts=b.ctx)[
            "drop"
        ]
    evidence = {}
    marginal = {}
    shuffled_y = np.random.default_rng([b.seed, 818]).permutation(y[idx])

    def diagnostic(flag, shuffled):
        key = (flag, shuffled)
        if key not in evidence:
            labels = shuffled_y if shuffled else y[idx]
            evidence[key] = provider_use_evidence(
                x[idx], labels, flag, fold_ids=half_ids
            )
        return evidence[key]

    if b.selector is not None:
        diagnostic(b.selector, False)
    for method in (*COMMON[4:], *EXTRAS.get(r, ())):
        flag = flags["permuted" if method == "permuted_env" else "original"]
        F = forecasts["native"]
        meta = dict(metadata["native"])
        if flag is None:
            forecasts[method], metadata[method] = F, meta
            continue
        e = diagnostic(flag, method == "shuffled_evidence")
        meta.update(
            drop=flag, direct_gain=e["direct_gain"], routing_gain=e["routing_gain"]
        )
        if method == "hard":
            keep = np.delete(np.arange(x.shape[1]), flag)
            reduced = native() if len(keep) > 1 else SimpleAverage()
            reduced.fit(np.ascontiguousarray(x[:, keep, :]), y, **options)
            forecasts[method] = reduced.predict(
                np.ascontiguousarray(queries[:, keep, :])
            )
            meta.update(strength=1.0, providers=used(reduced, len(keep)))
        else:
            s = strength(method, e)
            count = {"half_d2": 2, "half_d32": 32}.get(method, 8)
            key = (flag, count)
            if s and key not in marginal:
                marginal[key] = marginal_prediction(
                    fit_native.predict, queries, flag, source_donors(x, flag, count)
                )
            forecasts[method] = limit_reliance(F, marginal[key], s) if s else F
            meta["strength"] = s
        metadata[method] = meta
    return (
        forecasts,
        metadata,
        {
            "flags": flags,
            "fold_hash": digest(half_ids),
            "screen_reason": real_env["reason"],
            "evidence": {
                f"{j}/{'shuffled' if sh else 'real'}": e
                for (j, sh), e in evidence.items()
            },
            "selector": b.selector,
            "native_fits": 1 + int(flags["original"] is not None),
        },
    )


def measure(b, r):
    start = perf_counter()
    # Assemble source predictions and distinct query supports once. Target laws enter only scoring.
    blocks = {digest(b.x): b.x}
    for _, query, _, _, _ in b.worlds:
        blocks.setdefault(digest(query), query)
    slices = {}
    end = 0
    for key, block in blocks.items():
        slices[key] = slice(end, end + len(block))
        end += len(block)
    joined = np.concatenate(list(blocks.values()), axis=0)
    with threadpool_limits(limits=1):
        forecasts, meta, info = forecast_bank(r, b, joined)
    shared = dict(
        round=r,
        cohort=b.cohort,
        family=b.family,
        seed=b.seed,
        n_fit=len(b.x),
        n_screen=len(b.idx),
        true_shortcut=b.shortcut,
        true_specialist=b.specialist,
        source_x_hash=digest(b.x),
        source_y_hash=digest(b.y),
    )
    rows = []
    for method, forecast in forecasts.items():
        if (
            not np.isfinite(forecast).all()
            or forecast.min() < -1e-7
            or not np.allclose(forecast.sum(1), 1.0, atol=1e-6, rtol=0)
        ):
            raise ValueError(f"Invalid forecast for {b.family}/{method}")
        fit_hash = digest(forecast[slices[shared["source_x_hash"]]])
        for world, query, labels, mass, q in b.worlds:
            key = digest(query)
            p = forecast[slices[key]]
            rows.append(
                {
                    **shared,
                    "world": world,
                    "method": method,
                    **meta[method],
                    **measures(p, labels, mass, q),
                    "oracle_brier": None
                    if q is None
                    else float(mass @ (2 * q * (1 - q))),
                    "fit_hash": fit_hash,
                    "test_hash": digest(p),
                    "test_x_hash": key,
                    "target_hash": digest(
                        labels if mass is None else np.column_stack((mass, q))
                    ),
                }
            )
    return rows, {**shared, **info, "seconds": perf_counter() - start}


def main(r, workers):
    if r == 10:
        from analyze import committed_selection

        committed_selection()
    start = perf_counter()
    values, traces = [], []
    with multiprocessing.get_context("fork").Pool(workers) as pool:
        for rows, trace in pool.imap(partial(measure, r=r), sources(r), chunksize=2):
            values.extend(rows)
            traces.append(trace)
    OUT.mkdir(exist_ok=True)
    stem = f"r{r:02}"
    pd.DataFrame(values).to_csv(OUT / f"{stem}.csv", index=False)
    with (
        (OUT / f"{stem}-source.jsonl.gz").open("wb") as stream,
        gzip.GzipFile(fileobj=stream, mode="wb", mtime=0) as file,
    ):
        for trace in traces:
            file.write((json.dumps(trace, allow_nan=False) + "\n").encode())
    times = dict(
        round=r,
        sources=len(traces),
        target_conditions=len(values) // (len(COMMON) + len(EXTRAS.get(r, ()))),
        rows=len(values),
        native_fits=sum(t["native_fits"] for t in traces),
        workers=workers,
        seconds=perf_counter() - start,
    )
    (OUT / f"{stem}-time.json").write_text(json.dumps(times, indent=2) + "\n")
    print(times, flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("round", type=int, choices=range(1, 11))
    cli.add_argument("--workers", type=int, choices=range(1, 5), default=4)
    args = cli.parse_args()
    main(args.round, args.workers)

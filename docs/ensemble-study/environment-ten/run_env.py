"""Run one registered round on source data, then score the same fits in all target laws."""

import argparse
import gzip
import json
import multiprocessing
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import pandas as pd
from okapi.environment_stability import environment_screen
from threadpoolctl import threadpool_limits

from fusionbench import generate_dataset
from fusionbench.methods.averaging import SimpleAverage, TopN
from fusionbench.methods.native_pattern import RegularizedNativePatternOKAPI
from fusionbench.methods.stacking import StackingLR
from fusionbench.transfer import load_preset

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
GROUPED = {"grouped_rare_event", "shared_error_bloc", "rare_scene_specialist"}
CANDIDATES = ("raw", "adjusted", "strict", "sham")
REFS = ("native", "mean", "top_n", "lr")


def digest(a):
    return sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def standard_sources(r):
    base = 840000 + 1000 * r + 400
    n = 8 if r == 10 else 4 if r == 9 else 2
    for family in load_preset("standard-v6")["scenarios"]:
        for seed in range(base, base + n):
            d = generate_dataset(family, seed=seed, n_per_split=512)
            x = d.fit_preds
            idx = np.arange(len(x))
            if family in GROUPED:
                scenes = np.asarray(d.meta["evaluation_groups"]["fit"]["scene_id"])
                rng = np.random.default_rng([seed, 217])
                idx = np.sort(
                    [rng.choice(np.flatnonzero(scenes == g)) for g in np.unique(scenes)]
                )
            avg = x[idx].mean(1)
            dis = np.square(x[idx] - avg[:, None, :]).sum(axis=2).mean(axis=1)
            conf = avg.max(axis=1)
            yield SimpleNamespace(
                cohort="standard",
                family=family,
                seed=seed,
                x=x,
                y=d.fit_y,
                idx=idx,
                env=(dis > np.median(dis)).astype(int),
                ctx=(conf > np.median(conf)).astype(int),
                shortcut=None,
                specialist=None,
                worlds=[("test", d.test_preds, d.test_y, None, None)],
            )


def sources(r):
    from banks_env import focus_banks

    yield from standard_sources(r)
    for family, seed, b, targets in focus_banks(r):
        yield SimpleNamespace(
            cohort="focus",
            family=family,
            seed=seed,
            x=b.x,
            y=b.y,
            idx=np.arange(len(b.y)),
            env=b.env,
            ctx=b.context,
            shortcut=b.shortcut_index,
            specialist=b.specialist_index,
            worlds=[(name, t.x, None, t.mass, t.q) for name, t in targets.items()],
        )


def native():
    return RegularizedNativePatternOKAPI(seed=0, split_seed=0, gp_validation_top_k=8)


def fit_on_source(x, y, idx, env, ctx, seed, methods):
    mask_all = np.arange(x.shape[1])
    options = {
        "task": "binary" if x.shape[2] == 2 else "multiclass",
        "n_classes": x.shape[2],
    }

    def fit(model, indices):
        # Fancy indexing along the provider axis can alter floating summation order.
        data = (
            x if len(indices) == x.shape[1] else np.ascontiguousarray(x[:, indices, :])
        )
        return model.fit(data, y, **options), indices

    bank = {
        name: fit(model, mask_all)
        for name, model in zip(
            REFS,
            (native(), SimpleAverage(), TopN(metric="brier"), StackingLR(c_value=1.0)),
            strict=True,
        )
    }
    cache = {None: bank["native"]}
    info = {}
    for method in methods:
        source_env = (
            np.random.default_rng([seed, 283]).permutation(env)
            if method == "sham"
            else env
        )
        evidence = environment_screen(
            x[idx],
            y[idx],
            source_env,
            contexts=ctx,
            mode="raw" if method == "raw" else "adjusted",
            threshold=0.15 if method == "strict" else 0.10,
            z=2.58 if method == "strict" else 1.96,
            min_cell=8,
        )
        removed = evidence["drop"]
        if removed not in cache:
            keep = mask_all[mask_all != removed]
            cache[removed] = fit(native() if len(keep) > 1 else SimpleAverage(), keep)
        bank[method] = cache[removed]
        info[method] = evidence
    return bank, info, len(cache)


def probabilities(model, keep, x):
    query = x if len(keep) == x.shape[1] else np.ascontiguousarray(x[:, keep, :])
    p = np.asarray(model.predict(query), dtype=float)
    if (
        p.shape != (len(x), x.shape[2])
        or not np.isfinite(p).all()
        or p.min() < -1e-8
        or not np.allclose(p.sum(1), 1, atol=1e-6, rtol=0)
    ):
        raise ValueError("Invalid probability distribution")
    return p


def measures(p, y, mass, q):
    c = p.shape[1]
    if y is None:
        joint = mass[:, None] * np.column_stack((1 - q, q))
        loss = float(mass @ np.square(p).sum(1) - 2 * np.sum(joint * p) + 1)
    else:
        joint = np.eye(c)[y] / len(y)
        loss = float(np.square(p - np.eye(c)[y]).sum(1).mean())
    decision = p.argmax(1)
    conf = np.stack([joint[decision == k].sum(0) for k in range(c)])
    denom = conf.sum(0) + conf.sum(1)
    macro = float(
        np.divide(2 * np.diag(conf), denom, out=np.zeros(c), where=denom > 0).mean()
    )
    return {"brier": loss, "macro_f1": macro}


def measure(job):
    r, b = job
    start = perf_counter()
    methods = methods_for_round(r)
    with threadpool_limits(limits=1):
        fitted, ev, n_fits = fit_on_source(
            b.x, b.y, b.idx, b.env, b.ctx, b.seed, methods
        )
        train_seconds = perf_counter() - start
        common = {
            "round": r,
            "cohort": b.cohort,
            "family": b.family,
            "seed": b.seed,
            "n_fit": len(b.x),
            "n_screen": len(b.idx),
            "true_shortcut": b.shortcut,
            "true_specialist": b.specialist,
            "source_x_hash": digest(b.x),
            "source_y_hash": digest(b.y),
            "environment_hash": digest(b.env),
            "context_hash": digest(b.ctx),
            "screen_indices_hash": digest(b.idx),
        }
        rows = []
        # The 16-row exact support is common to shifted worlds on the same bank.
        predictions = {}
        for world, x, y, mass, q in b.worlds:
            key = digest(x)
            for method, (model, keep) in fitted.items():
                idx = (id(model), key)
                if idx not in predictions:
                    fp = probabilities(model, keep, b.x)
                    tree = getattr(model, "trees_", {}).get(
                        getattr(model, "chosen_", None)
                    )
                    used = (
                        len(tree.unique_value_node_ids)
                        if tree is not None
                        else getattr(model, "n_selected_", len(keep))
                    )
                    predictions[idx] = (probabilities(model, keep, x), digest(fp), used)
                p, source_hash, count = predictions[idx]
                rows.append(
                    {
                        **common,
                        "world": world,
                        "method": method,
                        **measures(p, y, mass, q),
                        "providers": count,
                        "test_hash": digest(p),
                        "fit_hash": source_hash,
                        "test_x_hash": key,
                        "target_hash": digest(
                            y if y is not None else np.column_stack((mass, q))
                        ),
                        "drop": ev.get(method, {}).get("drop"),
                        "reason": ev.get(method, {}).get("reason"),
                    }
                )
        trace = {
            **common,
            "train_seconds": train_seconds,
            "native_fits": n_fits,
            "evidence": ev,
        }
        return rows, trace


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("round", type=int, choices=range(1, 11))
    p.add_argument("--workers", type=int, choices=range(1, 5), default=4)
    args = p.parse_args()
    r = args.round
    if r == 10:
        from summarize import committed_selection

        committed_selection()
    start = perf_counter()
    data = []
    traces = []
    OUT.mkdir(exist_ok=True)
    with multiprocessing.get_context("fork").Pool(args.workers) as pool:
        for rows, trace in pool.imap(
            measure, ((r, b) for b in sources(r)), chunksize=2
        ):
            data.extend(rows)
            traces.append(trace)
    stem = f"r{r:02}"
    pd.DataFrame(data).to_csv(OUT / f"{stem}.csv", index=False)
    with (
        (OUT / f"{stem}-source.jsonl.gz").open("wb") as f,
        gzip.GzipFile(fileobj=f, mode="wb", mtime=0) as g,
    ):
        for trace in traces:
            g.write((json.dumps(trace, allow_nan=False) + "\n").encode())
    meta = {
        "round": r,
        "source_banks": len(traces),
        "target_conditions": len(data) // (len(methods_for_round(r)) + len(REFS)),
        "score_rows": len(data),
        "native_fits": sum(t["native_fits"] for t in traces),
        "seconds": perf_counter() - start,
        "workers": args.workers,
        "fit_worker_seconds": sum(t["train_seconds"] for t in traces),
    }
    (OUT / f"{stem}-time.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(meta, flush=True)


def methods_for_round(r):
    return CANDIDATES if r > 1 else CANDIDATES[:-1]


if __name__ == "__main__":
    main()

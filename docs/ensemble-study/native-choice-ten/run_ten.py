"""Run one preregistered round, fitting every choice before scoring its test bank."""

import argparse
import gzip
import json
import multiprocessing
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from banks_ten import banks, score
from threadpoolctl import threadpool_limits

from fusionbench.methods.averaging import SimpleAverage, TopN
from fusionbench.methods.native_pattern import RegularizedNativePatternOKAPI
from fusionbench.methods.stacking import StackingLR

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
ALL = ("cv", "one_se", "paired", "sign", "stress", "augmented")
ACTIVE = {
    1: ("apparent",),
    2: ("cv",),
    3: ("cv", "one_se"),
    4: ("cv", "paired", "sign"),
    5: ("cv", "paired", "stress"),
    6: ("augmented",),
}
KEYS = ("native", "mean", "top_n", "lr")


def digest(p):
    return sha256(np.ascontiguousarray(p).tobytes()).hexdigest()


def forecast(model, x):
    p = np.asarray(model.predict(x), dtype=np.float64)
    if (
        p.shape != (len(x), x.shape[-1])
        or not np.isfinite(p).all()
        or np.min(p) < -1e-7
        or not np.allclose(p.sum(1), 1, rtol=0, atol=1e-6)
    ):
        raise ValueError("Model produced invalid probabilities")
    return p


def source_fit(x, y, groups, seed, methods):
    options = {
        "task": "binary" if x.shape[-1] == 2 else "multiclass",
        "n_classes": x.shape[-1],
    }
    models = {
        "native": RegularizedNativePatternOKAPI(
            seed=seed, split_seed=0, gp_validation_top_k=8
        ),
        "mean": SimpleAverage(),
        "top_n": TopN(metric="brier"),
        "lr": StackingLR(c_value=1.0),
    }
    for model in models.values():
        model.fit(x, y, **options)
    oof = [m for m in methods if m not in ("apparent", "augmented")]
    decisions, evidence = {}, {}
    if oof:
        from routing import routes, source_evidence

        evidence = source_evidence(x, y, groups, seed=seed)
        all_choices = routes(evidence)
        decisions.update({name: all_choices[name] for name in oof})
    if "apparent" in methods:
        truth = np.eye(x.shape[-1])[y]
        losses = {
            k: np.square(forecast(models[k], x) - truth).sum(1).mean()
            for k in ("mean", "top_n", "native")
        }
        decisions["apparent"] = min(losses, key=losses.get)
    if "augmented" in methods:
        from native_options import fit_augmented

        model = fit_augmented(x, y, seed=seed)
        models["augmented"] = model
        decisions["augmented"] = model.action_
    return models, decisions, evidence


def measure(job):
    r, b = job
    start = perf_counter()
    methods = ACTIVE.get(r, ALL)
    with threadpool_limits(limits=1):
        models, decisions, evidence = source_fit(
            b.x, b.y, b.groups, b.optimizer_seed, methods
        )
        train_seconds = perf_counter() - start
        truth = np.eye(b.x.shape[-1])[b.y]
        rows = {}
        common = {
            "round": r,
            "cohort": b.cohort,
            "family": b.family,
            "seed": b.seed,
            "variant": b.variant,
            "optimizer_seed": b.optimizer_seed,
            "n_fit": len(b.y),
            "n_test": len(b.audit_x),
            "source_x_hash": digest(b.x),
            "source_y_hash": digest(b.y),
            "test_x_hash": digest(b.audit_x),
            "target_hash": digest(b.audit_y)
            if b.audit_y is not None
            else digest(np.column_stack((b.mass, b.q))),
        }
        for name, model in models.items():
            p, fit = forecast(model, b.audit_x), forecast(model, b.x)
            tree = getattr(model, "best_tree", None)
            if tree is None and hasattr(model, "chosen_"):
                tree = model.trees_[model.chosen_]
            providers = (
                len(tree.unique_value_node_ids)
                if tree is not None
                else getattr(model, "n_selected_", b.x.shape[1])
            )
            rows[name] = dict(
                **common,
                method=name,
                action=decisions.get(name, name),
                **score(b, p),
                fit_brier=np.square(fit - truth).sum(1).mean(),
                fit_hash=digest(fit),
                test_hash=digest(p),
                providers=providers,
                nodes=tree.nodes_count if tree is not None else None,
            )
        if "augmented" in models:
            model = models["augmented"]
            for x, field in ((b.x, "fit_hash"), (b.audit_x, "test_hash")):
                assert (
                    digest(np.asarray(model.predict_incumbent(x), dtype=np.float64))
                    == rows["native"][field]
                )
            # Native arithmetic trees and the NumPy baselines may sum in different orders.
            a = decisions["augmented"]
            assert abs(rows["augmented"]["brier"] - rows[a]["brier"]) < 1e-6
        for name, action in decisions.items():
            if name not in rows:
                rows[name] = {**rows[action], "method": name, "action": action}
        source = {
            **{
                k: common[k]
                for k in (
                    "round",
                    "cohort",
                    "family",
                    "seed",
                    "variant",
                    "optimizer_seed",
                )
            },
            "decisions": decisions,
            "native_component": models["native"].chosen_,
            "evidence": evidence,
            "train_seconds": train_seconds,
        }
        return list(rows.values()), source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("round", type=int, choices=range(1, 11))
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=3)
    args = parser.parse_args()
    r = args.round
    if r == 10:
        from summarize import committed_selection

        committed_selection()
    start = perf_counter()
    OUT.mkdir(exist_ok=True)
    rows, traces = [], []
    with multiprocessing.get_context("fork").Pool(args.workers) as pool:
        for batch, trace in pool.imap(measure, ((r, b) for b in banks(r)), chunksize=2):
            rows.extend(batch)
            traces.append(trace)
    df = pd.DataFrame(rows)
    if df.duplicated(["variant", "cohort", "family", "seed", "method"]).any():
        raise ValueError("Duplicate bank record")
    stem = f"r{r:02d}"
    df.to_csv(OUT / f"{stem}.csv", index=False)
    with (
        (OUT / f"{stem}-source.jsonl.gz").open("wb") as dest,
        gzip.GzipFile(fileobj=dest, mode="wb", mtime=0) as stream,
    ):
        for trace in traces:
            stream.write((json.dumps(trace, allow_nan=False) + "\n").encode())
    time = {
        "round": r,
        "bank_conditions": len(traces),
        "score_rows": len(rows),
        "workers": args.workers,
        "elapsed_seconds": perf_counter() - start,
        "fit_worker_seconds": sum(t["train_seconds"] for t in traces),
    }
    (OUT / f"{stem}-time.json").write_text(json.dumps(time, indent=2) + "\n")
    print(time, flush=True)


if __name__ == "__main__":
    main()

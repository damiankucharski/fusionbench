"""Fit all candidates for comparison; select deployable configurations using source inputs only."""

import argparse
import json
import multiprocessing
import subprocess
import sys
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import pandas as pd
from banks_auto import banks
from okapi.configuration import recommend
from okapi.pool_diagnostics import pool_fingerprint
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
sys.path.append(str(HERE.parent / "mechanism-four"))
from run_four import score_models

ACTION_NAMES = ("mean", "top_n", "cal_mean", "flat", "cal_pull")
OTHER = ("lr", "full", "previous8")


def source_decisions(x, y, groups, policy=None):
    from portfolio import cv_choice

    f = pool_fingerprint(x, y, groups=groups)
    if policy is None:
        return f, {}, {}
    t0 = perf_counter()
    choice = recommend(f, policy["policy"])
    tree_seconds = perf_counter() - t0
    cv, info = cv_choice(x, y, groups=groups)
    decisions = {
        "tree_rule": choice["action"],
        "cv_rule": cv,
        "constant": policy["constant"],
    }
    return f, decisions, {"tree": choice, "tree_seconds": tree_seconds, "cv": info}


def measure(job):
    r, b, policy = job
    from portfolio import fit_actions

    with threadpool_limits(limits=1):
        # Candidate training only receives the common source data. The branch choices
        # below are frozen before any call to score_models can read an audit outcome.
        start = perf_counter()
        f, decisions, metadata = source_decisions(b.x, b.y, b.groups, policy)
        models = fit_actions(b.x, b.y, groups=b.groups, actions=ACTION_NAMES + OTHER)
        fit_seconds = perf_counter() - start
        rows = score_models(r, b, models)
    raw = {row["method"]: row for row in rows}
    for row in rows:
        row["action"] = row["method"]
    if policy is not None:
        # A retrospective bound; never supplied to a fitted rule.
        decisions["portfolio_oracle"] = min(
            ACTION_NAMES,
            key=lambda name: (raw[name]["brier"], ACTION_NAMES.index(name)),
        )
        for name, action in decisions.items():
            rows.append({**raw[action], "method": name, "action": action})
    frow = {
        k: b.context[k]
        for k in ("cohort", "family", "seed", "fit_x_sha256", "fit_y_sha256")
    }
    frow.update(round=r, **f)
    trace = {
        "cohort": b.context["cohort"],
        "family": b.context["family"],
        "seed": b.context["seed"],
        "fit_seconds": fit_seconds,
        **metadata,
    }
    return rows, frow, trace


def read_policy():
    path = OUT / "r1-policy.json"
    root = Path(
        subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], cwd=HERE, text=True
        ).strip()
    )
    blob = subprocess.check_output(
        ["git", "show", f"HEAD:{path.relative_to(root)}"], cwd=root
    )
    if path.read_bytes() != blob:
        raise RuntimeError("The training-bank policy differs from the committed policy")
    return json.loads(blob)


def json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(type(obj).__name__)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("round", type=int, choices=(1, 2, 3))
    p.add_argument("--workers", type=int, choices=(1, 2, 3), default=2)
    opts = p.parse_args()
    r = opts.round
    policy = None if r == 1 else read_policy()
    locked_rule = None
    if r == 3:
        from decide_auto import locked

        locked_rule = locked()["chosen"]
    rows, fps, traces = [], [], []
    start = perf_counter()
    # Materialize banks lazily, grouped within the producer as in the generator.
    with multiprocessing.get_context("fork").Pool(opts.workers) as pool:
        for batch, fp, trace in pool.imap(
            measure,
            ((r, SimpleNamespace(**vars(b)), policy) for b in banks(r)),
            chunksize=2,
        ):
            rows.extend(batch)
            fps.append(fp)
            traces.append(trace)
            if len(fps) % 30 == 0:
                print(f"round {r}: {len(fps)} bank conditions", flush=True)
    assert len(fps) == (216, 186, 404)[r - 1]
    OUT.mkdir(exist_ok=True)
    table = pd.DataFrame(rows)
    assert not table.duplicated(["cohort", "family", "seed", "method"]).any()
    table.to_csv(OUT / f"r{r}-scores.csv", index=False)
    pd.DataFrame(fps).to_csv(OUT / f"r{r}-fingerprints.csv", index=False)
    with (OUT / f"r{r}-decisions.jsonl").open("w") as out:
        for trace in traces:
            out.write(json.dumps(trace, default=json_default, allow_nan=False) + "\n")
    counts = {
        "round": r,
        "banks": len(fps),
        "score_rows": len(table),
        "seconds": perf_counter() - start,
        "workers": opts.workers,
        "threads_per_worker": 1,
        "locked_rule": locked_rule,
    }
    if policy is not None:
        counts["r1_policy_sha256"] = sha256(
            (OUT / "r1-policy.json").read_bytes()
        ).hexdigest()
    (OUT / f"r{r}-time.json").write_text(json.dumps(counts, indent=2) + "\n")


if __name__ == "__main__":
    main()

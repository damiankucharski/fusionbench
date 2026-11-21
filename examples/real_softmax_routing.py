"""Time exact winner-only softmax routing against a fused arithmetic pool on one CPU thread."""

import argparse
import json
import platform
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_info, threadpool_limits

from okapi.softmax_routing import compile_softmax_router
from real_sigmoid_pool import Data
from real_softmax_certificate import brier, datasets, row_weights

CONDITIONS = ("disjoint", "overlap", "near_duplicate", "duplicate")
RANKS = (0, 1, 2, 4, 8, None)


class ArithmeticPool:
    def __init__(self, w, b, alpha):
        self.dim = w.shape[2]
        self.k, self.c = b.shape
        self.projection = np.ascontiguousarray(w.reshape(-1, self.dim).T)
        self.bias = np.ascontiguousarray(b.ravel())
        self.alpha = alpha
        self.wmean = np.ascontiguousarray(np.einsum("k,kcd->dc", alpha, w))
        self.bmean = alpha @ b

    def logits(self, x):
        return (x @ self.projection + self.bias).reshape(len(x), self.k, self.c)

    def pool_prob(self, logits):
        prob = np.exp(logits - logits.max(axis=2, keepdims=True))
        prob /= prob.sum(axis=2, keepdims=True)
        return np.einsum("nkc,k->nc", prob, self.alpha)

    def winner(self, x):
        return self.pool_prob(self.logits(x)).argmax(1)

    def unanimous(self, x):
        logits = self.logits(x)
        decisions = logits.argmax(2)
        result = decisions[:, 0].copy()
        disputed = np.any(decisions != result[:, None], axis=1)
        if disputed.all():
            return self.pool_prob(logits).argmax(1)
        if disputed.any():
            result[disputed] = self.pool_prob(logits[disputed]).argmax(1)
        return result

    def approximate(self, x):
        return (x @ self.wmean + self.bmean).argmax(1)


def routed(x, pool, router):
    winner, tv, agree = router.certified(x)
    accepted = tv | agree
    if not accepted.any():
        return pool.winner(x)
    if not accepted.all():
        winner = winner.copy()
        winner[~accepted] = pool.winner(x[~accepted])
    return winner


def fixed_member(data, scaler, z, columns, c):
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model = LogisticRegression(C=c, max_iter=4000, tol=1e-7).fit(
            z[data.train][:, columns], data.y[data.train]
        )
    w = np.zeros((len(model.classes_), z.shape[1]))
    w[:, columns] = model.coef_ / scaler.scale_[columns]
    return w, model.intercept_ - w @ scaler.mean_


def new_digit_split(original):
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=3101)
    folds = [idx for _, idx in cv.split(original.x, original.y, original.groups)]
    return Data(
        original.name,
        original.x,
        original.y,
        original.groups,
        np.concatenate(folds[2:]),
        folds[1],
        folds[0],
    )


def measure(x, methods, seed):
    rng = np.random.default_rng(seed)
    n = min(128, len(x))
    ix = np.sort(rng.choice(len(x), size=n, replace=False))
    sampled = x[ix]
    workloads = {
        "single": [sampled[i : i + 1] for i in range(n)],
        "batch64": [x[i : i + 64] for i in range(0, len(x), 64)],
        "whole": [x],
    }
    rows = []
    for schedule, chunks in workloads.items():
        order = list(methods)
        sample_count = sum(map(len, chunks))

        def process(fun):
            # Every method returns one contiguous vector for each workload.
            return fun(chunks[0]) if len(chunks) == 1 else np.concatenate([fun(part) for part in chunks])

        target = process(methods["full_pool"])
        for r in range(-2, 11):
            rng.shuffle(order)
            for name in order:
                t0 = time.perf_counter_ns()
                y = process(methods[name])
                micros = (time.perf_counter_ns() - t0) / 1e3
                if name != "compact_unverified" and not np.array_equal(y, target):
                    raise AssertionError(f"{name} changed winner in timed {schedule} workload")
                if r >= 0:
                    rows.append(
                        dict(
                            schedule=schedule,
                            method=name,
                            repeat=r,
                            sample_count=sample_count,
                            us_total=micros,
                            us_per_example=micros / sample_count,
                        )
                    )
    return rows


def frozen_pools(data, oldfits):
    scaler = StandardScaler().fit(data.x[data.train])
    z = scaler.transform(data.x)
    idx = np.arange(z.shape[1])
    full_c = float(oldfits.full_C.iloc[0])
    full_w, full_b = fixed_member(data, scaler, z, idx, full_c)
    nonconstant = idx[scaler.var_ > 0]
    omit = nonconstant[np.linspace(0, len(nonconstant) - 1, 5, dtype=int)]
    for cond in CONDITIONS:
        row = oldfits.loc[oldfits.condition == cond].iloc[0]
        if cond == "duplicate":
            members = [(full_w, full_b)] * 5
        else:
            members = []
            for j, c in enumerate(json.loads(row.member_C)):
                mask = (
                    idx != omit[j]
                    if cond == "near_duplicate"
                    else (idx % 5 == j if cond == "disjoint" else idx % 5 != j)
                )
                members.append(fixed_member(data, scaler, z, idx[mask], c))
        w, b = [np.stack(part) for part in zip(*members)]
        yield cond, w, b, full_w, full_b, int(row.selected_member), scaler.mean_, scaler.scale_


def softmax(x):
    p = np.exp(x - x.max(1, keepdims=True))
    return p / p.sum(1, keepdims=True)


def run_case(data, cond, w, b, full_w, full_b, selected, center, scale, old, seed):
    x = np.ascontiguousarray(data.x[data.test])
    weights = row_weights(data, data.test)
    alpha = np.full(len(w), 1 / len(w))
    pool = ArithmeticPool(w, b, alpha)
    lp = pool.logits(x)
    p = pool.pool_prob(lp)
    truth = p.argmax(1)
    q = softmax(x @ pool.wmean + pool.bmean)
    local_ids = lp.argmax(2)
    y = data.y[data.test]
    reference = dict(
        mean_tv=weights @ (0.5 * np.abs(p - q).sum(1)),
        top1_agreement=weights @ (truth == q.argmax(1)),
        pool_brier=weights @ brier(p, y),
        compiled_brier=weights @ brier(q, y),
        best_member_brier=weights @ brier(softmax(lp[:, selected]), y),
        full_logistic_brier=weights @ brier(softmax(x @ full_w.T + full_b), y),
    )
    reproduction = (
        [dict(metric=k, actual=v, published=old[k], difference=v - old[k]) for k, v in reference.items()]
        if old is not None
        else []
    )
    methods = {
        "full_pool": pool.winner,
        "all_logits_agreement": pool.unanimous,
        "compact_unverified": pool.approximate,
    }
    if cond == "duplicate":
        dw, db = w[0].T.copy(), b[0].copy()
        methods["deduplicated_exact"] = lambda samples: (samples @ dw + db).argmax(1)
    coverage = []
    for rank in RANKS:
        rankname = "full" if rank is None else str(rank)
        router = compile_softmax_router(w, b, alpha, center=center, scale=scale, rank=rank)
        name = "route_" + rankname
        methods[name] = lambda samples, gate=router: routed(samples, pool, gate)
        ids, tv, agree = router.certified(x)
        accepted = tv | agree
        if np.any((ids != truth) & accepted):
            raise AssertionError("An accepted example changed the true arithmetic-pool decision")
        cover = {}
        for key, mask in [("accepted", accepted), ("tv", tv), ("agreement", agree)]:
            cover[key + "_rows"] = mask.mean()
            cover[key + "_metric_weighted"] = weights @ mask
        coverage.append(
            dict(
                rank=rankname,
                **cover,
                full_unanimity_rows=np.all(local_ids == local_ids[:, :1], axis=1).mean(),
                compact_unverified_disagreement_rows=(q.argmax(1) != truth).mean(),
            )
        )
    return measure(x, methods, seed), coverage, reproduction


def verify_ties():
    x = np.array([[0.0], [1.0], [-1.0]])
    # All classes tied, two dominant classes tied, and opposite votes summing to an exact tie.
    for biases in [
        np.zeros((2, 3)),
        np.tile([0.0, 1.0, 1.0], (2, 1)),
        np.array([[1.0, 0.0, -1.0], [0.0, 1.0, -1.0]]),
    ]:
        w = np.zeros((2, 3, 1))
        a = np.array([0.5, 0.5])
        direct = ArithmeticPool(w, biases, a)
        target = direct.winner(x)
        np.testing.assert_array_equal(
            target, np.full(len(x), 1 if biases[0, 1] == biases[1, 1] == 1 else 0)
        )
        np.testing.assert_array_equal(direct.unanimous(x), target)
        for rank in RANKS:
            gate = compile_softmax_router(w, biases, a, center=np.zeros(1), scale=np.ones(1), rank=rank)
            np.testing.assert_array_equal(routed(x, direct, gate), target)


def run(har, previous, out, export=False):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    previous = Path(previous)
    oldfits = pd.read_csv(previous / "fits.csv")
    oldmetrics = pd.read_csv(previous / "results.csv")
    start = time.perf_counter()
    began = datetime.now(timezone.utc).isoformat()
    verify_ties()
    runs, coverage, reproduction = [], [], []
    digit, activity = datasets(har)
    for data, split in [
        (digit, "digits_3031"),
        (activity, "har_official"),
        (new_digit_split(digit), "digits_3101"),
    ]:
        for j, (cond, w, b, fw, fb, selected, center, scale) in enumerate(
            frozen_pools(data, oldfits.query("dataset==@data.name"))
        ):
            key = dict(split=split, condition=cond)
            if export:
                # Input units are raw. Center and scale describe the target routing geometry.
                np.savez_compressed(
                    out / f"experts_{split}_{cond}.npz",
                    weights=w,
                    intercepts=b,
                    mixing_weights=np.full(5, 0.2),
                    center=center,
                    scale=scale,
                    test_indices=data.test,
                )
            ref = (
                oldmetrics.loc[(oldmetrics.dataset == data.name) & (oldmetrics.condition == cond)].iloc[
                    0
                ]
                if split != "digits_3101"
                else None
            )
            timed, cov, checks = run_case(
                data, cond, w, b, fw, fb, selected, center, scale, ref, 31010 + j
            )
            runs.extend(dict(**key, **r) for r in timed)
            coverage.extend(dict(**key, **r) for r in cov)
            reproduction.extend(dict(**key, **r) for r in checks)
            # Durable checkpoints keep completed grids available during long timing runs.
            pd.DataFrame(runs).to_csv(
                out / "timing_draws.csv.gz", index=False, compression={"method": "gzip", "mtime": 0}
            )
            pd.DataFrame(coverage).to_csv(out / "coverage.csv", index=False)
            pd.DataFrame(reproduction).to_csv(out / "reproduction.csv", index=False)
    trials = pd.DataFrame(runs)
    group = trials.groupby(["split", "condition", "schedule", "method"]).us_per_example
    summary = group.agg(
        median="median", q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75)
    ).reset_index()
    baseline = summary.query('method=="full_pool"')[["split", "condition", "schedule", "median"]].rename(
        columns={"median": "full_pool_median"}
    )
    summary = summary.merge(baseline, on=["split", "condition", "schedule"])
    summary["speedup_vs_full"] = summary.full_pool_median / summary["median"]
    summary.to_csv(out / "runtime.csv", index=False)
    (out / "config.json").write_text(
        json.dumps(
            dict(
                compute_seconds=time.perf_counter() - start,
                compute_started_utc=began,
                compute_ended_utc=datetime.now(timezone.utc).isoformat(),
                numpy=np.__version__,
                sklearn=sklearn.__version__,
                cpu_arch=platform.machine(),
                threads=threadpool_info(),
                repeats=11,
                warmups=2,
                tie_checks=True,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--har", required=True)
    parser.add_argument("--previous", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--export-experts", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.har, args.previous, args.out, args.export_experts)

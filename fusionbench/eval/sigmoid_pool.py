"""Evaluate whether uncertainty-weighted affine sigmoids collapse into a single raw classifier."""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, softmax
from scipy.stats import t
from threadpoolctl import threadpool_limits

from fusionbench.eval.mixture_sentinels import fit_bank, probabilities_given_labels
from fusionbench.eval.weak_mixtures import Batch, draw, parameters, standardize
from okapi.sigmoid_pool import collapse_sigmoids


def fit_logistic(x, targets, reg):
    augmented = np.column_stack((x, np.ones(len(x))))

    theta = np.zeros(augmented.shape[1])

    def loss(theta):
        logits = augmented @ theta
        return (
            targets @ np.logaddexp(0, -logits)
            + (1 - targets) @ np.logaddexp(0, logits)
            + reg * (theta @ theta) / 2
        )

    for _ in range(100):
        p = expit(augmented @ theta)
        grad = augmented.T @ (p - targets) + reg * theta
        hessian = augmented.T @ ((p * (1 - p))[:, None] * augmented) + reg * np.eye(len(theta))
        step = np.linalg.solve(hessian, grad)
        decrement = grad @ step
        if decrement <= 2e-12:
            return theta[:-1], float(theta[-1])
        previous = loss(theta)
        scale = 1.0
        while loss(theta - scale * step) > previous - 0.01 * scale * decrement:
            scale /= 2
            if scale < 2**-30:
                raise RuntimeError("Newton line search failed")
        theta -= scale * step
    raise RuntimeError("Regularized logistic fit exceeded 100 Newton iterations")


def compare(w, b, alpha, train, test, posterior, audit, labels):
    center = train.x.mean(0)
    covariance = train.second - np.outer(center, center)
    ensemble = alpha @ expit(w @ test.x.T + b[:, None])
    records = {
        "pool": dict(excess_brier=np.mean((ensemble - posterior) ** 2), pool_rmse=0.0, build_ms=np.nan)
    }
    rules = {}
    for name in ["tangent", "secant", "average_weights", "top_weight", "direct_labels", "distilled"]:
        t0 = time.perf_counter()
        if name == "tangent":
            rule = collapse_sigmoids(w, b, alpha, center)
        elif name == "secant":
            rule = collapse_sigmoids(w, b, alpha, center, covariance=covariance)
        elif name == "average_weights":
            rule = (alpha @ w, float(alpha @ b))
        elif name == "top_weight":
            idx = alpha.argmax()
            rule = (w[idx], b[idx])
        elif name == "direct_labels":
            rule = fit_logistic(audit, labels, 1.0)
        else:
            targets = alpha @ expit(w @ train.x.T + b[:, None])
            rule = fit_logistic(train.x, targets, 0.0001)
        elapsed = time.perf_counter() - t0
        pred = expit(test.x @ rule[0] + rule[1])
        records[name] = dict(
            excess_brier=np.mean((pred - posterior) ** 2),
            pool_rmse=np.sqrt(np.mean((pred - ensemble) ** 2)),
            build_ms=1e3 * elapsed,
        )
        rules[name] = rule
    return records, rules


def benchmark(w, b, weights, rule, inputs):
    times = {}
    nonconstant = (w != 0).any(axis=1)
    fixed = float(weights[~nonconstant] @ expit(b[~nonconstant]))
    varying_w, varying_b, alpha = w[nonconstant], b[nonconstant], weights[nonconstant]
    for name, fun in [
        ("dense_pool", lambda: weights @ expit(w @ inputs.T + b[:, None])),
        (
            "grouped_constants_pool",
            lambda: fixed + alpha @ expit(varying_w @ inputs.T + varying_b[:, None]),
        ),
        ("single", lambda: expit(inputs @ rule[0] + rule[1])),
    ]:
        reference = fun()
        if name == "dense_pool":
            full = reference
        if name == "grouped_constants_pool":
            np.testing.assert_allclose(reference, full, atol=1e-12)
        ticks = []
        for _ in range(17):
            t0 = time.perf_counter_ns()
            fun()
            ticks.append((time.perf_counter_ns() - t0) * 1e-6)
        times[name] = np.median(ticks)
    return dict(
        batch=len(inputs), repeats=17, nonconstant_components=int(nonconstant.sum()), median_ms=times
    )


def run(out):
    tic = time.perf_counter()
    start = datetime.now(timezone.utc).isoformat()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    records, exports = [], []
    for phase, seeds in [("development", range(2600, 2604)), ("confirmation", range(2700, 2716))]:
        for seed in seeds:
            for condition in ("weak", "pair", "zero_signal", "plateau"):
                if condition == "plateau":
                    data = [np.random.default_rng([seed, n]).normal(size=(n, 1)) for n in (1024, 8192)]
                    train, test = map(Batch.make, data)
                    mm = np.zeros(1)
                    sd = np.ones(1)
                    w = np.array([[6.0], [6.0]])
                    b = np.array([-6.0, 6.0])
                    pooled = np.array([0.5, 0.5])
                    q = pooled @ expit(w @ test.x.T + b[:, None])
                else:
                    truth = parameters(condition)
                    raw = draw(seed, 0, 1024, truth)
                    mm, sd = raw.mean(0), raw.std(0)
                    train = Batch.make((raw - mm) / sd)
                    valid = Batch.make((draw(seed, 1, 1024, truth) - mm) / sd)
                    test = Batch.make((draw(seed, 2, 8192, truth) - mm) / sd)
                    q, _ = standardize(truth, mm, sd).evaluate(test)
                    bank = fit_bank(train, valid, seed)
                    density = np.log(bank.prior_mass) + 1024 * (bank.val_ll - bank.val_ll.max())
                    w, b = bank.coefficients, bank.intercepts
                for rep in range(1 if condition == "plateau" else 4):
                    if condition == "plateau":
                        audit = np.random.default_rng([seed, 77]).normal(size=(64, 1))
                        audit_q = pooled @ expit(w @ audit.T + b[:, None])
                    else:
                        audit = (draw(seed, 100 + rep, 64, truth) - mm) / sd
                        audit_q, _ = standardize(truth, mm, sd).evaluate(Batch.make(audit))
                    labels = (np.random.default_rng([seed, 80, rep]).uniform(size=64) < audit_q).astype(
                        int
                    )
                    for n in [16] if condition == "plateau" else [0, 16, 64]:
                        if condition == "plateau":
                            alpha = pooled
                        else:
                            label_ll = probabilities_given_labels(
                                w @ audit[:n].T + b[:, None], labels[:n]
                            )
                            alpha = softmax(density + label_ll)
                        results, rules = compare(w, b, alpha, train, test, q, audit[:n], labels[:n])
                        keys = dict(
                            phase=phase, seed=seed, condition=condition, budget=n, repetition=rep
                        )
                        records.extend(
                            dict(**keys, method=name, **metrics) for name, metrics in results.items()
                        )
                        if n == 16 and rep == 0:
                            for name in ["tangent", "secant"]:
                                weights, bias = rules[name]
                                raw_weights = weights / sd
                                raw_bias = bias - mm @ raw_weights
                                np.testing.assert_allclose(
                                    expit(test.x @ weights + bias),
                                    expit((test.x * sd + mm) @ raw_weights + raw_bias),
                                    atol=1e-12,
                                )
                                exports.append(
                                    dict(
                                        **keys,
                                        method=name,
                                        weights=raw_weights.tolist(),
                                        intercept=raw_bias,
                                    )
                                )
                        if seed == 2700 and condition == "weak" and n == 16 and rep == 0:
                            latency = benchmark(w, b, alpha, rules["tangent"], test.x[:1024])
    frame = pd.DataFrame(records)
    frame.to_csv(out / "audit_draws.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    bank = (
        frame.groupby(["phase", "seed", "condition", "budget", "method"])[
            ["excess_brier", "pool_rmse", "build_ms"]
        ]
        .mean()
        .reset_index()
    )
    bank.to_csv(out / "banks.csv", index=False)
    bank.groupby(["phase", "condition", "budget", "method"])[
        ["excess_brier", "pool_rmse", "build_ms"]
    ].mean().to_csv(out / "means.csv")
    p = bank.query('phase=="confirmation" and condition=="weak" and budget==16').pivot(
        index="seed", columns="method", values="excess_brier"
    )
    delta = p.tangent - p.pool
    radius = t.ppf(0.975, len(delta) - 1) * delta.sem()
    cfg = dict(
        primary=dict(
            difference=delta.mean(), lo=delta.mean() - radius, hi=delta.mean() + radius, margin=0.002
        ),
        latency=latency,
        compute_seconds=time.perf_counter() - tic,
        compute_started_utc=start,
        compute_ended_utc=datetime.now(timezone.utc).isoformat(),
    )
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    (out / "raw_models.json").write_text(json.dumps(exports, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    with threadpool_limits(limits=1):
        run(parser.parse_args().out)

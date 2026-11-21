"""Evaluate a sigmoid merge hierarchy, its accuracy, and pointwise error certificates."""

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
from okapi.sigmoid_merge import merge_sigmoids


def compare(w, b, a, x, test, q):
    mu = x.mean(0)
    cov = np.cov(x, rowvar=False, ddof=0).reshape(len(mu), len(mu))
    t0 = time.perf_counter()
    widths = (1, 2, 4, 8, 16, len(w))
    compiled = merge_sigmoids(w, b, a, mu, cov, widths=widths)
    elapsed = time.perf_counter() - t0
    original = expit(test @ w.T + b) @ a
    full = compiled[len(w)]
    np.testing.assert_allclose(full.predict(test), original, atol=1e-12)
    source_second = np.outer(np.r_[mu, 1.0], np.r_[mu, 1.0])
    source_second[:-1, :-1] += cov
    means = {k: np.sum(source_second * model.bound_matrix) for k, model in compiled.items()}
    chosen = next(k for k in sorted(means) if means[k] <= 0.01)
    records = []
    predictions = expit(test @ full.weights.T + full.intercepts)
    for name, cap in [*[(str(k), k) for k in sorted(compiled)], ("certificate_choice", chosen)]:
        model = compiled[cap]
        width = len(model.mixing)
        merged = model.predict(test)
        bound = model.error_bound(test)
        error = abs(merged - original)
        violation = float(np.max(error - bound))
        if violation > 1e-10:
            raise AssertionError(f"Violated probability certificate by {violation:g}")
        order = np.argsort(-full.mixing, kind="stable")[:width]
        top = predictions[:, order] @ (full.mixing[order] / full.mixing[order].sum())
        for method, p in [("merge", merged), ("top", top)]:
            discrepancy = np.abs(p - original)
            records.append(
                dict(
                    cap=name,
                    width=width,
                    method=method,
                    excess_brier=np.mean((p - q) ** 2),
                    pool_excess_brier=np.mean((original - q) ** 2),
                    rmse=np.sqrt(np.mean(discrepancy**2)),
                    actual_within_005=np.mean(discrepancy <= 0.05),
                    certified_within_005=np.mean(bound <= 0.05) if method == "merge" else np.nan,
                    mean_bound=np.mean(bound) if method == "merge" else np.nan,
                    train_bound=means[cap] if method == "merge" else np.nan,
                    worst_certificate_violation=violation if method == "merge" else np.nan,
                    hierarchy_ms=elapsed * 1e3 if method == "merge" else np.nan,
                )
            )
    return records


def generate_bank(seed, condition):
    if condition in {"plateau", "eight_axes"}:
        d = 1 if condition == "plateau" else 8
        if d == 1:
            w = np.full((2, 1), 6.0)
            b = np.array([-6.0, 6.0])
        else:
            w = 6 * np.eye(d)
            b = np.linspace(-3, 3, d)
        a = np.full(len(w), 1 / len(w))
        x = np.random.default_rng([seed, 0]).normal(size=(1024, d))
        test = np.random.default_rng([seed, 2]).normal(size=(8192, d))
        q = expit(test @ w.T + b) @ a
        for row in compare(w, b, a, x, test, q):
            yield dict(repetition=0, **row)
        return
    truth = parameters(condition)
    x = draw(seed, 0, 1024, truth)
    mu, sd = x.mean(0), x.std(0)
    train = Batch.make((x - mu) / sd)
    valid = Batch.make((draw(seed, 1, 1024, truth) - mu) / sd)
    test = Batch.make((draw(seed, 2, 8192, truth) - mu) / sd)
    bank = fit_bank(train, valid, seed)
    density = np.log(bank.prior_mass) + 1024 * (bank.val_ll - bank.val_ll.max())
    population = standardize(truth, mu, sd)
    q, _ = population.evaluate(test)
    for rep in range(3):
        audit = Batch.make((draw(seed, 100 + rep, 16, truth) - mu) / sd)
        targets = (
            np.random.default_rng([seed, 80, rep]).random(16) < population.evaluate(audit)[0]
        ).astype(int)
        ll = probabilities_given_labels(bank.logits(audit.x), targets)
        a = softmax(density + ll)
        for row in compare(bank.coefficients, bank.intercepts, a, train.x, test.x, q):
            yield dict(repetition=rep, **row)


def run(out):
    tic = time.perf_counter()
    start = datetime.now(timezone.utc).isoformat()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for phase, seeds in [("development", range(2800, 2804)), ("confirmation", range(2900, 2916))]:
        for seed in seeds:
            for condition in ("weak", "pair", "zero_signal", "plateau", "eight_axes"):
                rows.extend(
                    dict(phase=phase, seed=seed, condition=condition, **r)
                    for r in generate_bank(seed, condition)
                )
    df = pd.DataFrame(rows)
    df.to_csv(out / "draws.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    banks = (
        df.groupby(["phase", "seed", "condition", "cap", "method"])
        .mean(numeric_only=True)
        .drop(columns="repetition")
        .reset_index()
    )
    banks.to_csv(out / "banks.csv", index=False)
    banks.groupby(["phase", "condition", "cap", "method"]).mean(numeric_only=True).drop(
        columns="seed"
    ).to_csv(out / "means.csv")
    p = banks.query('phase=="confirmation" and condition=="weak" and cap=="4" and method=="merge"')
    d = p.excess_brier - p.pool_excess_brier
    rad = t.ppf(0.975, len(d) - 1) * d.sem()
    config = dict(
        primary=dict(difference=d.mean(), lo=d.mean() - rad, hi=d.mean() + rad, margin=0.002),
        largest_observed_violation=float(df.worst_certificate_violation.max()),
        compute_seconds=time.perf_counter() - tic,
        compute_started_utc=start,
        compute_ended_utc=datetime.now(timezone.utc).isoformat(),
    )
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    with threadpool_limits(limits=1):
        run(parser.parse_args().out)

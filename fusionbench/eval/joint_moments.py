"""Recover an affine binary classifier from unlabeled cross-sensor moments."""

import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import t
from sklearn.mixture import GaussianMixture
from threadpoolctl import threadpool_limits

STRENGTHS = np.array([1.05, 0.95, 0.85, 0.75, 0.65, 0.55])
OFFSETS = np.array([0.6, -0.4, 0.8, -0.5, 0.3, 0.1])
SCALES = np.array([0.7, 1.4, 0.9, 1.3, 0.8, 1.1])


@dataclass
class LinearRule:
    weights: list
    intercept: float
    prior: float

    def predict(self, x):
        return expit(x @ self.weights + self.intercept)


def moments_to_rule(cov, third, mean, scale, *, first=False, diagonal=False):
    ratios = []
    contrasts = [[] for _ in range(len(cov))]
    for i, j, k in combinations(range(len(cov)), 3):
        cij, cik, cjk = cov[i, j], cov[i, k], cov[j, k]
        if min(cij, cik, cjk) <= 1e-8:
            continue
        ratios.append(third[i, j, k] / np.sqrt(cij * cik * cjk))
        contrasts[i].append(cij * cik / cjk)
        contrasts[j].append(cij * cjk / cik)
        contrasts[k].append(cik * cjk / cij)

    def aggregate(values):
        if not values:
            return 0.0
        return values[0] if first else np.median(values)

    ratio = aggregate(ratios)
    prior = np.clip((1 - ratio / np.hypot(ratio, 2)) / 2, 0.05, 0.95)
    signal = np.array([aggregate(values) for values in contrasts])
    delta = np.sqrt(np.minimum(signal, 0.95 * np.diag(cov)) / (prior * (1 - prior)))
    residual = cov - prior * (1 - prior) * np.outer(delta, delta)
    if diagonal:
        residual = np.diag(np.diag(residual))
    ev, basis = np.linalg.eigh(residual)
    weights = basis @ ((basis.T @ delta) / np.maximum(ev, 0.05))
    raw_weights = weights / scale
    intercept = logit(prior) - (0.5 - prior) * (delta @ weights) - mean @ raw_weights
    return LinearRule(raw_weights.tolist(), float(intercept), float(prior))


def estimated_rules(x, seed):
    mean, scale = x.mean(axis=0), x.std(axis=0)
    z = (x - mean) / scale
    cov = z.T @ z / len(z)
    third = np.einsum("ni,nj,nk->ijk", z, z, z, optimize=True) / len(z)
    rules = {
        name: moments_to_rule(cov, third, mean, scale, first=first, diagonal=diagonal)
        for name, first, diagonal in [
            ("robust", False, False),
            ("first", True, False),
            ("diagonal", False, True),
        ]
    }
    mixture = GaussianMixture(2, covariance_type="tied", n_init=3, random_state=seed, max_iter=300).fit(
        z
    )
    positive = np.argmax(mixture.means_.sum(axis=1))
    negative = 1 - positive
    delta = mixture.means_[positive] - mixture.means_[negative]
    weights = np.linalg.solve(mixture.covariances_, delta)
    prior = mixture.weights_[positive]
    raw_weights = weights / scale
    mid = (mixture.means_[positive] + mixture.means_[negative]) / 2
    intercept = logit(prior) - mid @ weights - mean @ raw_weights
    rules["gmm"] = LinearRule(raw_weights.tolist(), float(intercept), float(prior))
    np.testing.assert_allclose(
        rules["gmm"].predict(x), mixture.predict_proba(z)[:, positive], atol=1e-10
    )
    return rules


def draw(seed, split, n, condition):
    rng = np.random.default_rng([seed, split])
    p = 0.5 if condition == "iid" else 0.75
    s = STRENGTHS * (0.4 if condition == "weak" else 1)
    covariance = np.eye(len(s))
    if condition in ("pair", "invisible"):
        covariance[0, 1] = covariance[1, 0] = 0.75
    if condition == "dense":
        covariance = 0.5 * np.eye(len(s)) + 0.5
    y = (rng.uniform(size=n) < p).astype(int)
    z = (2 * y - 1)[:, None] * s + rng.normal(size=(n, len(s))) @ np.linalg.cholesky(covariance).T
    q = expit(logit(p) + z @ np.linalg.solve(covariance, 2 * s))
    x = z if condition == "iid" else z * SCALES + OFFSETS
    if condition == "invisible":
        y = y ^ (rng.uniform(size=n) < 0.25)
        q = 0.25 + 0.5 * q
        p = 0.25 + 0.5 * p
    return x, y, q, p


def contrast(frame, baseline, challenger):
    by = frame.pivot(index="seed", columns="method", values="excess_brier")
    d = by[baseline] - by[challenger]
    radius = t.ppf(0.975, len(d) - 1) * d.sem()
    return dict(reduction=d.mean(), lo=d.mean() - radius, hi=d.mean() + radius)


def run_study(out):
    start = time.perf_counter()
    began = datetime.now(timezone.utc).isoformat()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    records, weights = [], []
    for phase, seeds in [("development", range(2000, 2004)), ("confirmation", range(2100, 2116))]:
        for seed in seeds:
            pair_rules = {}
            for condition in ("iid", "independent", "pair", "invisible", "dense", "weak"):
                x = draw(seed, 0, 1024, condition)[0]
                xt, y, q, prior = draw(seed, 1, 8192, condition)
                for n in (128, 1024):
                    if condition == "invisible":
                        rules = pair_rules[n]
                    else:
                        rules = estimated_rules(x[:n], seed)
                        rules["frozen"] = LinearRule((2 * STRENGTHS).tolist(), 0.0, 0.5)
                    if condition == "pair":
                        pair_rules[n] = rules
                    for method, rule in rules.items():
                        pred = rule.predict(xt)
                        keys = dict(phase=phase, seed=seed, condition=condition, budget=n, method=method)
                        records.append(
                            dict(
                                **keys,
                                prior=rule.prior,
                                true_prior=prior,
                                abs_prior_error=abs(rule.prior - prior),
                                excess_brier=np.mean((pred - q) ** 2),
                                accuracy=np.mean((pred >= 0.5) == y),
                            )
                        )
                        if method == "robust" and n == 1024:
                            weights.append(dict(**keys, parameters=asdict(rule)))
    df = pd.DataFrame(records)
    df.to_csv(out / "banks.csv", index=False)
    df.groupby(["phase", "condition", "budget", "method"]).mean(numeric_only=True).drop(
        columns="seed"
    ).to_csv(out / "means.csv")
    (out / "linear_models.json").write_text(json.dumps(weights, indent=2) + "\n")
    primary = df.query('phase=="confirmation" and condition=="pair" and budget==1024')
    cfg = dict(
        primary=contrast(primary, "first", "robust"),
        comparison_to_gmm=contrast(primary, "gmm", "robust"),
        compute_seconds=time.perf_counter() - start,
        compute_started_utc=began,
        compute_ended_utc=datetime.now(timezone.utc).isoformat(),
    )
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    with threadpool_limits(limits=1):
        run_study(parser.parse_args().out)

"""Use tiny random labeled audits to choose ambiguous latent-class mixture predictors."""

import argparse
import json
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit, softmax
from scipy.stats import t
from threadpoolctl import threadpool_limits

from fusionbench.eval.weak_mixtures import (
    CHECKPOINTS,
    Batch,
    Mixture,
    draw,
    initial,
    m_step,
    parameters,
    standardize,
)


@dataclass
class Candidates:
    names: list[str]
    coefficients: np.ndarray
    intercepts: np.ndarray
    val_ll: np.ndarray
    prior_mass: np.ndarray
    constant: np.ndarray

    def logits(self, standardized):
        return self.coefficients @ standardized.T + self.intercepts[:, None]


def fit_bank(train, validation, seed):
    # Every constant candidate fits the same Gaussian marginal.
    d = train.x.shape[1]
    marginal = Mixture(
        0.5,
        np.stack([train.mean, train.mean]),
        train.second - np.outer(train.mean, train.mean) + 1e-6 * np.eye(d),
    )
    _, constant_ll = marginal.evaluate(validation)
    grid = sorted(np.arange(1, 20) / 20, key=lambda p: abs(p - 0.5))
    names = [f"constant_{p:.2f}" for p in grid]
    intercepts = list(logit(grid))
    weights = [np.zeros(d) for _ in grid]
    val_ll = [constant_ll] * len(grid)
    # The imported helper can optionally attach a simulator-oracle diagnostic.
    starts = initial(train, None, seed)
    del starts["oracle_start"]
    for name, state in starts.items():
        for step in range(max(CHECKPOINTS) + 1):
            if step in CHECKPOINTS:
                names.append(f"{name}_{step}")
                w = np.linalg.solve(state.cov, state.means[1] - state.means[0])
                weights.append(w)
                intercepts.append(logit(state.prior) - state.means.mean(0) @ w)
                val_ll.append(state.evaluate(validation)[1])
            if step < max(CHECKPOINTS):
                state = m_step(train, state.evaluate(train)[0])

    n_em = len(starts) * len(CHECKPOINTS)
    prior_mass = np.r_[np.full(len(grid), 0.5 / len(grid)), np.full(n_em, 0.5 / n_em)]
    return Candidates(
        names,
        np.asarray(weights),
        np.asarray(intercepts),
        np.asarray(val_ll),
        prior_mass,
        np.arange(len(names)) < len(grid),
    )


def intercept_adjustment(logits, labels):
    if not len(labels):
        return 0.0

    def derivative(delta):
        return (expit(logits + delta) - labels).sum() + delta / 4

    bound = 4 * len(labels) + 1
    return brentq(derivative, -bound, bound)


def choose_brier(probs, labels, val_ll):
    if not len(labels):
        return int(val_ll.argmax())
    errors = np.mean((probs - labels) ** 2, axis=1)
    tied = np.isclose(errors, errors.min(), rtol=0, atol=1e-14)
    return int(np.argmax(np.where(tied, val_ll, -np.inf)))


def probabilities_given_labels(logits, labels):
    # Conditional log likelihood of each frozen candidate; stable at extreme logits.
    return np.sum(-np.logaddexp(0, -(2 * labels - 1) * logits), axis=1)


def evaluate_sentinels(bank, val_n, test_x, posterior, audits):
    test_logits = bank.logits(test_x)
    predictions = expit(test_logits)
    differences = predictions - posterior
    gram = differences @ differences.T / len(posterior)
    candidate_risk = np.diag(gram)
    candidate_means = predictions.mean(axis=1)
    prior_log_weights = np.log(bank.prior_mass)
    density_log_weights = prior_log_weights + val_n * (bank.val_ll - bank.val_ll.max())
    density_weights = softmax(density_log_weights)
    winner = bank.val_ll.argmax()
    hindsight = candidate_risk.argmin()
    results = []

    def ensemble(name, weights):
        return dict(
            method=name,
            excess_brier=weights @ gram @ weights,
            predicted_mean=weights @ candidate_means,
            constant_mass=weights[bank.constant].sum(),
            effective_count=1 / (weights @ weights),
        )

    def selected(name, index):
        return dict(
            method=name,
            excess_brier=candidate_risk[index],
            predicted_mean=candidate_means[index],
            constant_mass=float(bank.constant[index]),
            effective_count=1.0,
        )

    static = [
        ensemble("density_pool", density_weights),
        selected("density_select", winner),
        selected("hindsight", hindsight),
    ]
    for repetition, (x, y) in enumerate(audits):
        audit_logits = bank.logits(x)
        for budget in (0, 4, 16, 64):
            logits, labels = audit_logits[:, :budget], y[:budget]
            label_ll = probabilities_given_labels(logits, labels)
            idx = choose_brier(expit(logits), labels, bank.val_ll)
            adjusted = expit(test_logits[winner] + intercept_adjustment(logits[winner], labels))
            constant = (labels.sum() + 1) / (budget + 2)
            rows = [
                *static,
                ensemble("density_plus_labels", softmax(density_log_weights + label_ll)),
                ensemble("labels_pool", softmax(prior_log_weights + label_ll)),
                selected("label_brier", idx),
                dict(
                    method="calibrated_density",
                    excess_brier=np.mean((adjusted - posterior) ** 2),
                    predicted_mean=adjusted.mean(),
                    constant_mass=float(bank.constant[winner]),
                    effective_count=1.0,
                ),
                dict(
                    method="label_constant",
                    excess_brier=np.mean((constant - posterior) ** 2),
                    predicted_mean=constant,
                    constant_mass=1.0,
                    effective_count=1.0,
                ),
            ]
            results.extend(dict(repetition=repetition, budget=budget, **row) for row in rows)
    return results


def bank_study(seed, condition):
    truth = parameters(condition)
    raw_train = draw(seed, 0, 1024, truth)
    mean, sd = raw_train.mean(axis=0), raw_train.std(axis=0)
    train = Batch.make((raw_train - mean) / sd)
    valid = Batch.make((draw(seed, 1, 1024, truth) - mean) / sd)
    bank = fit_bank(train, valid, seed)
    trial = Batch.make((draw(seed, 2, 8192, truth) - mean) / sd)
    audit_features = [Batch.make((draw(seed, 100 + i, 64, truth) - mean) / sd) for i in range(16)]
    uniforms = np.random.default_rng([seed, 99]).random((16, 64))
    worlds = (
        {"zero_p25": replace(truth, prior=0.25), "zero_p75": truth}
        if condition == "zero_signal"
        else {condition: truth}
    )
    for world, data_law in worlds.items():
        standardized_law = standardize(data_law, mean, sd)
        q, _ = standardized_law.evaluate(trial)
        audits = [
            (features.x, (uniforms[i] < standardized_law.evaluate(features)[0]).astype(int))
            for i, features in enumerate(audit_features)
        ]
        for row in evaluate_sentinels(bank, len(valid.x), trial.x, q, audits):
            yield dict(condition=world, **row)


def run(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    began = datetime.now(timezone.utc).isoformat()
    records = []
    for phase, seeds in [("development", range(2400, 2404)), ("confirmation", range(2500, 2524))]:
        for seed in seeds:
            for condition in ("weak", "pair", "zero_signal"):
                records.extend(
                    dict(phase=phase, seed=seed, **row) for row in bank_study(seed, condition)
                )
    df = pd.DataFrame(records)
    df.to_csv(out / "sentinel_draws.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    outcomes = ["excess_brier", "predicted_mean", "constant_mass", "effective_count"]
    banks = df.groupby(["phase", "seed", "condition", "budget", "method"])[outcomes].mean().reset_index()
    banks.to_csv(out / "banks.csv", index=False)
    banks.groupby(["phase", "condition", "budget", "method"])[outcomes].mean().to_csv(out / "means.csv")
    p = banks.query(
        'phase=="confirmation" and condition=="weak" and method=="density_plus_labels"'
    ).pivot(index="seed", columns="budget", values="excess_brier")
    diff = p[0] - p[16]
    radius = t.ppf(0.975, len(diff) - 1) * diff.sem()
    (out / "config.json").write_text(
        json.dumps(
            dict(
                primary=dict(reduction=diff.mean(), lo=diff.mean() - radius, hi=diff.mean() + radius),
                source_draws=0,
                target_fit=1024,
                target_density_validation=1024,
                test=8192,
                sentinel_replications=16,
                development_banks=4,
                confirmation_banks=24,
                compute_seconds=time.perf_counter() - start,
                compute_started_utc=began,
                compute_ended_utc=datetime.now(timezone.utc).isoformat(),
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    with threadpool_limits(limits=1):
        run(parser.parse_args().out)

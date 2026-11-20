"""Trace how unlabeled mixture likelihood relates to latent-class posterior risk."""

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit, xlogy
from scipy.stats import t
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

from fusionbench.eval.joint_moments import OFFSETS, SCALES, STRENGTHS, moments_to_rule

CHECKPOINTS = {0, 1, 4, 16, 64, 192}


@dataclass
class Batch:
    x: np.ndarray
    mean: np.ndarray
    second: np.ndarray

    @classmethod
    def make(cls, x):
        return cls(x, x.mean(0), x.T @ x / len(x))


@dataclass
class Mixture:
    prior: float
    means: np.ndarray
    cov: np.ndarray

    def evaluate(self, data):
        precision = np.linalg.inv(self.cov)
        difference = self.means[1] - self.means[0]
        evidence = (data.x - self.means.mean(0)) @ (precision @ difference)
        p = self.prior
        r = expit(logit(p) + evidence)
        centroid = self.means[0]
        second = (
            data.second
            - np.outer(data.mean, centroid)
            - np.outer(centroid, data.mean)
            + np.outer(centroid, centroid)
        )
        log0 = -0.5 * (
            len(centroid) * np.log(2 * np.pi)
            + np.linalg.slogdet(self.cov)[1]
            + np.sum(precision * second)
        )
        density = log0 + np.logaddexp(np.log1p(-p), np.log(p) + evidence).mean()
        return r, density


def m_step(batch, r):
    mass = r.sum()
    n = len(r)
    # Clipping only avoids degenerate numerical masses during very weak fits.
    p = np.clip(mass / n, 1e-8, 1 - 1e-8)
    if np.ptp(r) <= 16 * np.finfo(float).eps:
        # Identical responsibilities give exactly equal component means.
        cov = batch.second - np.outer(batch.mean, batch.mean) + 1e-6 * np.eye(len(batch.mean))
        return Mixture(p, np.stack([batch.mean, batch.mean]), cov)
    positive = (r @ batch.x) / (p * n)
    negative = (batch.mean - p * positive) / (1 - p)
    cov = batch.second - p * np.outer(positive, positive) - (1 - p) * np.outer(negative, negative)
    cov += 1e-6 * np.eye(len(negative))
    # Equal null-model means have no orientation. Keep the incoming component
    # identity when their projected difference is within floating-point noise.
    tolerance = 64 * np.finfo(float).eps * (1 + np.abs(positive).sum() + np.abs(negative).sum())
    if (positive - negative).sum() < -tolerance:
        p = 1 - p
        negative, positive = positive, negative
    return Mixture(p, np.stack([negative, positive]), cov)


def parameters(condition):
    strength = {"weak": 0.4, "pair": 1.0, "zero_signal": 0.0}[condition] * STRENGTHS
    cov = np.eye(len(strength))
    if condition == "pair":
        cov[0, 1] = cov[1, 0] = 0.75
    return Mixture(
        0.75,
        np.stack([OFFSETS - SCALES * strength, OFFSETS + SCALES * strength]),
        cov * np.outer(SCALES, SCALES),
    )


def draw(seed, split, n, truth):
    rng = np.random.default_rng([seed, split])
    y = (rng.random(n) < truth.prior).astype(int)
    return truth.means[y] + rng.normal(size=(n, truth.means.shape[1])) @ np.linalg.cholesky(truth.cov).T


def standardize(truth, mean, sd):
    return Mixture(truth.prior, (truth.means - mean) / sd, truth.cov / np.outer(sd, sd))


def initial(batch, truth, seed):
    x = batch.x
    states = {}
    for i in range(3):
        labels = KMeans(2, n_init=1, random_state=seed * 3 + i).fit_predict(x)
        states[f"kmeans_{i}"] = m_step(batch, labels)
    third = np.einsum("ni,nj,nk->ijk", x, x, x, optimize=True) / len(x)
    rule = moments_to_rule(batch.second, third, np.zeros(x.shape[1]), np.ones(x.shape[1]))
    states["moments"] = m_step(batch, rule.predict(x))
    states["oracle_start"] = truth
    return states


def trajectory(start, train, validation, test, posterior, density):
    state = start
    prev = None
    stop = None
    rows = []
    for k in range(193):
        probabilities, train_ll = state.evaluate(train)
        if prev is not None and stop is None and train_ll - prev <= 0.001:
            stop = k
        if k in CHECKPOINTS or k == stop:
            _, valid_ll = state.evaluate(validation)
            predicted, test_ll = state.evaluate(test)
            pred_clip = np.clip(predicted, 1e-15, 1 - 1e-15)
            conditional = xlogy(posterior, posterior / pred_clip) + xlogy(
                1 - posterior, (1 - posterior) / (1 - pred_clip)
            )
            rows.append(
                dict(
                    iteration=k,
                    standard_stop=k == stop,
                    grid_point=k in CHECKPOINTS,
                    prior=state.prior,
                    train_ll=train_ll,
                    valid_ll=valid_ll,
                    density_kl=density - test_ll,
                    conditional_kl=conditional.mean(),
                    excess_brier=np.mean((predicted - posterior) ** 2),
                )
            )
        prev = train_ll
        if k < 192:
            state = m_step(train, probabilities)
    if stop is None:
        rows[-1]["standard_stop"] = True
    return rows


def select(rows):
    available = rows.query('start!="oracle_start"')
    grid = available.query("grid_point")
    final = grid.query("iteration==192")
    kmeans_final = final.loc[final.start.str.startswith("kmeans")]
    standard = available.loc[available.start.str.startswith("kmeans") & available.standard_stop]
    return {
        "default": standard.loc[standard.train_ll.idxmax()],
        "strict_train": final.loc[final.train_ll.idxmax()],
        "validated": grid.loc[grid.valid_ll.idxmax()],
        "train_grid": grid.loc[grid.train_ll.idxmax()],
        "kmeans_final": kmeans_final.loc[kmeans_final.train_ll.idxmax()],
        "moments_only": grid.query('start=="moments" and iteration==0').iloc[0],
        "moment_em": grid.query('start=="moments" and iteration==192').iloc[0],
        "oracle_after": rows.query('start=="oracle_start" and iteration==192').iloc[0],
        "oracle_before": rows.query('start=="oracle_start" and iteration==0').iloc[0],
        "hindsight": grid.loc[grid.excess_brier.idxmin()],
    }


def plot_trace(frame, destination):
    import matplotlib.pyplot as plt

    subset = frame.query(
        'phase=="confirmation" and condition=="weak" and budget==1024 and grid_point'
    ).copy()
    subset.loc[subset.start.str.startswith("kmeans"), "start"] = "kmeans"
    banks = subset.groupby(["seed", "start", "iteration"]).mean(numeric_only=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), layout="constrained")
    for ax, metric, title in zip(
        axes, ["density_kl", "excess_brier"], ["Błąd gęstości wejść [nat]", "Nadwyżka Brier klas"]
    ):
        for key, label, color in [
            ("kmeans", "k-means (średnia 3 startów)", "#32689d"),
            ("moments", "Momenty", "#b45b1c"),
            ("oracle_start", "Prawdziwy model", "#35734f"),
        ]:
            values = banks.xs(key, level="start")[metric].groupby("iteration")
            avg, se = values.mean(), values.sem()
            steps = np.log1p(avg.index)
            ax.plot(steps, avg, marker="o", color=color, label=label, markersize=3)
            ax.fill_between(steps, avg - se, avg + se, color=color, alpha=0.12, lw=0)
        points = sorted(CHECKPOINTS)
        ax.set_xticks(np.log1p(points), points)
        ax.set_xlabel("Iteracja EM")
        ax.set_title(title, loc="left", fontsize=11)
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def run(out):
    start = time.perf_counter()
    began = datetime.now(timezone.utc).isoformat()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    traces, selections = [], []
    for phase, seeds in [("development", range(2200, 2204)), ("confirmation", range(2300, 2312))]:
        for seed in seeds:
            for condition in ("weak", "pair", "zero_signal"):
                truth = parameters(condition)
                raw = draw(seed, 0, 4096, truth)
                val = draw(seed, 1, 1024, truth)
                trial = draw(seed, 2, 8192, truth)
                for n in (1024, 4096):
                    mu, sd = raw[:n].mean(0), raw[:n].std(0)
                    train, valid, test = [Batch.make((z - mu) / sd) for z in (raw[:n], val, trial)]
                    normalized = standardize(truth, mu, sd)
                    q, density = normalized.evaluate(test)
                    keys = dict(phase=phase, seed=seed, condition=condition, budget=n)
                    rows = []
                    for name, state in initial(train, normalized, seed).items():
                        for record in trajectory(state, train, valid, test, q, density):
                            rows.append(dict(**keys, start=name, **record))
                    traces.extend(rows)
                    for method, row in select(pd.DataFrame(rows)).items():
                        selections.append(dict(method=method, **row.to_dict()))
    df = pd.DataFrame(selections)
    trace_frame = pd.DataFrame(traces)
    trace_frame.to_csv(out / "trajectories.csv", index=False)
    plot_trace(trace_frame, out / "em-traces.png")
    df.to_csv(out / "selections.csv", index=False)
    df.groupby(["phase", "condition", "budget", "method"])[
        ["iteration", "prior", "density_kl", "conditional_kl", "excess_brier"]
    ].mean().to_csv(out / "means.csv")
    p = df.query('phase=="confirmation" and condition=="weak" and budget==1024').pivot(
        index="seed", columns="method", values="excess_brier"
    )
    delta = p.default - p.validated
    radius = t.ppf(0.975, len(delta) - 1) * delta.sem()
    (out / "config.json").write_text(
        json.dumps(
            dict(
                primary=dict(reduction=delta.mean(), lo=delta.mean() - radius, hi=delta.mean() + radius),
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

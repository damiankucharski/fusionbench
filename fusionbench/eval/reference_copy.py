"""Falsify unlabeled reference agreement as a test of expert reliability."""
from __future__ import annotations

import argparse
import itertools
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from fusionbench.eval.context_crossings import conditional_risk
from fusionbench.methods.okapi import OKAPIMethod

WEAK, STRONG, REFERENCE = .25, .85, .75
REGIMES = np.array([[STRONG, WEAK], [WEAK, STRONG]])
PATTERNS = np.array(list(itertools.product((-1., 1.), repeat=2)))


def alias_parameters(weak=WEAK, strong=STRONG, reference=REFERENCE):
    """Match E[weak*H], E[strong*H] to those of the opposite clean regime."""
    residual = weak * reference * (1 - strong**2) / (strong * (1 - weak**2))
    copy_probability = strong * reference - weak * residual
    return copy_probability, residual / (1 - copy_probability)


def draw(seed, shape, regime, fraction):
    rng = np.random.default_rng(seed)
    y = 2 * rng.integers(2, size=shape) - 1
    def sensor(strength):
        return y * np.where(rng.random(shape) < (1 + strength) / 2, 1, -1)
    x = np.stack([sensor(r) for r in REGIMES[regime]], axis=-1)
    h_clean, other = sensor(REFERENCE), sensor(REFERENCE)
    prob_copy, noisy_strength = alias_parameters()
    alias = np.where(rng.random(shape) < prob_copy, x[..., np.argmin(REGIMES[regime])], sensor(noisy_strength))
    primary = np.where(rng.random(shape) < fraction, alias, h_clean)
    return x, y, primary, other


def scores(x):
    positive = (1 + .55 * x) / 2
    return np.stack([1 - positive, positive], axis=-1)


def reference_features(x, reference):
    return (x != reference[..., None]).astype(float)


def choose(features, clean, risk):
    from okapi.selection import estimate_context_prevalence

    prevalence = np.array([estimate_context_prevalence(row, clean[0], clean[1], .5) for row in features])
    estimated_loss = risk[0] + prevalence[:, None] * (risk[1] - risk[0])
    return estimated_loss.argmin(axis=1), prevalence


def fit_trees(seed, generations, n_source):
    prediction, trees, reference_source, calibration = [], [], [], []
    for index in range(2):
        x, y, _, _ = draw([seed, 0, index], (n_source,), index, 0.)
        model = OKAPIMethod(seed=1000*seed+index, fitness='brier', generations=generations,
                            n_restarts=1, n_seeds=4, population_size=2, tournament_size=2,
                            population_multiplier=2)
        try:
            model.fit(scores(x), (1+y)//2, task='binary', n_classes=2)
            prediction.append(model.predict(scores(PATTERNS))[:, 1])
            trees.append(dict(seed=seed, regime=index, gp_seed=1000*seed+index,
                              tree=model.tree_repr(), nodes=int(model.best_nodes)))
        finally:
            model._tmp.cleanup()
        vx, vy, vh, _ = draw([seed, 1, index], (n_source,), index, 0.)
        reference_source.append(reference_features(vx, vh))
        pattern_idx = 2 * (vx[:, 0] > 0).astype(int) + (vx[:, 1] > 0)
        calibration.append((pattern_idx, (vy+1)/2))
    pred = np.array(prediction)
    source_risk = np.array([((pred[:, i] - y)**2).mean(axis=1) for i,y in calibration])
    return pred, trees, reference_source, source_risk


def run_cell(seed, generations, n_source, batches):
    pred, trees, clean, source_risk = fit_trees(seed, generations, n_source)
    candidates = np.row_stack((pred, pred.mean(axis=0)))
    records = []
    for j, strengths in enumerate(REGIMES):
        risks = [conditional_risk(p, PATTERNS, strengths[None, :]) for p in candidates]
        briers, accuracy = (np.array(metric).ravel() for metric in zip(*risks))
        for fraction in (0., .25, .5, .75, 1.):
            # Common random numbers for all corruption fractions and directions of selection.
            x, _, ref, other = draw([seed, 2, j], (batches, 128), j, fraction)
            for n in (8, 32, 128):
                h, prob_h = choose(reference_features(x[:, :n], ref[:, :n]).mean(axis=1), clean, source_risk)
                d, prob_d = choose(reference_features(x[:, :n], other[:, :n]).mean(axis=1), clean, source_risk)
                for method, selected in {
                        'reference': h, 'other_reference': d,
                        'disagreement_fallback': np.where(h == d, h, 2),
                        # The same H reused as a purported witness agrees identically.
                        'duplicate_fallback': h, 'equal_mixture': np.full(batches, 2),
                        'true_regime': np.full(batches, np.argmin(source_risk[j]))}.items():
                    records.append(dict(seed=seed, regime=j, corruption=fraction, support=n,
                                        method=method, brier=briers[selected].mean(),
                                        accuracy=accuracy[selected].mean(),
                                        mismatch=(h != d).mean(), select_mixture=(selected==2).mean(),
                                        reference_estimate=prob_h.mean(), other_estimate=prob_d.mean(),
                                        select_true_tree=(selected==j).mean()))
    return records, trees


def summarize(output, frame):
    from scipy.stats import t
    group = ['seed','corruption','support','method']
    wide = frame.groupby(group)[['accuracy','brier','mismatch','select_true_tree']].mean().reset_index()
    wide.groupby(group[1:])[['accuracy','brier','mismatch','select_true_tree']].mean().to_csv(output/'means.csv')
    contrasts = []
    for (fraction, n), subset in wide.groupby(['corruption','support']):
        for metric in ['accuracy','brier']:
            scores = subset.pivot(index='seed',columns='method',values=metric)
            for name in ('other_reference','equal_mixture','disagreement_fallback','duplicate_fallback'):
                delta = scores[name] - scores['reference']
                half = t.ppf(.975,len(delta)-1)*delta.sem()
                contrasts.append(dict(corruption=fraction,support=n,metric=metric,method=name,
                                      reference='reference',delta=delta.mean(),low=delta.mean()-half,
                                      high=delta.mean()+half))
    pd.DataFrame(contrasts).to_csv(output/'paired.csv',index=False)


def run(output, seeds, generations=8, n_source=384, batches=200):
    from loguru import logger
    logger.disable('okapi')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    seeds = list(seeds)
    if len(seeds)<2 or len(set(seeds))!=len(seeds) or min(seeds)<0 or min(generations,n_source,batches)<=0:
        raise ValueError('use distinct nonnegative seeds and positive budgets')
    start = datetime.now(timezone.utc)
    timer = time.perf_counter()
    records, trees = [], []
    for seed in seeds:
        rows, bank = run_cell(seed, generations, n_source, batches)
        records.extend(rows)
        trees.extend(bank)
    frame = pd.DataFrame(records)
    frame.to_csv(output/'risks.csv',index=False)
    pd.DataFrame(trees).to_csv(output/'trees.csv',index=False)
    summarize(output,frame)
    config = dict(seeds=seeds, generations=generations, fit_per_regime=n_source,
                  calibration_per_regime=n_source, target_batches_per_regime=batches,
                  strengths=REGIMES.tolist(), reference_strength=REFERENCE,
                  alias_parameters=alias_parameters(), compute_start_utc=start.isoformat(),
                  compute_seconds=time.perf_counter()-timer)
    (output/'config.json').write_text(json.dumps(config,indent=2))


if __name__ == '__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output',type=Path,required=True)
    cli.add_argument('--seeds',type=int,nargs='+',default=list(range(400,412)))
    args=cli.parse_args()
    run(args.output,args.seeds)

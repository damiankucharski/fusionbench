"""Assess held-out shrinkage under search variability and model competence shift."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits

from fusionbench.methods.averaging import SimpleAverage
from fusionbench.methods.okapi import OKAPIMethod
from fusionbench.methods.shrinkage import brier_tree_weight
from fusionbench.methods.stacking import StackingLogit
from fusionbench.scenarios import calibrated
from fusionbench.scenarios.calibration import MulticlassCommittee
from fusionbench.scenarios.committee import DeadWeight
from fusionbench.scenarios.correlated import CorrelatedRedundancy
from fusionbench.scenarios.gates import Compound
from fusionbench.scenarios.mixture import GatedExperts


def scenarios():
    return {
        'deadweight': (DeadWeight(), DeadWeight(accuracy=.65)),
        'compound': (Compound(), Compound(acc_core=.67, acc_rescue=.74, acc_safe=.76)),
        'correlated': (CorrelatedRedundancy(), CorrelatedRedundancy(corr=.1)),
        'gated': (GatedExperts(), GatedExperts(acc_select=.65, acc_expert=.72)),
        'multiclass': (MulticlassCommittee(), MulticlassCommittee(accuracy=.40)),
    }


def draw(generator, size, seed, stream):
    x, y, _ = generator._generate(size, np.random.default_rng([seed, stream]))
    return x, y


def metrics(pred, y):
    yhat = pred.argmax(axis=1)
    return {
        'accuracy': float(np.mean(yhat == y)),
        'brier': float(np.mean(np.sum((pred - np.eye(pred.shape[1])[y]) ** 2, axis=1))),
        'macro_f1': float(f1_score(y, yhat, average='macro', zero_division=0)),
    }


def fixed_data(name, seed, fit_n, cal_n, test_n):
    source, shifted = scenarios()[name]
    fit = draw(source, fit_n, seed, 0)
    calibration = draw(source, cal_n, seed, 1)
    trials = {'iid': draw(source, test_n, seed, 2),
              'parameter_shift': draw(shifted, test_n, seed, 3)}
    if name == 'deadweight':
        # Existing useful identities stay stable. Former noise models become
        # systematically wrong, so an anchor averaging them can extrapolate poorly.
        x, y = trials['iid']
        corrupt = x.copy()
        rng = np.random.default_rng([seed, 4])
        for j in range(source.n_good, corrupt.shape[1]):
            corrupt[:, j] = calibrated.to_binary_proba(calibrated.binary_calibrated(1 - y, .85, rng))
        trials['anti_experts'] = (corrupt, y)
    return fit, calibration, trials


def evaluate_cell(spec):
    from loguru import logger
    logger.disable('okapi')
    name, data_seed, seeds, fit_n, cal_n, test_n, generations = spec
    (x, y), (x_cal, y_cal), trials = fixed_data(name, data_seed, fit_n, cal_n, test_n)
    full_x, full_y = np.concatenate((x, x_cal)), np.concatenate((y, y_cal))
    task = 'binary' if x.shape[-1] == 2 else 'multiclass'
    fit_kw = dict(task=task, n_classes=x.shape[-1])
    rows, trees = [], []
    base_cal = SimpleAverage().predict(x_cal)
    anchor = {d: SimpleAverage().predict(xq) for d, (xq, _) in trials.items()}
    stack = StackingLogit().fit(full_x, full_y, **fit_kw)
    common = dict(scenario=name, data_seed=data_seed, fit_n=fit_n, cal_n=cal_n, test_n=test_n)
    for domain, (xq, yq) in trials.items():
        for method, pred in [('mean', anchor[domain]), ('logit_stack_full', stack.predict(xq))]:
            rows.append(dict(**common, domain=domain, method=method, objective='independent',
                             search_seed=-1, weight=np.nan, **metrics(pred, yq)))
    for objective in ('macro_f1', 'brier'):
        seed_pool = {d: [] for d in trials}
        for seed in seeds:
            options = dict(seed=seed, fitness=objective, generations=generations, n_restarts=1,
                           population_size=8, population_multiplier=2, tournament_size=3, n_seeds=6)
            for budget, fit_x, fit_y in [('split', x, y), ('full', full_x, full_y)]:
                gp = OKAPIMethod(**options)
                try:
                    gp.fit(fit_x, fit_y, **fit_kw)
                    tree_cal = gp.predict(x_cal) if budget == 'split' else None
                    weight = brier_tree_weight(tree_cal, base_cal, y_cal) if tree_cal is not None else np.nan
                    trees.append(dict(**common, search_seed=seed, objective=objective,
                                      budget=budget, tree=gp.tree_repr(), nodes=int(gp.best_nodes), weight=weight))
                    for domain, (xq, yq) in trials.items():
                        pred = gp.predict(xq)
                        if budget == 'full':
                            models = [('raw_full', pred, 1.)]
                        else:
                            simple = anchor[domain]
                            blend = weight * pred + (1. - weight) * simple
                            models = [('raw', pred, 1.), ('fixed_half', (pred + simple) / 2, .5),
                                      ('shrunk', blend, weight)]
                            seed_pool[domain].append(pred)
                        for method, scores, w in models:
                            rows.append(dict(**common, domain=domain, method=method, objective=objective,
                                             search_seed=seed, weight=w, **metrics(scores, yq)))
                finally:
                    gp._tmp.cleanup()
        for domain, (_, yq) in trials.items():
            rows.append(dict(**common, domain=domain, method='mean_seeds', objective=objective,
                             search_seed=-1, weight=np.nan, **metrics(np.mean(seed_pool[domain], axis=0), yq)))
    return rows, trees


def paired_contrasts(table):
    """Treat each independent dataset as one unit after averaging search replicates."""
    from scipy.stats import t

    contrasts = []
    for (name, domain), block in table.groupby(['scenario', 'domain']):
        controls = block[block.objective == 'independent']
        for objective, trial in block[block.objective != 'independent'].groupby('objective'):
            trials = pd.concat([controls, trial], ignore_index=True)
            for column in ('accuracy', 'brier', 'macro_f1'):
                scores = trials.pivot_table(index='data_seed', columns='method', values=column)
                for reference in scores.columns.drop('shrunk'):
                    delta = scores['shrunk'] - scores[reference]
                    margin = t.ppf(.975, len(delta) - 1) * delta.sem() if len(delta) > 1 else np.nan
                    contrasts.append(dict(scenario=name, domain=domain, objective=objective,
                                          metric=column, reference=reference, data_seeds=len(delta),
                                          difference=delta.mean(), ci95_low=delta.mean() - margin,
                                          ci95_high=delta.mean() + margin))
    return pd.DataFrame(contrasts)


def run_study(output, data_seeds=range(4), search_seeds=range(3), generations=8,
              fit_n=384, cal_n=128, test_n=3072, workers=1, scenario_names=None):
    """Cross independent datasets and GP searches; score on fresh evaluation splits."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    names = list(scenarios()) if scenario_names is None else list(scenario_names)
    data_seeds, search_seeds = list(data_seeds), list(search_seeds)
    if (not names or not search_seeds or not data_seeds or
            min(fit_n, cal_n, test_n, workers, generations) <= 0):
        raise ValueError('use nonempty seeds/scenarios and positive budgets')
    config = dict(data_seeds=data_seeds, search_seeds=search_seeds, generations=generations,
                  fit_n=fit_n, cal_n=cal_n, test_n=test_n, scenarios=names)
    (output / 'config.json').write_text(json.dumps(config, indent=2))
    jobs = [(name, s, search_seeds, fit_n, cal_n, test_n, generations) for name in names for s in data_seeds]
    rows, trees = [], []
    pool_context = ProcessPoolExecutor(max_workers=workers) if workers > 1 else nullcontext()
    with threadpool_limits(limits=1), pool_context as pool:
        results = pool.map(evaluate_cell, jobs) if pool is not None else map(evaluate_cell, jobs)
        for scores, fitted in results:
            rows.extend(scores)
            trees.extend(fitted)
            pd.DataFrame(rows).to_csv(output / 'metrics.csv', index=False)
            pd.DataFrame(trees).to_csv(output / 'trees.csv', index=False)
    table = pd.DataFrame(rows)
    keys = ['scenario', 'domain', 'objective', 'method', 'data_seed']
    per_data = table.groupby(keys)
    by_seed = per_data[['brier', 'accuracy', 'macro_f1']].mean()
    summary = by_seed.groupby(keys[:-1]).mean()
    variability = per_data[['brier', 'accuracy']].std().groupby(keys[:-1]).mean()
    summary.join(variability.add_prefix('within_data_search_sd_')).to_csv(output / 'summary.csv')
    paired_contrasts(table).to_csv(output / 'paired.csv', index=False)
    return table

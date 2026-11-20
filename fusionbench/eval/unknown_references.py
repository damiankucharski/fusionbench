"""Unlabeled tree selection when any one deployment reference can share errors."""
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
from fusionbench.eval.reference_copy import PATTERNS, REFERENCE, REGIMES, alias_parameters, fit_trees

CONDITIONS = {
    'honest': .75, 'alias_one': .75, 'weak_honest': .40,
    'weak_one_copy': .40, 'weak_two_duplicate': .40,
}


def four_bit_law(world):
    """World 0 has its second reference faulty; world 1 has its first faulty."""
    k, residual = alias_parameters()
    strengths = REGIMES[world]
    result = []
    for a,b,h,d in itertools.product((-1,1), repeat=4):
        weak = (a,b)[int(np.argmin(strengths))]
        bad, good = (d,h) if world == 0 else (h,d)
        mass = 0.
        for y in (-1,1):
            primary = (1+y*a*strengths[0])*(1+y*b*strengths[1])/4
            honest = (1+y*good*REFERENCE)/2
            copied = k*(bad==weak) + (1-k)*(1+y*bad*residual)/2
            mass += .5 * primary * honest * copied
        result.append(mass)
    return np.array(result)


def session(seed, regime, count, budget, n_refs, condition):
    rng = np.random.default_rng(seed)
    y = 2*rng.integers(2, size=(count,budget)) - 1
    x = y[...,None] * np.where(rng.random((count,budget,2)) < (1+REGIMES[regime])/2, 1, -1)
    h = y[...,None] * np.where(rng.random((count,budget,n_refs)) < (1+CONDITIONS[condition])/2, 1, -1)
    order = np.argsort(rng.random((count,n_refs)),axis=1)
    if condition not in ('honest','weak_honest'):
        weak = x[..., int(np.argmin(REGIMES[regime]))]
        if condition == 'alias_one':
            k, residual = alias_parameters()
            alias = np.where(rng.random(y.shape)<k, weak,
                             y*np.where(rng.random(y.shape)<(1+residual)/2,1,-1))
        else:
            alias = weak
        bad_count = 2 if condition == 'weak_two_duplicate' else 1
        for j in range(bad_count):
            h[np.arange(count),:,order[:,j]] = alias
    return x, h, order[:,-1]


def risk_selection(prevalence, calibration):
    expected = calibration[0] + prevalence[:,None] * (calibration[1]-calibration[0])
    return expected.argmin(axis=1)


def run_cell(seed, n_source=384, batches=200):
    from okapi.selection import estimate_context_prevalences
    pred, trees, _, risk = fit_trees(seed,8,n_source)
    candidates = np.vstack((pred,pred.mean(axis=0)))
    rows = []
    for condition, ref_r in CONDITIONS.items():
        cal_condition = 'honest' if ref_r == .75 else 'weak_honest'
        cal = []
        for j in (0,1):
            x,h,_ = session([seed,4,j],j,1,n_source,1,cal_condition)
            cal.append((x[0]!=h[0,:,0,None]).astype(float))
        for j in (0,1):
            losses = [conditional_risk(p,PATTERNS,REGIMES[j,None,:]) for p in candidates]
            brier, accuracy = (np.array(metric).ravel() for metric in zip(*losses))
            for m in (2,3):
                if m==2 and condition=='weak_two_duplicate':
                    continue
                x,h,honest = session([seed,5,j,m],j,batches,128,m,condition)
                for n in (8,32,128):
                    # Matrix of mean disagreements [batch, reference, primary].
                    features = (h[:,:n,:,None] != x[:,:n,None,:]).mean(axis=1)
                    pooled = estimate_context_prevalences(features.mean(axis=1),*cal,.5)
                    estimates = estimate_context_prevalences(features,*cal,.5)
                    methods = {'median': np.median(estimates,axis=1), 'pooled': pooled,
                               'mean_clipped': estimates.mean(axis=1), 'first': estimates[:,0],
                               'known_honest': estimates[np.arange(batches),honest]}
                    selectors = {k: risk_selection(p,risk) for k,p in methods.items()}
                    selectors['equal_mixture'] = np.full(batches,2)
                    selectors['known_regime'] = np.full(batches,risk[j].argmin())
                    for name, selected in selectors.items():
                        rows.append(dict(seed=seed,regime=j,condition=condition,n_refs=m,support=n,
                                         method=name,brier=brier[selected].mean(),
                                         accuracy=accuracy[selected].mean(),
                                         select_true=(selected==j).mean()))
    return rows, trees


def run(output, seeds):
    from loguru import logger
    from scipy.stats import t
    logger.disable('okapi')
    start = datetime.now(timezone.utc)
    timer = time.perf_counter()
    seeds = list(seeds)
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    rows, trees = [], []
    for seed in seeds:
        trial, bank = run_cell(seed)
        rows.extend(trial)
        trees.extend(bank)
    result = pd.DataFrame(rows)
    result.to_csv(output/'risks.csv',index=False)
    pd.DataFrame(trees).to_csv(output/'trees.csv',index=False)
    group = ['condition','n_refs','support']
    by_seed = result.groupby(group+['method','seed'])[['brier','accuracy','select_true']].mean().reset_index()
    by_seed.groupby(group+['method'])[['brier','accuracy','select_true']].mean().to_csv(output/'means.csv')
    rows = []
    for k,table in by_seed.groupby(group):
        for metric in ['accuracy','brier']:
            wide = table.pivot(index='seed',columns='method',values=metric)
            for ref in ['pooled','mean_clipped','first','equal_mixture','known_honest','known_regime']:
                delta = wide['median']-wide[ref]
                margin=t.ppf(.975,len(delta)-1)*delta.sem()
                rows.append(dict(zip(group,k),metric=metric,reference=ref,
                                 median_minus_reference=delta.mean(),low=delta.mean()-margin,high=delta.mean()+margin))
    pd.DataFrame(rows).to_csv(output/'paired.csv',index=False)
    pd.DataFrame({'world_0': four_bit_law(0),'world_1':four_bit_law(1)},
                 index=list(itertools.product((-1,1),repeat=4))).to_csv(output/'four_bit_mass.csv')
    config=dict(seeds=seeds,source_per_regime=384,calibration_per_regime=384,
                target_batches_per_regime=200,gp_generations=8,compute_seconds=time.perf_counter()-timer,
                compute_start_utc=start.isoformat())
    (output/'config.json').write_text(json.dumps(config,indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seeds',type=int,nargs='+',default=list(range(500,512)))
    args=parser.parse_args()
    run(args.output,args.seeds)

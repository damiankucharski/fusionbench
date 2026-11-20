"""Control false witness declarations under randomized paired interventions."""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom, t

from fusionbench.eval.context_crossings import conditional_risk
from fusionbench.eval.intervention_witness import probes
from fusionbench.eval.reference_copy import PATTERNS, REGIMES, fit_trees
from fusionbench.eval.unknown_references import risk_selection, session


def pair_calls(ordinary, alternative, randomized, first_rate, second_rate, seed):
    rng = np.random.default_rng(seed)
    baseline = ordinary * np.where(rng.random(ordinary.shape)<.1,-1,1)
    random_order = rng.random(ordinary.shape[:-1]+(1,)) < .5
    active_first = random_order if randomized else np.zeros_like(random_order)
    # Independent position-specific noise allows the distribution of the
    # second call to differ even when both witnesses satisfy the sharp null.
    first = np.where(active_first,alternative,ordinary) * np.where(rng.random(ordinary.shape)<first_rate,-1,1)
    second = np.where(active_first,ordinary,alternative) * np.where(rng.random(ordinary.shape)<second_rate,-1,1)
    return np.where(active_first,first,second)!=baseline, np.where(active_first,second,first)!=baseline


def fixed_decisions(active, sham):
    difference = active.astype(int)-sham
    positive = (difference>0).sum(axis=1)
    discordant = (difference!=0).sum(axis=1)
    pvalue = binom.sf(positive-1, discordant, .5)
    crosses = pvalue <= .025
    exact = np.where(crosses.sum(axis=1)==1,crosses.argmax(axis=1),-1)
    contrast = difference.sum(axis=1)
    raw = np.where(contrast[:,0]==contrast[:,1],-1,contrast.argmax(axis=1))
    return exact,raw


def cell(seed, batches):
    from okapi.probes import first_probe_declaration, probe_log_evalues
    from okapi.selection import estimate_context_prevalences

    bank, trees, clean, calibration = fit_trees(seed,8,384)
    models=np.vstack((bank,bank.mean(axis=0)))
    source_rng=np.random.default_rng([seed,15])
    source=[np.where(source_rng.random((len(a),1))<.1,1-a,a) for a in clean]
    rows=[]
    for regime in (0,1):
        x,h,honest=session([seed,11,regime],regime,batches,32,2,'alias_one')
        bad=1-honest
        rng=np.random.default_rng([seed,12,regime])
        h=h*np.where(rng.random(h.shape)<.1,-1,1)
        support=(h[...,None]!=x[:,:,None,:]).mean(axis=1)
        prevalence=estimate_context_prevalences(support,*source,.5)
        vote=np.column_stack([risk_selection(prevalence[:,i],calibration) for i in (0,1)])
        ordinary,inverted=probes(seed,regime,bad,64)
        losses=[conditional_risk(p,PATTERNS,REGIMES[regime,None,:]) for p in models]
        brier,accuracy=(np.array(a).ravel() for a in zip(*losses))
        for mechanism,new in [('direct',inverted),('upstream',ordinary)]:
            for setting,first_rate,second_rate in [('stationary',.1,.1),('order_shift',.02,.24)]:
                for randomized in (True,False):
                    active,sham=pair_calls(ordinary,new,randomized,first_rate,second_rate,[seed,13,regime])
                    evidence=probe_log_evalues(active,sham)
                    for budget in (4,16,64):
                        sequential=first_probe_declaration(evidence[:,:budget])
                        exact,raw=fixed_decisions(active[:,:budget],sham[:,:budget])
                        answers={
                            'sequential': (sequential.index,sequential.records),
                            'fixed_sign': (exact,budget),
                            'unthresholded': (raw,budget),
                            'passive_mix': (np.full(batches,-1),0),
                            'known_honest': (bad,0),
                        }
                        for name,(guess,consumed) in answers.items():
                            declared=guess>=0
                            candidate=np.where(declared,vote[np.arange(batches),1-guess.clip(0,1)],2)
                            if name=='known_honest':
                                # Privileged selection has no tested declaration.
                                declared=np.zeros(batches,dtype=bool)
                            true=(guess==bad)&(mechanism=='direct')
                            rows.append(dict(seed=seed,regime=regime,mechanism=mechanism,noise=setting,
                                             randomized=randomized,budget=budget,method=name,
                                             brier=brier[candidate].mean(),accuracy=accuracy[candidate].mean(),
                                             false_declaration=(declared&~true).mean(),
                                             true_declaration=(declared&true).mean(),records=np.mean(consumed)))
    return rows,trees


def run(output,seeds):
    from loguru import logger
    logger.disable('okapi')
    start=datetime.now(timezone.utc)
    timer=time.perf_counter()
    seeds=list(seeds)
    root=Path(output)
    root.mkdir(parents=True,exist_ok=True)
    observations,trees=[],[]
    for seed in seeds:
        obs,bank=cell(seed,200)
        observations.extend(obs)
        trees.extend(bank)
    df=pd.DataFrame(observations)
    df.to_csv(root/'risks.csv',index=False)
    pd.DataFrame(trees).to_csv(root/'trees.csv',index=False)
    groups=['mechanism','noise','randomized','budget']
    measures=['brier','accuracy','false_declaration','true_declaration','records']
    per_seed=df.groupby(groups+['method','seed'])[measures].mean().reset_index()
    per_seed.groupby(groups+['method'])[measures].mean().to_csv(root/'means.csv')
    rows=[]
    for k,block in per_seed.groupby(groups):
        for metric in measures:
            wide=block.pivot(index='seed',columns='method',values=metric)
            for ref in ('passive_mix','fixed_sign','unthresholded','known_honest'):
                delta=wide['sequential']-wide[ref]
                margin=t.ppf(.975,len(delta)-1)*delta.sem()
                rows.append(dict(zip(groups,k),metric=metric,reference=ref,delta=delta.mean(),
                                 low=delta.mean()-margin,high=delta.mean()+margin))
    pd.DataFrame(rows).to_csv(root/'paired.csv',index=False)
    (root/'config.json').write_text(json.dumps(dict(seeds=seeds,fit_per_regime=384,
            calibration_per_regime=384,test_batches_per_regime=200,ordinary_support=32,
            alpha=.05,gp_generations=8,compute_start_utc=start.isoformat(),
            compute_seconds=time.perf_counter()-timer),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seeds',type=int,nargs='+',default=list(range(700,712)))
    args=parser.parse_args()
    run(args.output,args.seeds)

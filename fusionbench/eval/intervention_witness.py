"""Identify a directly dependent reference using paired queries without labels."""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from fusionbench.eval.context_crossings import conditional_risk
from fusionbench.eval.reference_copy import PATTERNS, REGIMES, REFERENCE, alias_parameters, fit_trees
from fusionbench.eval.unknown_references import risk_selection, session


def replay_tree(private_sign, supplied_sign, gate):
    """A real OKAPI far-threshold witness chooses the more confident input."""
    from okapi.node import FarThresholdNode, ValueNode
    from okapi.tree import Tree

    private = .5 + .2*private_sign
    supplied = .5 + np.where(gate,.4,.05)*supplied_sign
    root = ValueNode(None,private,'private')
    root.add_child(FarThresholdNode([ValueNode(None,supplied,'supplied')],.5))
    tree = Tree.create_tree_from_root(root)
    observed = tree.predict()
    intervened = tree.do_pred_on_another_tensors(current_tensors={'private':private,'supplied':1-supplied})
    return np.where(observed>.5,1,-1), np.where(intervened>.5,1,-1)


def probes(seed, regime, faulty, count=16):
    rng = np.random.default_rng([seed,7,regime])
    y = 2*rng.integers(2,size=(len(faulty),count))-1
    k, residual = alias_parameters()
    def measure(r):
        return y*np.where(rng.random(y.shape)<(1+r)/2,1,-1)
    weak = measure(REGIMES[regime].min())
    gates = rng.random(y.shape)<k
    observed, intervened = replay_tree(measure(residual),weak,gates)
    healthy = measure(REFERENCE)
    def order(signal):
        return np.where(faulty[:,None,None] == np.arange(2),signal[...,None],healthy[...,None])
    return order(observed), order(intervened)


def pick(measure, votes):
    """Infer the directly responding witness, or use the fixed model mixture on ties."""
    difference = measure[:,0] - measure[:,1]
    inferred_bad = np.where(difference>0,0,1)
    choice=votes[np.arange(len(votes)),1-inferred_bad]
    undecided=difference==0
    return np.where(undecided,2,choice),np.where(undecided,-1,inferred_bad)


def cell(seed, batches):
    from okapi.selection import estimate_context_prevalences
    bank, trees, clean, calibration = fit_trees(seed,8,384)
    models=np.vstack((bank,bank.mean(axis=0)))
    rows=[]
    for j in range(2):
        x, h, honest = session([seed,8,j],j,batches,32,2,'alias_one')
        bad=1-honest
        orig, inverted=probes(seed,j,bad)
        risks=[conditional_risk(p,PATTERNS,REGIMES[j,None,:]) for p in models]
        bs,acc=(np.array(a).ravel() for a in zip(*risks))
        for jitter in (0.,.10):
            rng=np.random.default_rng([seed,9,j])
            def read(arr):
                return arr*np.where(rng.random(arr.shape)<jitter,-1,1)
            # Independent measurement flips of one reference invert both disagreements.
            source_rng=np.random.default_rng([seed,10])
            source=[np.where(source_rng.random((len(c),1))<jitter,1-c,c) for c in clean]
            features=(read(h)[...,None]!=x[:,:,None,:]).mean(axis=1)
            p=estimate_context_prevalences(features,*source,.5)
            votes=np.column_stack([risk_selection(p[:,i],calibration) for i in range(2)])
            baseline=read(orig)
            sham=read(orig)
            placebo=read(orig)
            noise=np.where(rng.random(orig.shape)<jitter,-1,1)
            for name, latent in [('direct',inverted),('upstream',orig)]:
                flipped=latent*noise
                for m in (1,4,16):
                    n_flip=(flipped[:,:m]!=baseline[:,:m]).sum(axis=1)
                    n_sham=(sham[:,:m]!=baseline[:,:m]).sum(axis=1)
                    n_placebo=(placebo[:,:m]!=baseline[:,:m]).sum(axis=1)
                    informed={
                        'paired_excess': n_flip - n_sham,
                        'flip_count': n_flip,
                        'any_change': (n_flip>0).astype(int),
                        'sham_only': n_placebo - n_sham,
                    }
                    selections={key:pick(stat,votes) for key,stat in informed.items()}
                    selections['passive_mix']=(np.full(batches,2),np.full(batches,-1))
                    selections['known_honest']=(votes[np.arange(batches),honest],bad)
                    for method,(selected,guess) in selections.items():
                        rows.append(dict(seed=seed,regime=j,mechanism=name,jitter=jitter,probes=m,method=method,
                                         accuracy=acc[selected].mean(),brier=bs[selected].mean(),
                                         correct_witness=(guess==bad).mean(),misidentify=((guess>=0)&(guess!=bad)).mean(),
                                         abstain=(guess<0).mean()))
    return rows,trees


def run(output, seeds):
    from loguru import logger
    from scipy.stats import t
    logger.disable('okapi')
    start=datetime.now(timezone.utc)
    timer=time.perf_counter()
    seeds=list(seeds)
    folder=Path(output)
    folder.mkdir(parents=True,exist_ok=True)
    rows,trees=[],[]
    for seed in seeds:
        records, bank = cell(seed,200)
        rows.extend(records)
        trees.extend(bank)
    d=pd.DataFrame(rows)
    d.to_csv(folder/'risks.csv',index=False)
    pd.DataFrame(trees).to_csv(folder/'trees.csv',index=False)
    groups=['mechanism','jitter','probes']
    study=d.groupby(groups+['method','seed'])[['brier','accuracy','correct_witness','misidentify','abstain']].mean().reset_index()
    study.groupby(groups+['method'])[['brier','accuracy','correct_witness','misidentify','abstain']].mean().to_csv(folder/'means.csv')
    contrasts=[]
    for key,sub in study.groupby(groups):
        for metric in ('brier','accuracy'):
            wide=sub.pivot(index='seed',columns='method',values=metric)
            for ref in ('passive_mix','sham_only','known_honest','any_change','flip_count'):
                diff=wide['paired_excess']-wide[ref]
                half=t.ppf(.975,len(diff)-1)*diff.sem()
                contrasts.append(dict(zip(groups,key),metric=metric,reference=ref,
                                      delta=diff.mean(),low=diff.mean()-half,high=diff.mean()+half))
    pd.DataFrame(contrasts).to_csv(folder/'paired.csv',index=False)
    (folder/'config.json').write_text(json.dumps(dict(seeds=seeds,target_batches_per_regime=200,
        ordinary_support=32,probes=[1,4,16],jitter=[0.,.1],gp_generations=8,fit_per_regime=384,
        calibration_per_regime=384,compute_start_utc=start.isoformat(),compute_seconds=time.perf_counter()-timer),indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seeds',type=int,nargs='+',default=list(range(600,612)))
    args=parser.parse_args()
    run(args.output,args.seeds)

"""Native digit views: separate reliance on a channel from unique predictive utility."""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from fusionbench.methods.clr import StackingCLR
from fusionbench.methods.okapi import OKAPIMethod


def logistic(x,y):
    return LogisticRegression(C=1., tol=1e-8, max_iter=4000).fit(x,y)


def score(prob,y):
    return dict(brier=float(((prob-np.eye(10)[y])**2).sum(axis=1).mean()),
                accuracy=float(np.mean(prob.argmax(axis=1)==y)))


def compute_fold(fold,x,y,fold_id):
    base=(fold_id!=fold) & (fold_id!=(fold+1)%5)
    fit=fold_id==(fold+1)%5
    test=fold_id==fold
    masks=[np.tile(np.arange(8)<4,8),np.tile(np.arange(8)>=4,8)]
    experts=[logistic(x[base][:,m],y[base]) for m in masks]
    def expert_pred(rows):
        return np.stack([model.predict_proba(x[rows][:,m]) for m,model in zip(masks,experts)],axis=1)
    p,q=expert_pred(fit),expert_pred(test)
    duplicated=[0,0,1]
    dfit, dtest = p[:,duplicated], q[:,duplicated]
    def wrapper(training,multiplicities=None):
        return StackingCLR(multiplicities=multiplicities).fit(training,y[fit],task='multiclass',n_classes=10)
    head=wrapper(dfit)
    w=np.zeros((2,10,64))
    for i,(m,model) in enumerate(zip(masks,experts)):
        w[i][:,m]=model.coef_
    bias=np.stack([model.intercept_ for model in experts])
    compiled=head.compile(w[duplicated],bias[duplicated])
    full=head.predict(dtest)
    effective=compiled.predict(x[test])
    pred={'left':q[:,0],'right':q[:,1],'mean_views':q.mean(axis=1),'mean_with_clone':dtest.mean(axis=1),
          'clr_clone':full,'compiled':effective,'refit_no_clone':wrapper(p).predict(q),
          'refit_sqrt2':wrapper(p,[2.,1.]).predict(q),
          'refit_left_only':wrapper(p[:,[0]]).predict(q[:,[0]]),
          'raw_base':logistic(x[base],y[base]).predict_proba(x[test]),
          'raw_union':logistic(x[~test],y[~test]).predict_proba(x[test])}
    trees=[]
    for seed in (31,47,67):
        gp=OKAPIMethod(seed=seed,fitness='brier',generations=12,n_restarts=1,n_seeds=8,
                       population_size=3,tournament_size=3,population_multiplier=2)
        try:
            gp.fit(dfit,y[fit],task='multiclass',n_classes=10)
            pred[f'okapi_{seed}']=gp.predict(dtest)
            trees.append(dict(fold=fold,search_seed=seed,tree=gp.tree_repr(),nodes=int(gp.best_nodes)))
        finally:
            gp._tmp.cleanup()
    risks=[dict(fold=fold,method=key,n_test=int(test.sum()),**score(probs,y[test])) for key,probs in pred.items()]
    checks=dict(fold=fold,n_base=int(base.sum()),n_fusion=int(fit.sum()),n_test=int(test.sum()),
                compile_max_abs=np.max(np.abs(full-effective)),
                multiplicity_max_abs=np.max(np.abs(full-pred['refit_sqrt2'])),
                ordinary_max_abs=np.max(np.abs(full-pred['refit_no_clone'])))
    probes=[]
    for seed in range(12):
        permutation=np.random.default_rng([fold,seed,17]).permutation(len(dtest))
        for key,cols in [('single_copy',[0]),('left_pair',[0,1]),('right',[2]),('exact_twin',[])]:
            swapped=dtest.copy()
            if cols:
                swapped[:,cols]=dtest[permutation][:,cols]
            else:
                swapped[:,0]=dtest[:,1]
            after=head.predict(swapped)
            probes.append(dict(fold=fold,seed=seed,probe=key,
                               total_variation=.5*np.abs(after-full).sum(axis=1).mean(),
                               brier_change=score(after,y[test])['brier']-score(full,y[test])['brier']))
    # Save paired held-out predictions for independent checks without retraining.
    return risks,probes,checks,trees,np.flatnonzero(test),pred


def run(folder):
    from loguru import logger
    logger.disable('okapi')
    start=datetime.now(timezone.utc)
    timer=time.perf_counter()
    folder=Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    x,y=load_digits(return_X_y=True)
    x=x/16.
    fold_id=np.zeros(len(y),dtype=int)
    for index,(_,held) in enumerate(StratifiedKFold(5,shuffle=True,random_state=7813).split(x,y)):
        fold_id[held]=index
    risks,probes,checks,trees=[],[],[],[]
    oof={}
    with threadpool_limits(limits=1):
        for index in range(5):
            r,p,c,t,rows,prob=compute_fold(index,x,y,fold_id)
            risks.extend(r)
            probes.extend(p)
            checks.append(c)
            trees.extend(t)
            for method,values in prob.items():
                if method not in oof:
                    oof[method]=np.zeros((len(y),10))
                oof[method][rows]=values
    pd.DataFrame(risks).to_csv(folder/'folds.csv',index=False)
    pd.DataFrame(probes).to_csv(folder/'perturbations.csv',index=False)
    pd.DataFrame(checks).to_csv(folder/'compilation.csv',index=False)
    pd.DataFrame(trees).to_csv(folder/'trees.csv',index=False)
    pd.DataFrame([dict(method=k,**score(v,y)) for k,v in oof.items()]).to_csv(folder/'scores.csv',index=False)
    np.savez_compressed(folder/'held_out.npz',labels=y,fold=fold_id,**oof)
    (folder/'config.json').write_text(json.dumps(dict(fold_seed=7813,folds=5,logistic_C=1.,
        gp_seeds=[31,47,67],gp_generations=12,compute_start_utc=start.isoformat(),
        compute_seconds=time.perf_counter()-timer),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)

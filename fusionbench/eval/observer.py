"""Compare a selected disagreement bit with all pairwise disagreement moments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fusionbench.eval.seed_selection import disagreement_features, observations, subset_weights


def run_study(bank, output, val_n=256):
    from okapi.selection import estimate_context_prevalence

    bank, output = Path(bank), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((bank/'config.json').read_text())
    if val_n not in cfg['val_ns']:
        raise ValueError('requested validation budget is missing from bank')
    ranks = pd.read_csv(bank/'ranks.csv', float_precision='round_trip').set_index(
        ['data_seed', 'val_n', 'repeat', 'test_p'])
    candidates = pd.read_csv(bank/'candidates.csv', float_precision='round_trip').set_index('data_seed')
    archived = pd.read_csv(bank/'validation.csv', float_precision='round_trip').set_index(
        ['data_seed','val_n','repeat']).sort_index()
    rows = []
    verified = []
    fit_p = cfg['fit_p']
    for data_seed in cfg['data_seeds']:
        ds = candidates.loc[data_seed].set_index('gp_seed').loc[cfg['gp_seeds']]
        for phase, offset in [('replay',0), ('fresh_observations',50000)]:
            for repeat in range(-1, cfg['val_repeats']):
                if repeat == -1 and offset:
                    continue
                if repeat == -1:
                    a, b, _ = observations(data_seed, 2, cfg['eval_n'])
                    moments_a, _ = disagreement_features(a)
                    moments_b, _ = disagreement_features(b)
                else:
                    max_n = max(cfg['val_ns'])
                    a, b, _ = observations(data_seed, offset+100+repeat, max_n)
                    mask = np.random.default_rng([data_seed,13+offset,repeat]).random(max_n) < fit_p
                    src = np.where(mask[:,None,None], b, a)
                    src_moments, _ = disagreement_features(src)
                ta, tb, _ = observations(data_seed, offset+5001+repeat, cfg['target_n'])
                u = np.random.default_rng([data_seed,17+offset,repeat+1]).random(cfg['target_n'])
                for p in cfg['test_grid']:
                    target = np.where((u<p)[:,None,None], tb, ta)
                    t_moments, _ = disagreement_features(target)
                    t_mean = t_moments.mean(0)
                    budgets = [2*cfg['eval_n']] if repeat == -1 else (cfg['val_ns'] if offset==0 else [val_n])
                    for n in budgets:
                        if repeat == -1:
                            clean, changed = moments_a, moments_b
                        else:
                            c = mask[:n]
                            clean, changed = src_moments[:n][~c], src_moments[:n][c]
                        full = estimate_context_prevalence(t_mean,clean,changed,fit_p)
                        if not offset:
                            original = ranks.loc[(data_seed,n,repeat,p),'estimated_test_p']
                            verified.append(abs(full-original))
                        if n != val_n or repeat == -1:
                            continue
                        val = archived.loc[(data_seed,n,repeat)].set_index('gp_seed').loc[cfg['gp_seeds']]
                        # Ties select the first pair in lexicographic order.
                        col = int(np.argmax(abs(changed.mean(0) - clean.mean(0)))) if len(clean) and len(changed) else 0
                        bit = estimate_context_prevalence(t_mean[col:col+1],clean[:,col:col+1],changed[:,col:col+1],fit_p)
                        i,j = np.triu_indices(src.shape[1],1)
                        variants = {'all_pairs':full, 'one_pair':bit}
                        for name, source, dest in [
                                ('single_score',src[:n,0,1],target[:,0,1]),
                                ('single_vote',src[:n,0,1]>.5,target[:,0,1]>.5)]:
                            variants[name] = estimate_context_prevalence(np.array([dest.mean()]),source[~c,None],source[c,None],fit_p)
                        test = ((1-p)*ds.test_clean + p*ds.test_flip).to_numpy()
                        actual = val.val.to_numpy()
                        for model, estimate in {**variants, 'validation': np.nan, 'random_seed': np.nan}.items():
                            criterion = actual
                            if model in variants and val.n_clean.min()>0 and val.n_flip.min()>0:
                                criterion = ((1-estimate)*val.val_clean + estimate*val.val_flip).to_numpy()
                            w = subset_weights(criterion,1 if model=='random_seed' else len(ds))
                            loss = w @ test
                            rows.append(dict(phase=phase,data_seed=data_seed,repeat=repeat,test_p=p,
                                             method=model,estimated_p=estimate,
                                             absolute_error=abs(estimate-p), brier=loss, regret=loss-test.min(),
                                             selected_left=int(i[col]),selected_right=int(j[col])))
    replay_error = float(max(verified))
    if replay_error > 1e-12:
        raise AssertionError(f'Published observer replay error {replay_error} exceeds 1e-12')
    out = pd.DataFrame(rows)
    out.to_csv(output/'observer.csv',index=False)
    metrics=['estimated_p','absolute_error','brier','regret']
    draws=out.groupby(['phase','data_seed','test_p','method'])[metrics].mean()
    draws.to_csv(output/'draws.csv')
    draws.groupby(['phase','test_p','method']).mean().to_csv(output/'means.csv')
    (output/'verification.json').write_text(json.dumps(dict(comparisons=len(verified),max_abs_difference=replay_error,
        val_n=val_n,target_n=cfg['target_n'],independent_feature_offset=50000),indent=2))
    return out


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bank',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--val-n',type=int,default=256)
    args = parser.parse_args()
    run_study(args.bank,args.out,args.val_n)

"""Transport validation rankings of independently evolved OKAPI champions.

Context inverts one expert's probability. Labels, every other expert, and
within-context score distributions are fixed across deployment prevalences.
Evaluation uses common random numbers under the two counterfactual contexts.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, rankdata

from fusionbench.eval.search_stability import _predict_trees, _tensor_map
from fusionbench.methods.okapi import OKAPIMethod, _identity, _brier_objective
from fusionbench.scenarios.calibrated import binary_calibrated


def observations(seed, split, n):
    rng = np.random.default_rng([seed, split])
    y = rng.integers(2, size=n)
    common = rng.normal(size=n)
    strengths = (.95, .85, .82, .79, .77, .73, .68)
    columns = [
        binary_calibrated(y, strength, rng, shared_noise=common, shared_weight=0 if j == 0 else .65)
        for j, strength in enumerate(strengths)
    ]
    positive = np.stack(columns, axis=1)
    clean = np.stack([1 - positive, positive], axis=-1)
    flipped = clean.copy()
    flipped[:, 0] = clean[:, 0, ::-1]
    return clean, flipped, y


def search_champion(train, y, seed, generations):
    from okapi.okapi import Okapi
    from okapi.pareto import maximize

    adapter = OKAPIMethod(n_seeds=8)
    try:
        directory = adapter._write_split(train, 'fit')
        gt = Path(adapter._tmp.name) / 'gt.npy'
        np.save(gt, y)
        paths = [directory / key for key in _tensor_map(train)]
        search = Okapi(paths, gt, population_size=len(paths), population_multiplier=2,
                       tournament_size=3, minimize_node_count=True, objective_functions=(_brier_objective,),
                       objectives=(maximize,), backend='numpy', seed=seed, postprocessing_function=_identity)
        adapter._seed_population(search, np.random.default_rng(seed))
        for _ in range(generations):
            search.run_iteration()
        best = int(np.argmax(search.pareto_fitnesses[:, 0]))
        return search.pareto_trees[best]
    finally:
        adapter._tmp.cleanup()


def subset_weights(values, size):
    """Probability of winning a uniformly drawn k-subset, with uniform tie breaks."""
    n = len(values)
    weights = np.array([math.comb(n-i-1, size-1)/math.comb(n, size) for i in range(n)])
    order = np.argsort(values, kind='stable')
    sorted_values = values[order]
    for start in np.r_[0, np.where(np.diff(sorted_values) != 0)[0]+1]:
        end = np.searchsorted(sorted_values, sorted_values[start], side='right')
        weights[start:end] = weights[start:end].mean()
    return weights[np.argsort(order)]


def weighted_upper_tail(values, weights, mass=.25):
    """Exact mean Brier of worst mass of selected seeds (fractional boundary)."""
    order = np.argsort(-values)
    w = weights[order]
    before = np.cumsum(w) - w
    return np.dot(values[order], np.clip(mass - before, 0, w)) / mass


def pair_order(scores):
    rows, cols = np.triu_indices(len(scores), 1)
    return np.sign(scores[rows] - scores[cols])


def correlations(a, b):
    a_rank, b_rank = rankdata(a), rankdata(b)
    if max(a_rank) == min(a_rank) or max(b_rank) == min(b_rank):
        return np.nan, np.nan
    return np.corrcoef(a_rank, b_rank)[0, 1], kendalltau(a, b).statistic


def disagreement_features(scores):
    """Pairwise disagreement moments and polarity-invariant vote patterns."""
    votes = scores[:, :, 1] > .5
    a, b = np.triu_indices(votes.shape[1], 1)
    disagreements = votes[:, a] != votes[:, b]
    pattern = (votes[:, 1:] != votes[:, :1]) @ (2 ** np.arange(votes.shape[1]-1))
    return disagreements, pattern


def vote_pattern_risk(losses, patterns, target_probabilities, prior_strength=4):
    """Direct standardization on observed vote patterns, with sparse-bin shrinkage."""
    global_mean = losses.mean(1)
    conditional = np.broadcast_to(global_mean, (len(target_probabilities), len(global_mean))).copy()
    for pattern in np.unique(patterns):
        mask = patterns == pattern
        conditional[pattern] = (losses[:, mask].sum(1) + prior_strength*global_mean) / (mask.sum()+prior_strength)
    return target_probabilities @ conditional


def run_study(output, data_seeds, gp_seeds=range(12), generations=10, fit_n=384,
              eval_n=8192, val_ns=(64, 256, 1024), val_repeats=12, target_n=512,
              fit_p=.10, test_grid=(0., .1, .25, .5, .75, 1.), robust_max=.50):
    from okapi.selection import estimate_context_prevalence

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    gp_seeds, data_seeds, val_ns = list(gp_seeds), list(data_seeds), list(val_ns)
    if not data_seeds or len(set(gp_seeds)) != len(gp_seeds) or len(gp_seeds) < 4:
        raise ValueError("need data and at least four distinct GP seeds")
    if not (0 < fit_p < 1 and fit_p < robust_max <= 1):
        raise ValueError("use a source prevalence and a larger robust endpoint in [0,1]")
    config = dict(data_seeds=data_seeds, gp_seeds=gp_seeds, generations=generations, fit_n=fit_n,
                  eval_n=eval_n, val_ns=val_ns, val_repeats=val_repeats,
                  fit_p=fit_p, test_grid=test_grid, robust_max=robust_max, target_n=target_n,
                  vote_pattern_prior_strength=4)
    (output / 'config.json').write_text(json.dumps(config, indent=2))
    candidates, scores, ranks, selectors = [], [], [], []
    for data_seed in data_seeds:
        clean, changed, y = observations(data_seed, 0, fit_n)
        ctx = np.random.default_rng([data_seed, 11]).random(fit_n) < fit_p
        train = np.where(ctx[:, None, None], changed, clean)
        bank = [search_champion(train, y, seed, generations) for seed in gp_seeds]

        def losses(split, n):
            a, b, y = observations(data_seed, split, n)
            pa, pb = _predict_trees(bank, a), _predict_trees(bank, b)
            return (pa-y)**2, (pb-y)**2, np.mean(abs(pb-pa), axis=1), a, b

        t0, t1, sensitivity, _, _ = losses(1, eval_n)
        lower, upper = t0.mean(1), t1.mean(1)
        v0, v1, val_sensitivity, ref_a, ref_b = losses(2, eval_n)
        large0, large1 = v0.mean(1), v1.mean(1)
        for i, tree in enumerate(bank):
            nodes = [v.id for v in tree.nodes['value_nodes']]
            candidates.append(dict(data_seed=data_seed, gp_seed=gp_seeds[i],
                                   source_fit_flips=int(ctx.sum()), tree=repr(tree), nodes=tree.nodes_count,
                                   shortcut_count=nodes.count('m0.npy'), delta_probability=sensitivity[i],
                                   probe_delta_probability=val_sensitivity[i],
                                   test_clean=lower[i], test_flip=upper[i],
                                   large_val_clean=large0[i], large_val_flip=large1[i]))

        # Same independent target feature batch for all validation budgets.
        # Context is used only by the simulator to draw the target observations.
        def target_features(repeat):
            a, b, _ = observations(data_seed, 5000 + repeat + 1, target_n)
            u = np.random.default_rng([data_seed, 17, repeat+1]).random(target_n)
            features = {}
            for p in test_grid:
                target = np.where((u < p)[:,None,None], b, a)
                moments, patterns = disagreement_features(target)
                features[p] = moments.mean(0), np.bincount(patterns, minlength=2**(a.shape[1]-1))/len(patterns)
            return features

        def audit(n, repeat, l0, l1, actual_source, n0, n1, features, clean_mom, flip_mom,
                  val_pattern, val_loss):
            # The robust selector knows context labels and a declared interval,
            # but sees the same observed validation predictions/labels as ERM.
            estimated_source = (1-fit_p)*l0 + fit_p*l1
            informed = min(n0, n1) > 0
            robust = np.maximum(estimated_source, (1-robust_max)*l0 + robust_max*l1) if informed else actual_source
            for i in range(len(bank)):
                scores.append(dict(data_seed=data_seed,gp_seed=gp_seeds[i],val_n=n,repeat=repeat,
                                   n_clean=n0,n_flip=n1,val_clean=l0[i],val_flip=l1[i],val=actual_source[i]))
            for p in test_grid:
                test = (1-p)*lower + p*upper
                calibrated = (1-p)*l0 + p*l1 if informed else actual_source
                target_mom, target_pattern = features[p]
                p_hat = estimate_context_prevalence(target_mom, clean_mom, flip_mom, fit_p)
                adapted = (1-p_hat)*l0 + p_hat*l1 if informed else actual_source
                pattern_score = vote_pattern_risk(val_loss, val_pattern, target_pattern)
                rs, tau = correlations(actual_source, test)
                ref_source = (1-fit_p)*large0 + fit_p*large1
                ref_order, true_order = pair_order(ref_source), pair_order(test)
                v_order, standardized_order = pair_order(actual_source), pair_order(calibrated)
                comparable = (ref_order != 0) & (true_order != 0)
                finite = (v_order != 0) & (true_order != 0)
                m = (standardized_order != 0) & (true_order != 0)
                ranks.append(dict(data_seed=data_seed,val_n=n,repeat=repeat,test_p=p,
                                  spearman=rs,kendall=tau,estimated_test_p=p_hat,
                                  adapted_spearman=correlations(adapted,test)[0],
                                  pattern_spearman=correlations(pattern_score,test)[0],
                                  source_to_target_inversions=np.mean(ref_order[comparable]!=true_order[comparable]) if comparable.any() else np.nan,
                                  observed_inversions=np.mean(v_order[finite]!=true_order[finite]) if finite.any() else np.nan,
                                  adjusted_inversions=np.mean(standardized_order[m]!=true_order[m]) if m.any() else np.nan))
                for k in (1,4,len(bank)):
                    for name, criterion in [('validation',actual_source),('robust',robust),
                                            ('estimated_prevalence',adapted),('vote_pattern',pattern_score),
                                            ('known_prevalence',calibrated),('hindsight',test)]:
                        w = subset_weights(criterion, k)
                        expected = w @ test
                        selectors.append(dict(data_seed=data_seed,val_n=n,repeat=repeat,test_p=p,k=k,selector=name,
                                              brier=expected,regret=expected-test.min(),
                                              worst_tail_brier=weighted_upper_tail(test,w),
                                              prob_at_or_above_bank_p75_brier=w[test>=np.quantile(test,.75)].sum()))

        # Large reference uses paired, equally sized conditional MC banks. Its
        # empirical score is integrated with the source prevalence. For the
        # unconditional pattern baseline, emulate the same source mixture.
        ref_m0, ref_pat0 = disagreement_features(ref_a)
        ref_m1, ref_pat1 = disagreement_features(ref_b)
        ref_mask = np.random.default_rng([data_seed, 16]).random(eval_n) < fit_p
        audit(2*eval_n, -1, large0, large1, (1-fit_p)*large0+fit_p*large1, eval_n, eval_n,
              target_features(-1), ref_m0, ref_m1, np.where(ref_mask, ref_pat1, ref_pat0),
              np.where(ref_mask[None,:], v1, v0))
        for repeat in range(val_repeats):
            val0, val1, _, a_scores, b_scores = losses(100+repeat, max(val_ns))
            features = target_features(repeat)
            context = np.random.default_rng([data_seed, 13, repeat]).random(max(val_ns)) < fit_p
            moments, patterns = disagreement_features(np.where(context[:,None,None], b_scores, a_scores))
            observed_losses = np.where(context[None,:], val1, val0)
            for n in val_ns:
                mask = context[:n]
                # No extrapolation when a context is absent. The selector falls
                # back to empirical validation and the missing cell is recorded.
                a = val0[:, :n][:, ~mask]
                b = val1[:, :n][:, mask]
                mu_a = a.mean(1) if a.shape[1] else np.full(len(bank), np.nan)
                mu_b = b.mean(1) if b.shape[1] else np.full(len(bank), np.nan)
                audit(n, repeat, mu_a, mu_b, (a.sum(1)+b.sum(1))/n, a.shape[1], b.shape[1],
                      features, moments[:n][~mask], moments[:n][mask], patterns[:n], observed_losses[:,:n])
        for fname, rows in [('candidates',candidates),('validation',scores),('ranks',ranks),('selection',selectors)]:
            pd.DataFrame(rows).to_csv(output/f'{fname}.csv', index=False)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--data-seeds',type=int,nargs='+',default=[100,101,102,103])
    parser.add_argument('--gp-seeds',type=int,nargs='+',default=list(range(12)))
    parser.add_argument('--generations',type=int,default=10)
    args=parser.parse_args()
    run_study(args.out, args.data_seeds, args.gp_seeds, args.generations)

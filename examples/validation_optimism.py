"""Does enlarging the GP menu improve independent risk and bias its validation score?

Cawley and Talbot (2010) studied bias from optimizing a noisy selection
criterion. Here the same validation-selected tree is rescored on fresh cases
to separate optimism on the selection set from the gain on an independent test.
See cawley2010overfitting in docs/ensemble-study/literature/README.md.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.stats import t
from threadpoolctl import threadpool_limits

from fusionbench.eval.search_stability import _predict_trees
from fusionbench.eval.seed_selection import observations, search_champion
from fusionbench.methods.averaging import SimpleAverage, TopN
from fusionbench.methods.okapi import _brier_objective
from fusionbench.methods.stacking import StackingLR, StackingLogit


def permuted_winners(panel, seed, repeat, n, candidates, a, b, fit, iid, shift):
    order = np.random.default_rng([99041, seed, repeat]).permutation(len(a))
    first = order[0]
    rows = []
    for menu in ("trees", "copies"):
        for k in (1, 4, 12):
            prefix = order[:k] if menu == "trees" else np.repeat(first, k)
            winner = prefix[np.argmin(a[prefix])]
            rows.append(dict(panel=panel, data_seed=seed, repeat=repeat, n=n, menu=menu, k=k,
                             first_seed=candidates[first], winner_seed=candidates[winner],
                             fit=fit[winner], a=a[winner], b=b[winner], iid=iid[winner], shift=shift[winner],
                             optimism=b[winner]-a[winner],
                             excess=(b[winner]-a[winner])-(b[first]-a[first]),
                             apparent=a[first]-a[winner], gain=iid[first]-iid[winner],
                             shift_gain=shift[first]-shift[winner]))
    return rows


def archive(zipfile):
    rows = []
    with ZipFile(zipfile) as z:
        for phase in ("development", "confirmation"):
            path = "seed_selection/"+phase+"/"
            with z.open(path+"validation.csv") as f:
                val = pd.read_csv(f)
            with z.open(path+"candidates.csv") as f:
                raw = pd.read_csv(f)
            for ds, frame in raw.groupby("data_seed"):
                raw_bank = frame.sort_values("gp_seed")
                clean, flipped = raw_bank.test_clean.to_numpy(), raw_bank.test_flip.to_numpy()
                ids = raw_bank.gp_seed.to_numpy()
                scores = val[val.data_seed == ds].pivot(index=["val_n", "repeat"], columns="gp_seed", values="val")[ids]
                for rep in range(6):
                    for n in (64, 256):
                        rows.extend(permuted_winners("archive", ds, rep, n, ids,
                                                    scores.loc[n, 2*rep].to_numpy(),
                                                    scores.loc[n, 2*rep+1].to_numpy(),
                                                    np.full(12, np.nan), .9*clean+.1*flipped,
                                                    .5*clean+.5*flipped))
    return rows


def mixture(ds, split, mask_seed, n):
    a, b, y = observations(ds, split, n)
    mask = np.random.default_rng([ds, mask_seed]).random(n) < .1
    return np.where(mask[:, None, None], b, a), y


def models(preds, y, include_mean):
    bank = {"top_n": TopN("brier"), "stack_p": StackingLR(), "stack_logit": StackingLogit()}
    if include_mean:
        bank["mean"] = SimpleAverage()
    for head in bank.values():
        head.fit(preds, y, task="binary", n_classes=2)
    return bank


def squared(predict, inputs):
    preds, y = inputs
    return np.square(predict(preds)-y)


def scores(predict, splits):
    fit, a, b, clean, flipped = (squared(predict, batch).mean(axis=-1) for batch in splits)
    return fit, a, b, .9*clean+.1*flipped, .5*clean+.5*flipped


def study(ds):
    train = mixture(ds, 0, 11, 384)
    test_clean, test_flipped, yt = observations(ds, 900, 8192)
    tests = [(test_clean, yt), (test_flipped, yt)]
    trees, metadata = [], []
    for gp_seed in range(820, 832):
        calls = 0

        def count(p, y):
            nonlocal calls
            calls += 1
            return _brier_objective(p, y)

        start = perf_counter()
        tree = search_champion(*train, gp_seed, 10, objective=count)
        metadata.append(dict(data_seed=ds, gp_seed=gp_seed, nodes=tree.nodes_count,
                             calls=calls, seconds=perf_counter()-start, tree=repr(tree)))
        trees.append(tree)
    ids = np.arange(820, 832)
    def predict_trees(p):
        return _predict_trees(trees, p)
    fit, clean, flipped = (squared(predict_trees, x).mean(1) for x in [train, *tests])
    iid, shift = .9*clean+.1*flipped, .5*clean+.5*flipped
    refs = models(*train, True)
    fixed = {}
    for name, model in refs.items():
        predict = lambda p: model.predict(p)[:, 1]
        rf, rc, rs = (squared(predict, x).mean() for x in [train, *tests])
        fixed[name] = (rf, .9*rc+.1*rs, .5*rc+.5*rs)
    decisions, baselines, candidate_scores = [], [], []
    for rep in range(12):
        a = mixture(ds, 1000+2*rep, 2000+2*rep, 256)
        b = mixture(ds, 1001+2*rep, 2001+2*rep, 256)
        a_loss, b_loss = squared(predict_trees, a), squared(predict_trees, b)
        for n in (64, 256):
            ar, br = a_loss[:, :n].mean(1), b_loss[:, :n].mean(1)
            for i, gp in enumerate(ids):
                candidate_scores.append(dict(data_seed=ds, repeat=rep, n=n, gp_seed=gp,
                                             fit=fit[i], a=ar[i], b=br[i], iid=iid[i], shift=shift[i]))
            decisions.extend(permuted_winners("new", ds, rep, n, ids, ar, br, fit, iid, shift))
            ax = tuple(x[:n] for x in a)
            bx = tuple(x[:n] for x in b)
            refits = models(np.concatenate((train[0], ax[0])), np.concatenate((train[1], ax[1])), False)
            for policy, methods in (("fit", refs), ("fit_plus_A", refits)):
                for name, model in methods.items():
                    predict = lambda p: model.predict(p)[:, 1]
                    if policy == "fit":
                        rf, ri, rs = fixed[name]
                        ra, rb = (squared(predict, x).mean() for x in (ax, bx))
                    else:
                        rf, ra, rb, ri, rs = scores(predict, [train, ax, bx, *tests])
                    baselines.append(dict(data_seed=ds, repeat=rep, n=n, policy=policy, method=name,
                                          fit=rf, a=ra, b=rb, iid=ri, shift=rs,
                                          n_selected=getattr(model, "n_selected_", None)))
    return decisions, baselines, metadata, candidate_scores


def summarize(rows):
    metrics = ["fit", "a", "b", "iid", "shift", "optimism", "excess", "apparent", "gain", "shift_gain"]
    bank = rows.groupby(["panel", "n", "menu", "k", "data_seed"])[metrics].mean().reset_index()
    result = []
    for keys, group in bank.groupby(["panel", "n", "menu", "k"]):
        for metric in metrics:
            x = group[metric].dropna()
            if not len(x):
                continue
            half = t.ppf(.975, len(x)-1)*x.std(ddof=1)/np.sqrt(len(x))
            result.append(dict(zip(["panel", "n", "menu", "k"], keys), metric=metric, banks=len(x),
                               mean=x.mean(), lower=x.mean()-half, upper=x.mean()+half))
    return pd.DataFrame(result)


def run(archive_path, out):
    start = perf_counter()
    decisions = archive(archive_path)
    baselines, metadata, candidate_scores = [], [], []
    for ds in range(98000, 98016):
        rd, rb, rm, rc = study(ds)
        decisions.extend(rd)
        baselines.extend(rb)
        metadata.extend(rm)
        candidate_scores.extend(rc)
    out.mkdir(parents=True, exist_ok=True)
    data = pd.DataFrame(decisions)
    copies = data[data.menu == "copies"]
    assert (copies.first_seed == copies.winner_seed).all()
    assert (copies[["excess", "apparent", "gain", "shift_gain"]] == 0).all().all()
    assert (data.apparent >= 0).all()
    for name, frame in (("decisions", data), ("baselines", pd.DataFrame(baselines)),
                        ("candidates", pd.DataFrame(candidate_scores))):
        frame.to_csv(out/(name+".csv.gz"), index=False, compression={"method":"gzip","mtime":0})
    pd.DataFrame(metadata).to_csv(out/"search.csv", index=False)
    summarize(data).to_csv(out/"estimates.csv", index=False)
    pd.DataFrame(baselines).groupby(["n", "policy", "method"])[["fit", "a", "b", "iid", "shift"]].mean().to_csv(out/"baseline_means.csv")
    (out/"timing.json").write_text(json.dumps({"seconds":perf_counter()-start}, indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("docs/ensemble-study/results.zip"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.archive, args.out)

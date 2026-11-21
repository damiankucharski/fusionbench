"""Require independent paired loss evidence before deploying a searched tree."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.special import expit, logit, logsumexp
from threadpoolctl import threadpool_limits

from fusionbench.methods.averaging import SimpleAverage
from fusionbench.methods.okapi import OKAPIMethod
from fusionbench.methods.stacking import StackingLogit


def log_evidence(gain):
    """Log e-value for a single frozen nominee pair, given IID bounded gains."""
    d = np.asarray(gain, dtype=float)
    if d.ndim != 1 or not len(d) or not np.isfinite(d).all() or np.max(np.abs(d)) > 1:
        raise ValueError("paired gains must be a nonempty vector in [-1,1]")
    with np.errstate(divide="ignore"):
        logs = [np.log1p(stake * d).sum() for stake in (.25, .5, 1.)]
    return float(logsumexp(logs) - np.log(len(logs)))


def draw(n, seed, split, w):
    x_rng, y_rng = [np.random.default_rng(s) for s in
                    np.random.SeedSequence([seed, split]).spawn(2)]
    channels = x_rng.uniform(.02, .98, (n, 3))
    p = w * channels.max(axis=1) + (1 - w) * expit(logit(channels) @ [.5, .75, 1])
    preds = np.stack([1-channels, channels], axis=-1)
    return preds, (y_rng.random(n) < p).astype(int), p


def nominate(losses, indices, start, stop):
    return min(indices, key=lambda i: losses[start:stop, i].mean())


def run(out, data_seeds=range(4100, 4112)):
    out = Path(out)
    out.mkdir(exist_ok=True, parents=True)
    start = time.perf_counter()
    policies, candidates = [], []
    times = []
    with threadpool_limits(limits=1):
        for data_seed in data_seeds:
            for w in (0., .5, 1.):
                sets = [draw(n, data_seed, ix, weight)
                        for ix, (n, weight) in enumerate([(256, w), (512, w), (8192, w), (8192, 1-w)])]
                x, y, _ = sets[0]
                stacker = StackingLogit().fit(x, y, task="binary", n_classes=2)
                baselines = [stacker, SimpleAverage()]
                refits = {}
                for n in (32, 128, 512):
                    xv, yv, _ = sets[1]
                    refits[n] = StackingLogit().fit(np.concatenate([x,xv[:n]]), np.r_[y,yv[:n]],
                                                    task="binary", n_classes=2)
                for bank in range(2):
                    models = list(baselines)
                    names = ["stacker", "mean"]
                    for search_seed in range(700+4*bank, 704+4*bank):
                        gp = OKAPIMethod(fitness="brier", seed=search_seed, n_restarts=1,
                                         generations=16, n_seeds=10, population_multiplier=2, max_nodes=11)
                        tic = time.perf_counter()
                        gp.fit(x, y, task="binary", n_classes=2)
                        times.append(dict(data_seed=data_seed, w=w, bank=bank, gp_seed=search_seed,
                                          seconds=time.perf_counter()-tic, nodes=gp.best_nodes))
                        models.append(gp)
                        names.append(f"gp{search_seed}")
                    predictions = [np.column_stack([m.predict(inputs)[:,1] for m in models])
                                   for inputs, _, _ in sets]
                    obs = [(p - labels[:, None]) ** 2 for p, (_, labels, _) in zip(predictions, sets)]
                    # Conditional risk retains irreducible Bernoulli uncertainty.
                    conditional = [(p - truth[:, None])**2 + (truth*(1-truth))[:,None]
                                   for p, (_, _, truth) in zip(predictions, sets)]
                    for n in (32, 128, 512):
                        half = n // 2
                        base = nominate(obs[1], range(2), 0, half)
                        tree = nominate(obs[1], range(2,6), 0, half)
                        diff = obs[1][half:n, base] - obs[1][half:n, tree]
                        log_e = log_evidence(diff)
                        choices = dict(stacker=0, mean=1, baseline_nominee=base, unconditional_gp=tree,
                                       naive_full=nominate(obs[1], range(6), 0, n),
                                       naive_half=nominate(obs[1], range(6), 0, half),
                                       pooled_nominees=nominate(obs[1], (base, tree), 0, n),
                                       gate_half_min=nominate(obs[1], (base, tree), half, n),
                                       independent_gate=tree if log_e >= np.log(20) else base)
                        for domain, ix in (("iid",2), ("shift",3)):
                            risks = conditional[ix].mean(axis=0)
                            empirical = obs[ix].mean(axis=0)
                            fields = dict(w=w, seed=data_seed, bank=bank, n=n, domain=domain)
                            for i, model in enumerate(names):
                                candidates.append(dict(**fields, model=model, fit=obs[0][:,i].mean(),
                                                       validation=obs[1][:n,i].mean(), risk=risks[i],
                                                       realized=empirical[i]))
                            for policy, i in choices.items():
                                policies.append(dict(**fields, policy=policy, chosen=names[i],
                                                     baseline=names[base], tree=names[tree], log_e=log_e,
                                                     adopted=i>=2, nominee_gain=risks[base]-risks[tree],
                                                     gain_vs_base=risks[base]-risks[i],
                                                     gain_vs_stack=risks[0]-risks[i], risk=risks[i],
                                                     realized=empirical[i], fit=obs[0][:,i].mean(),
                                                     validation=obs[1][:n,i].mean()))
                            xt, yt, pt = sets[ix]
                            pred = refits[n].predict(xt)[:,1]
                            r = np.mean((pred-pt)**2 + pt*(1-pt))
                            policies.append(dict(**fields, policy="stacker_refit_all", chosen="stacker_refit",
                                                 baseline=names[base], tree=names[tree], log_e=log_e, adopted=False,
                                                 nominee_gain=risks[base]-risks[tree], gain_vs_base=risks[base]-r,
                                                 gain_vs_stack=risks[0]-r, risk=r, realized=np.mean((pred-yt)**2)))
                    for m in models[2:]:
                        m._tmp.cleanup()

    records = pd.DataFrame(policies)
    records["false_adoption"] = records.adopted & (records.gain_vs_base < 0)
    records["missed_beneficial_nominee"] = ~records.adopted & (records.nominee_gain > 0)
    summary = records.groupby(["w","n","domain","policy"],as_index=False).agg(
        decisions=("chosen","size"), adopted=("adopted","sum"), harmful_adoptions=("false_adoption","sum"),
        missed=("missed_beneficial_nominee","sum"), beneficial_nominees=("nominee_gain",lambda g:(g>0).sum()),
        risk=("risk","mean"), realized=("realized","mean"), gain_vs_base=("gain_vs_base","mean"),
        gain_vs_stack=("gain_vs_stack","mean"))
    # Paired policy contrast with 12 independent data seeds as sampling units.
    panel = records.groupby(["w","n","domain","seed","policy"]).risk.mean().unstack()
    contrast = panel.naive_full - panel.independent_gate
    paired = contrast.groupby(["w","n","domain"]).agg(["mean","std","count"]).reset_index()
    from scipy.stats import t
    halfwidth = t.ppf(.975, paired["count"]-1) * paired["std"] / np.sqrt(paired["count"])
    paired["lo"], paired["hi"] = paired["mean"]-halfwidth, paired["mean"]+halfwidth
    for filename, df in (("policies",records),("candidates",pd.DataFrame(candidates)),
                         ("search",pd.DataFrame(times)),("summary",summary),("paired",paired)):
        df.to_csv(out / f"{filename}.csv",index=False)
    (out / "timing.json").write_text(json.dumps({"seconds":time.perf_counter()-start,
                                                 "searches":len(times)},indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    run(parser.parse_args().out)

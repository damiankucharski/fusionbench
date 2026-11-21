"""Prospective, score-blind family scheduling for one native OKAPI append."""
import argparse
from collections import Counter
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.stats import t
from threadpoolctl import threadpool_limits

from fusionbench.methods.averaging import SimpleAverage, TopN
from fusionbench.methods.stacking import StackingLR, StackingLogit
from fusionbench.scenarios.native_switch import conditional_probability, draw_native_switch
from okapi.mutation import append_new_node_mutation
from okapi.node import ValueNode
from okapi.operators import CLOSE_THRESHOLD, FAR_THRESHOLD, MAX, MEAN, MIN, WEIGHTED_MEAN
from okapi.random_state import use_rng
from okapi.tree import Tree


OPS = (MEAN, MIN, MAX, WEIGHTED_MEAN, FAR_THRESHOLD, CLOSE_THRESHOLD)
NAMES = ("mean", "min", "max", "weighted_mean", "far", "close")
EPS = 1e-12


def observed(seed, law):
    parts = []
    for idx, n in enumerate((256, 256, 4096, 4096)):
        d = draw_native_switch(seed, idx, n)
        p = d.preds[:, :, 1]
        if idx == 3:
            p = np.minimum(p, draw_native_switch(seed, 4, n).preds[:, :, 1])
        a, b = p[:, d.parent], p[:, d.child]
        if law == "min":
            eta = np.minimum(a, b)
        else:
            eta = conditional_probability(a, b, float(law == "logistic"))
        parts.append((p, (d.uniform < eta).astype(int), eta))
    return parts


def objective(p, target):
    return float(np.square(p-target).mean())


def objective_set(output, parts):
    row = {}
    pos = 0
    for name, (_, y, eta) in zip(("validation", "iid", "shift"), parts):
        p = output[pos:pos+len(y)]
        row[name] = objective(p, y)
        row[name+"_excess"] = objective(p, eta)
        pos += len(y)
    assert pos == len(output)
    return row


def schedules(seed, rep, root_id):
    rng = np.random.default_rng([seed, rep, 310])
    with_replacement = rng.integers(6, size=6)
    balanced = rng.permutation(6)
    partners = rng.choice(np.delete(np.arange(6), root_id), size=6)
    fixed = [partners[0]]*6
    return (("replacement", with_replacement, fixed), ("balanced", balanced, fixed),
            ("vary_companion", with_replacement, partners))


def two_channels(p):
    return np.stack((1-p, p), axis=-1)


def grow(parent, p, operand, family, seed):
    with use_rng(np.random.RandomState(seed)):
        child = append_new_node_mutation(parent, [p[:, operand]], ids=[operand], allowed_ops=(OPS[family],))
    op = child.root.children[0]
    assert (type(op) is OPS[family] and child.nodes_count == 3 and child.root.id == parent.root.id
            and op.children[0].id == operand and parent.nodes_count == 1)
    parameter = op.threshold if hasattr(op, "threshold") else getattr(op, "weights", None)
    return child, json.dumps(None if parameter is None else np.asarray(parameter).tolist())


def run_bank(seed, law):
    data = observed(seed, law)
    p, y, _ = data[0]
    root_id = int(np.argmin(np.square(p-y[:, None]).mean(0)))
    parent = Tree(ValueNode(None, p[:, root_id], root_id))
    base_fit = objective(parent.predict().ravel(), y)
    eval_p = np.concatenate([part[0] for part in data[1:]])
    tensors = dict(enumerate(eval_p.T))
    parent_scores = objective_set(eval_p[:, root_id], data[1:])
    refs = []
    for name, model in (("all_mean", SimpleAverage()), ("top_n", TopN("brier")),
                        ("stack_p", StackingLR()), ("stack_logit", StackingLogit())):
        model.fit(two_channels(p), y, task="binary", n_classes=2)
        output = model.predict(two_channels(eval_p))[:, 1]
        refs.append(dict(seed=seed, law=law, method=name, n_selected=getattr(model, "n_selected_", None),
                         fit=objective(model.predict(two_channels(p))[:, 1], y),
                         **objective_set(output, data[1:])))
    refs.append(dict(seed=seed, law=law, method="parent", fit=base_fit, **parent_scores))
    candidates, menus = [], []
    for rep in range(8):
        for policy, families, operands in schedules(seed, rep, root_id):
            best, best_fit = None, base_fit
            occurrences = Counter()
            proposals = []
            for slot, (family, operand) in enumerate(zip(families, operands)):
                family, operand = int(family), int(operand)
                occurrence = occurrences[family, operand]
                occurrences[family, operand] += 1
                key = [seed, rep, family, operand, occurrence, 311]
                factory_seed = int(np.random.SeedSequence(key).generate_state(1)[0])
                child, param = grow(parent, p, operand, family, factory_seed)
                loss = objective(child.predict().ravel(), y)
                rec = dict(seed=seed, law=law, repeat=rep, policy=policy, slot=slot, root=root_id,
                           operand=operand, family=NAMES[family], factory_seed=factory_seed,
                           parameter=param, fit=loss)
                proposals.append((child, rec))
                if loss < best_fit-EPS:
                    best, best_fit = slot, loss
            # Freeze the winner. No independent scores enter the choice.
            for child, rec in proposals:
                vals = child.do_pred_on_another_tensors(current_tensors=tensors).ravel()
                rec.update(objective_set(vals, data[1:]))
                candidates.append(rec)
            selected = parent_scores if best is None else proposals[best][1]
            iid_gain = parent_scores["iid_excess"]-selected["iid_excess"]
            menus.append(dict(seed=seed, law=law, repeat=rep, policy=policy, root=root_id,
                              n_families=len(set(families)), n_operands=len(set(operands)),
                              n_distinct_fits=len(set(r["fit"] for c, r in proposals)),
                              fit_improver=any(r["fit"]<base_fit-EPS for c, r in proposals),
                              iid_improver=any(r["iid_excess"]<parent_scores["iid_excess"]-EPS for c, r in proposals),
                              deploy_improves=iid_gain>EPS, iid_gain=iid_gain, fit_gain=base_fit-best_fit,
                              chosen_family="parent" if best is None else proposals[best][1]["family"],
                              slot=-1 if best is None else best, fit=best_fit,
                              **{key:selected[key] for key in parent_scores}))
    return menus, candidates, refs


def compare(menus):
    metrics = ["iid_improver", "iid_gain", "fit_improver", "fit_gain", "deploy_improves", "shift_excess"]
    differences = []
    for law in ("all", "switch", "logistic", "min"):
        subset = menus if law == "all" else menus[menus.law == law]
        bank = subset.groupby(["seed", "policy"])[metrics].mean()
        for alternative in ("balanced", "vary_companion"):
            by_seed = bank.xs(alternative, level="policy")-bank.xs("replacement", level="policy")
            for metric, values in by_seed.items():
                width = t.ppf(.975, len(values)-1)*values.std(ddof=1)/np.sqrt(len(values))
                differences.append(dict(law=law, alternative=alternative, metric=metric, banks=len(values),
                                        difference=values.mean(), lower=values.mean()-width, upper=values.mean()+width))
    return pd.DataFrame(differences)


def run(out):
    start = perf_counter()
    menus, candidates, references = [], [], []
    for seed in range(118300, 118324):
        for law in ("switch", "logistic", "min"):
            m, c, r = run_bank(seed, law)
            menus.extend(m)
            candidates.extend(c)
            references.extend(r)
    out.mkdir(parents=True, exist_ok=True)
    menus = pd.DataFrame(menus)
    for name, frame in (("menus", menus), ("candidates", pd.DataFrame(candidates)),
                        ("baselines", pd.DataFrame(references))):
        frame.to_csv(out/(name+".csv.gz"), index=False, compression={"method":"gzip", "mtime":0})
    compare(menus).to_csv(out/"contrasts.csv", index=False)
    menus.groupby(["law", "policy"]).mean(numeric_only=True).drop(columns=["seed", "repeat", "root", "slot"]).to_csv(out/"means.csv")
    (out/"cost.json").write_text(json.dumps(dict(seconds=perf_counter()-start, fit_scores=len(candidates)), indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    with threadpool_limits(limits=1):
        run(parser.parse_args().out)

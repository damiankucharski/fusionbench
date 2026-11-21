"""Recompute menu decisions from all individual native append measurements."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t

out = Path(__file__).parent/"results"
c = pd.read_csv(out/"candidates.csv.gz", float_precision="round_trip")
m = pd.read_csv(out/"menus.csv.gz", float_precision="round_trip")
refs = pd.read_csv(out/"baselines.csv.gz", float_precision="round_trip")
parent = refs[refs.method=="parent"].set_index(["seed","law"])
all_ops = {"mean","min","max","weighted_mean","close","far"}
assert len(c)==24*3*8*3*6 and len(m)*6==len(c)
keys = ["seed","law","repeat","policy"]
expected = m.set_index(keys)
outcomes = ["fit", "validation", "iid", "iid_excess", "shift", "shift_excess"]
for indices, gp in c.groupby(keys):
    seed, law, rep, policy = indices
    p = parent.loc[seed, law]
    assert gp.slot.to_list() == list(range(6))
    assert gp.root.nunique()==1 and (gp.root!=gp.operand).all()
    if policy=="balanced":
        assert set(gp.family)==all_ops
    if policy!="vary_companion":
        assert gp.operand.nunique()==1
    best, slot = p, -1
    for child in gp.itertuples():
        if child.fit < best.fit-1e-12:
            best, slot = gp.loc[gp.slot==child.slot].iloc[0], child.slot
    summary = expected.loc[indices]
    assert slot == summary.slot
    assert bool((gp.iid_excess < p.iid_excess-1e-12).any())==summary.iid_improver
    np.testing.assert_array_equal(summary[outcomes].astype(float),best[outcomes].astype(float))
# Matched family/operand/occurrence uses exactly the same parameter draw and risks.
for col in outcomes:
    assert c.groupby(["seed","law","repeat","operand","family","factory_seed"])[col].nunique().max()==1
table = pd.read_csv(out/"contrasts.csv").set_index(["law","alternative","metric"])
for metric in ("iid_improver","iid_gain"):
    by_bank = m.groupby(["seed","policy"])[metric].mean().unstack()
    delta = by_bank.balanced-by_bank.replacement
    half = t.ppf(.975,23)*delta.std(ddof=1)/np.sqrt(24)
    np.testing.assert_allclose(table.loc[("all","balanced",metric),["difference","lower","upper"]],
                               [delta.mean(),delta.mean()-half,delta.mean()+half])
print(f"Verified {len(c)} candidates, {len(m)} winners, and both primary paired intervals.")
# Descriptive support coverage, computed after the selections are frozen.
roles = {s: set(np.random.default_rng([s, 15]).permutation(6)[:2]) for s in c.seed.unique()}
c["signal_pair"] = [{root, operand} == roles[seed] for seed, root, operand in zip(c.seed, c.root, c.operand)]
print(c.groupby(keys).signal_pair.max().groupby("policy").mean().rename("unordered signal pair present"))

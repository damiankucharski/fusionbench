"""Check fresh winner identities and published contrasts without rerunning the GP."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

root = Path(__file__).parent / "results"
raw = pd.read_csv(root/"candidates.csv.gz", float_precision="round_trip")
decisions = pd.read_csv(root/"decisions.csv.gz", float_precision="round_trip")
summary = pd.read_csv(root/"estimates.csv", float_precision="round_trip")
baseline = pd.read_csv(root/"baselines.csv.gz", float_precision="round_trip")
archive = decisions[decisions.panel == "archive"]
new = decisions[decisions.panel == "new"]
assert (len(archive), len(new), len(raw), len(baseline)) == (16*6*2*6, 16*12*2*6, 16*12*2*12, 16*12*2*7)
scores = ["fit", "a", "b", "iid", "shift"]
groups = raw.groupby(["data_seed", "repeat", "n"])
for (seed, repeat, n), policies in new.groupby(["data_seed", "repeat", "n"]):
    candidates = groups.get_group((seed, repeat, n)).set_index("gp_seed").sort_index()
    order = np.random.default_rng([99041, seed, repeat]).permutation(candidates.index)
    for choice in policies.itertuples():
        prefix = order[:choice.k] if choice.menu == "trees" else [order[0]]
        winner = candidates.loc[prefix].a.idxmin()
        assert winner == choice.winner_seed and order[0] == choice.first_seed
        np.testing.assert_array_equal(candidates.loc[winner, scores].to_numpy(),
                                      [getattr(choice, c) for c in scores])
copies = decisions[decisions.menu == "copies"]
assert (copies.first_seed == copies.winner_seed).all()
assert (copies[["gain", "excess", "apparent", "shift_gain"]] == 0).all().all()
primary = new[(new.n == 64) & (new.k == 12) & (new.menu == "trees")]
for metric in ["excess", "gain"]:
    vals = primary.groupby("data_seed")[metric].mean().to_numpy()
    width = t.ppf(.975, 15)*np.std(vals, ddof=1)/4
    row = summary[(summary.panel == "new") & (summary.n == 64) & (summary.k == 12)
                  & (summary.menu == "trees") & (summary.metric == metric)].iloc[0]
    np.testing.assert_allclose([row["mean"], row.lower, row.upper],
                               [vals.mean(), vals.mean()-width, vals.mean()+width], atol=1e-14)
print(f"Verified {len(new)} fresh selections, {len(copies)} copy controls and both primary intervals.")

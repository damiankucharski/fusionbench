"""Rebuild the study figure from the published per-seed results archive."""
from pathlib import Path
from zipfile import ZipFile

import matplotlib.pyplot as plt
import pandas as pd

here = Path(__file__).parent
with ZipFile(here / "results.zip") as raw:
    ranks = pd.read_csv(raw.open("seed_selection/confirmation/rank_means.csv"))
    blind = pd.read_csv(raw.open("blind/runs.csv"))

fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4.1), layout="constrained")
ranks = ranks[ranks.val_n == 256]
for column, label, color in [("spearman", "Source validation", "#983c24"),
                             ("adapted_spearman", "Adjusted using target votes", "#176576")]:
    left.plot(ranks.test_p, ranks[column], marker="o", label=label, color=color)
left.axvline(.1, color="0.4", ls=":", lw=1)
left.axhline(0, color="0.65", lw=.7)
left.set(xlabel="Fraction of cases with a reversed expert", ylabel="Spearman correlation with test ranking",
         title="Choosing among 12 GP seeds", ylim=(-1, 1.05))
left.legend(frameon=False, fontsize=9)
mask = (blind.task == "two_pathways") & (blind.n_fit == 32)
for i, domain in enumerate(["bimodal", "gaussian"]):
    frame = blind[mask & (blind.domain == domain)]
    for j, (axis, method, color, label) in enumerate([("ica", "circuit", "#176576", "ICA + min/max"),
                                                      ("raw", "mlp", "#756594", "Supervised MLP")]):
        scores = frame[(frame["axes"] == axis) & (frame.model == method)].accuracy
        if scores.empty:
            raise ValueError(f"Missing {domain}, {axis}, {method}")
        right.scatter([i + (j-.5)*.24]*len(scores), scores, alpha=.26, color=color, s=15)
        right.plot(i + (j-.5)*.24, scores.mean(), marker="_", markersize=20, markeredgewidth=3,
                   color=color, label=label if i == 0 else None)
right.set(xticks=[0,1], xticklabels=["Bimodal", "Gaussian"], ylabel="Test accuracy",
          title="Single model trained with 32 labels", xlim=(-.45,1.45), ylim=(.45,1))
right.legend(frameon=False, fontsize=9, loc="lower left")
for ax in (left,right):
    ax.spines[["top","right"]].set_visible(False)
fig.savefig(here / "results.png", dpi=185)

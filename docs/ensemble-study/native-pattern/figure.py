"""Show the native tree's IID contrast with the two public logistic comparators."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t

out = Path(__file__).resolve().parent
raw = pd.read_csv(out / "scores.csv.gz").query("split == 'iid'")
panel = raw.pivot(index=["scenario", "replicate"], columns="method", values="brier")
order = (panel.native - panel.logit).groupby("scenario").mean().sort_values().index
fig, ax = plt.subplots(figsize=(9, 8.5), layout="constrained")
positions = np.arange(len(order))
for ref, label, offset, color in [
    ("logit", "LR on logits", -0.14, "#15616d"),
    ("prob_lr", "LR on probabilities", 0.14, "#a85418"),
]:
    grouped = (panel.native - panel[ref]).groupby("scenario")
    ax.errorbar(
        grouped.mean().reindex(order),
        positions + offset,
        xerr=t.ppf(0.975, 11) * grouped.sem().reindex(order),
        fmt="o",
        ms=4,
        capsize=2,
        color=color,
        label=label,
    )
ax.axvline(0, color=".35", lw=1)
ax.set_yticks(positions, order)
ax.invert_yaxis()
ax.set_xlabel("Single OKAPI tree minus LR · full-class Brier (lower favors OKAPI)")
ax.set_title(
    "One derivative-free native tree on the 19 benchmark families",
    loc="left",
    fontweight="bold",
)
ax.legend(loc="best")
ax.grid(axis="y", alpha=0.14)
fig.savefig(out / "families.png", dpi=160)

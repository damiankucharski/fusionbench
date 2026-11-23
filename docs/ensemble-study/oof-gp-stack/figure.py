"""Draw paired IID contrasts across the 12 banks within each benchmark family."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t

out = Path(__file__).resolve().parent
df = pd.read_csv(out / "scores.csv.gz")
wide = df.query("split == 'iid'").pivot(
    index=["scenario", "replicate"], columns="method", values="brier"
)
order = (wide.hybrid - wide.logit).groupby("scenario").mean().sort_values().index
fig, ax = plt.subplots(figsize=(8.8, 8.8), layout="constrained")
y = np.arange(len(order))
for ref, label, offset, color in [
    ("logit", "Logistic stack on logits", -0.14, "#15616d"),
    ("linear_blend", "Blend of both logistic stackers", 0.14, "#bd641f"),
]:
    groups = (wide.hybrid - wide[ref]).groupby("scenario")
    avg = groups.mean().reindex(order)
    err = (groups.sem() * t.ppf(0.975, 11)).reindex(order)
    ax.errorbar(
        avg,
        y + offset,
        xerr=err,
        fmt="o",
        markersize=4,
        capsize=2,
        color=color,
        label=label,
    )
ax.axvline(0, color="#41474c", linewidth=1)
ax.set_yticks(y, order)
ax.invert_yaxis()
ax.set_xlabel("Hybrid minus comparator · full-class Brier (lower favors the hybrid)")
ax.set_title(
    "Out-of-fold GP fusion on held-out examples", loc="left", fontweight="bold", pad=12
)
ax.legend(loc="lower left", frameon=True)
ax.grid(axis="y", alpha=0.14)
for edge in ("right", "top", "left"):
    ax.spines[edge].set_visible(False)
fig.savefig(out / "family-contrasts.png", dpi=175)

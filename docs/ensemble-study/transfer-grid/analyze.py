"""Summarize the fixed suite, using paired banks as the units of replication."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t

OUT = Path(__file__).resolve().parent
KEY = ["scenario", "replicate", "data_seed", "noise", "mode", "severity"]
CANDIDATES = ["flat", "augmented", "gp", "half"]
REFERENCES = ["deployed", "prob_lr", "logit", "top_n", "mean"]
SHIFTS = ["provider_swap", "log_jitter"]


def replicate_means(frame, values):
    families = frame.groupby(["replicate", "scenario"])[values].mean()
    return families.groupby("replicate").mean()


def contrast(sample):
    mean = sample.mean()
    half = t.ppf(0.975, len(sample) - 1) * sample.sem()
    return {"difference": mean, "low": mean - half, "high": mean + half}


def transfer_diagnostics(frame):
    val = frame[["val_" + k for k in CANDIDATES]].to_numpy()
    test = frame[["test_" + k for k in CANDIDATES]].to_numpy()
    picked = val.argmin(axis=1)
    selected = test[np.arange(len(test)), picked]
    best = test.min(axis=1)
    frame = frame[KEY].copy()
    frame["validation_hit"] = np.isclose(selected, best, rtol=0, atol=1e-12)
    frame["regret"] = selected - best
    return frame


def main():
    scores = pd.read_csv(OUT / "results.csv.gz")
    panel = scores.pivot(index=KEY, columns="method", values="brier").reset_index()
    diagnostics = transfer_diagnostics(pd.read_csv(OUT / "diagnostics.csv.gz"))
    assert len(panel) == len(diagnostics) == 3_840
    panel = panel.merge(diagnostics, on=KEY, validate="one_to_one")
    assert len(panel) == 3_840
    methods = list(scores.method.unique())
    values = [*methods, "regret", "validation_hit"]
    panel.groupby(["mode", "noise", "severity"])[values].mean().to_csv(
        OUT / "conditions.csv"
    )
    groups = {
        "generic disturbances": panel[panel["mode"].isin(SHIFTS)],
        "clean source and test": panel.query("mode == 'clean' and noise == 0"),
        "all clean tests": panel.query("mode == 'clean'"),
        "scenario-specific shift": panel.query("mode == 'canonical_shift'"),
        **{mode: panel[panel["mode"] == mode] for mode in SHIFTS},
    }
    contrasts, means = [], []
    for name, frame in groups.items():
        repl = replicate_means(frame, values)
        means.append(
            {"panel": name, "scenarios": frame.scenario.nunique(), **repl.mean()}
        )
        for ref in REFERENCES:
            contrasts.append(
                {"panel": name, "reference": ref, **contrast(repl.robust - repl[ref])}
            )
    pd.DataFrame(means).to_csv(OUT / "panels.csv", index=False)
    pd.DataFrame(contrasts).to_csv(OUT / "contrasts.csv", index=False)
    groups["generic disturbances"].groupby("scenario")[values].mean().to_csv(
        OUT / "families.csv"
    )
    figure(panel)


def figure(panel):
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.3), sharex=True, layout="constrained")
    clean = panel.query("mode == 'clean'")
    regret_max = 1.1 * panel.groupby(["mode", "noise", "severity"]).regret.mean().max()
    for k, (mode, title) in enumerate(
        [
            ("provider_swap", "One faulty provider · 19 scenarios"),
            ("log_jitter", "Noisy provider logits · 19 scenarios"),
            ("canonical_shift", "Generator shift · 9 scenarios"),
        ]
    ):
        current = panel[panel["mode"] == mode]
        neutral = clean[clean.scenario.isin(current.scenario.unique())]
        frame = pd.concat([neutral, current])
        for q, group in frame.groupby("noise"):
            avg = group.groupby("severity")[["regret", "validation_hit"]].mean()
            axes[0, k].plot(avg.index, 100 * avg.validation_hit, "o-", label=f"{q:.0%}")
            axes[1, k].plot(avg.index, avg.regret, "o-")
        axes[0, k].set_title(title, fontsize=11)
        axes[1, k].set_xlabel("Test disturbance severity")
        for ax in axes[:, k]:
            ax.grid(axis="y", alpha=0.2)
            ax.set_xticks([0, 0.25, 0.5, 1])
        axes[0, k].set_ylim(0, 100)
        axes[1, k].set_ylim(0, regret_max)
    axes[0, 0].set_ylabel("Validation winner also best on test (%)")
    axes[1, 0].set_ylabel("Excess test Brier of validation winner")
    axes[0, 0].legend(title="Dev label noise", fontsize=8)
    fig.suptitle(
        "Does validation select the right fusion tree? Four fixed candidates per bank"
    )
    fig.savefig(OUT / "transfer.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()

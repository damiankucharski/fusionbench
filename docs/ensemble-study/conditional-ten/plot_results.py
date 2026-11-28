"""Verify R10 scenario aggregation with the public taxonomy and render paired effects."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from fusionbench.taxonomy import (
    APPLICATION_PROFILES,
    MECHANISM_GROUPS,
    PURPOSE_TAGS,
    TAXONOMY_PRESET,
    TAXONOMY_VERSION,
    scenario_weights,
)
from fusionbench.transfer import load_preset

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
MECHANISMS = {
    "evidence_calibration": "Evidence and calibration",
    "pool_selection": "Pool selection",
    "error_dependence": "Error dependence",
    "structured_rules": "Structured rules",
    "context_routing": "Context routing",
    "transfer_shift": "Transfer shift",
    "grouped_sampling": "Grouped sampling",
}
PROFILES = {
    "standard": "Standard macro",
    "equal_cluster": "Equal mechanism groups",
    "multi_site": "Multi-site profile",
    "rule_design": "Rule-design profile",
}
FOCAL = {
    "mixed": "Mixed",
    "reversed": "Reversed association",
    "narrow": "Narrow activity contrast",
    "extreme": "Sparse overlap",
    "small128": "128 source examples",
    "noisy_context": "Noisy source context",
    "case_mix": "Case mix only",
    "varied": "Activity only",
    "homogeneous": "Homogeneous",
}
COMPARISONS = {
    "native": ("#00605e", "o", "Locked minus ordinary native"),
    "guarded": ("#a94b20", "s", "Locked minus unconditional weighting"),
}


def verify_weights(raw, levels):
    categories = {
        "standard": scenario_weights(),
        "equal_cluster": scenario_weights(
            group_weights=dict.fromkeys(MECHANISM_GROUPS, 1.0)
        ),
        **{
            p: scenario_weights(group_weights=weights)
            for p, weights in APPLICATION_PROFILES.items()
        },
        **{f"cluster/{g}": scenario_weights(groups=g) for g in MECHANISM_GROUPS},
        **{f"purpose/{p}": scenario_weights(purposes=p) for p in PURPOSE_TAGS},
    }
    assert set(MECHANISMS) == set(MECHANISM_GROUPS)
    members = [s for group in MECHANISM_GROUPS.values() for s in group]
    standard = raw.query("cohort == 'standard' and world == 'test'")
    names = set(load_preset(TAXONOMY_PRESET)["scenarios"])
    if len(members) != 30 or set(members) != names or set(standard.family) != names:
        raise ValueError(
            "Mechanism assignments do not partition all 30 registered scenarios"
        )
    methods = sorted(standard.method.unique())
    per_bank = standard.pivot(
        index=["family", "seed"], columns="method", values="brier"
    ).reindex(columns=methods)
    if (
        per_bank.isna().any().any()
        or not per_bank.groupby(level="family").size().eq(8).all()
    ):
        raise ValueError("Expected complete eight-seed standard panel")
    risks = per_bank.groupby(level="family").mean()
    estimates = levels.pivot(index="group", columns="method", values="risk").reindex(
        columns=methods
    )
    error = 0.0
    for name, weights in categories.items():
        np.testing.assert_allclose(sum(weights.values()), 1, atol=1e-15, rtol=0)
        measured = (
            np.asarray([weights[s] for s in weights])
            @ risks.loc[list(weights)].to_numpy()
        )
        reported = estimates.loc[name].to_numpy()
        np.testing.assert_allclose(measured, reported, atol=1e-12, rtol=0)
        error = max(error, float(np.max(np.abs(measured - reported))))
    # Check each declared profile's total weight per group. The two adjusted
    # summaries convey user priorities and do not enter the registered macro.
    for name, declared in APPLICATION_PROFILES.items():
        for group, members in MECHANISM_GROUPS.items():
            measured = sum(categories[name].get(s, 0.0) for s in members)
            np.testing.assert_allclose(
                measured, declared.get(group, 0), atol=1e-15, rtol=0
            )
    return {
        "taxonomy_version": TAXONOMY_VERSION,
        "preset": TAXONOMY_PRESET,
        "families": len(risks),
        "seeds_per_family": 8,
        "method_count": len(methods),
        "aggregates_checked": list(categories),
        "absolute_risks_checked": len(methods) * len(categories),
        "max_abs_error": error,
    }


def paired_panel(ax, rows, names, chosen, simple_levels):
    pairs = rows.set_index(["group", "method", "reference"])
    y = np.arange(len(names))
    for ref, (color, marker, _) in COMPARISONS.items():
        if chosen == ref:
            changes = np.zeros((len(names), 3))
        else:
            changes = np.array(
                [
                    pairs.loc[(g, chosen, ref), ["delta", "lo", "hi"]].to_numpy(
                        dtype=float
                    )
                    for g in names
                ]
            )
        direct = np.array(
            [simple_levels.at[g, chosen] - simple_levels.at[g, ref] for g in names]
        )
        np.testing.assert_allclose(changes[:, 0], direct, atol=1e-12, rtol=0)
        delta, lo, hi = (changes * 1000).T
        height = y + (-0.15 if ref == "native" else 0.15)
        ax.hlines(height, lo, hi, color=color, lw=1.35)
        ax.plot(delta, height, ls="none", marker=marker, mew=0, ms=5.0, color=color)
    ax.set_yticks(y, list(names.values()))
    ax.set_ylim(len(y) - 0.5, -0.5)
    ax.axvline(0, lw=1, color="#526169", zorder=0)
    ax.grid(axis="x", alpha=0.20)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#9ea8a9")
    ax.tick_params(axis="both", length=0, pad=5)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))


def main():
    chosen = json.loads((HERE / "selection.json").read_text())["chosen"]
    raw = pd.read_csv(OUT / "r10.csv", float_precision="round_trip")
    levels = pd.read_csv(OUT / "r10-levels.csv", float_precision="round_trip")
    contrasts = pd.read_csv(OUT / "r10-contrasts.csv", float_precision="round_trip")
    check = verify_weights(raw, levels)
    risks = levels.pivot(index="group", columns="method", values="risk")
    cluster_names = {f"cluster/{g}": label for g, label in MECHANISMS.items()}
    focal_names = {f"{f}/max_regret": label for f, label in FOCAL.items()}
    with plt.rc_context(
        {
            "font.size": 9,
            "axes.titleweight": "bold",
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
        }
    ):
        fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.8))
        fig.subplots_adjust(left=0.181, right=0.98, top=0.88, bottom=0.21, wspace=0.78)
        paired_panel(axes[0], contrasts, {**cluster_names, **PROFILES}, chosen, risks)
        paired_panel(axes[1], contrasts, focal_names, chosen, risks)
        axes[0].axhline(len(cluster_names) - 0.5, lw=0.8, ls=":", color="#9ba7ab")
        axes[0].set_title("A   Standard-v6", loc="left", pad=11)
        axes[0].set_xlabel("Change in mean Brier × 1,000", labelpad=7)
        axes[1].set_title("B   Two-environment sources", loc="left", pad=11)
        axes[1].set_xlabel("Change in mean maximum regret × 1,000", labelpad=7)
        markers = [
            Line2D([], [], color=c, marker=m, lw=1.3, ms=5.0, label=label)
            for c, m, label in COMPARISONS.values()
        ]
        fig.legend(
            handles=markers,
            loc="lower center",
            bbox_to_anchor=(0.53, 0.045),
            frameon=False,
            ncol=2,
        )
        fig.text(
            0.53,
            0.026,
            f"Round 9 selected the {chosen} rule. Lines show paired 95% seed-bootstrap intervals. Negative values favor the locked rule.",
            ha="center",
            fontsize=8,
        )
        for suffix in ("png", "pdf"):
            fig.savefig(
                OUT / f"r10-conditional.{suffix}",
                dpi=190,
                bbox_inches="tight",
                pad_inches=0.15,
            )
        plt.close(fig)
    (OUT / "r10-taxonomy-audit.json").write_text(json.dumps(check, indent=2) + "\n")
    print(check)


if __name__ == "__main__":
    main()

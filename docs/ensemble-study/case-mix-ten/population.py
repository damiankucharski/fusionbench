"""Enumerate source-standardized Bayes forecasts and both endpoint risks, with no sample fitting."""

import csv
from pathlib import Path

import numpy as np
from banks import PREVIOUS, PROFILES

from fusionbench.scenarios.rare_shortcut_noise import RareShortcutNoise

HERE = Path(__file__).resolve().parent
FOCUS = (
    "mixed",
    "reversed",
    "case_mix",
    "varied",
    "extreme",
    "small128",
    "noisy_context",
    "homogeneous",
    "no_shortcut",
)
METHODS = {"unweighted": 1.0, "cap2": 2.0, "full": np.inf}


def joint(region, shortcut):
    """P(report pattern, Y) in lexicographic order, with the true cue at column two."""
    _, m, q = RareShortcutNoise(region_fit=region, shortcut_fit=shortcut).support("fit")
    return m[:, None] * np.column_stack((1 - q, q))


def posterior(law):
    return law[:, 1] / law.sum(axis=1)


def source_population(region, activity, noise, cap):
    # At a given report pattern the cue identifies the true region. The source
    # metadata are an independently flipped copy; condition on this cue before
    # taking either moment of the weights. Each source environment is equally large.
    source = np.stack([joint(pi, a) for pi, a in zip(region, activity, strict=True)])
    cue = (np.arange(16) & 2) != 0
    prob_observed = np.where(cue, 1 - noise, noise)
    by_observed = np.stack((1 - prob_observed, prob_observed), axis=0)
    cells = np.einsum("exy,ux->eu", source, by_observed) / len(region)
    independent = cells.sum(axis=1)[:, None] * cells.sum(axis=0)
    if cap == 1:
        weights = np.ones_like(cells)
    else:
        weights = np.minimum(independent / cells, cap)
    weights /= (weights * cells).sum()
    reweighted = weights * cells
    np.testing.assert_allclose(reweighted.sum(axis=0), cells.sum(axis=0), atol=2e-15)
    np.testing.assert_allclose(reweighted.sum(axis=1), cells.sum(axis=1), atol=2e-15)
    expected_w = weights @ by_observed
    law = np.einsum("exy,ex->xy", source, expected_w) / len(region)
    ess_fraction = 1 / np.sum(cells * weights**2)
    np.testing.assert_allclose(law.sum(), 1, rtol=0, atol=2e-15)
    mass = source.sum(axis=2) * expected_w
    by_region = np.column_stack([mass[:, cue == r].sum(axis=1) for r in (0, 1)])
    mix = (activity @ by_region) / by_region.sum(axis=0)
    return law, mix, ess_fraction, by_region.sum(axis=0)[1] / len(region)


def endpoint_metrics(p, endpoints):
    result = {}
    for name, target in endpoints.items():
        target_prob, target_mass = posterior(target), target.sum(axis=1)
        risk = 2 * np.sum(target[:, 0] * p**2 + target[:, 1] * (1 - p) ** 2)
        excess = 2 * np.sum(target_mass * (p - target_prob) ** 2)
        bayes = 2 * np.sum(target_mass * target_prob * (1 - target_prob))
        np.testing.assert_allclose(risk - bayes, excess, rtol=0, atol=2e-15)
        result[f"risk_{name}"] = risk
        result[f"regret_{name}"] = excess
    result["max_regret"] = max(result["regret_s0"], result["regret_s100"])
    return result


def calculate():
    endpoints = {k: joint(*settings) for k, settings in PREVIOUS["TARGETS"].items()}
    reference = (endpoints["s0"] + endpoints["s100"]) / 2
    np.testing.assert_allclose(reference, endpoints["s50"], atol=2e-15, rtol=0)
    profiles = {**PROFILES, "no_shortcut": ((0.15, 0.85), (0.0, 0.0))}
    rows = []
    for name in FOCUS:
        region, activity = profiles[name]
        noise = 0.25 if name == "noisy_context" else 0
        size = 128 if name == "small128" else 512
        alternatives = {
            method: source_population(region, np.array(activity), noise, cap)
            for method, cap in METHODS.items()
        }
        alternatives["guarded"] = next(
            alternatives[method]
            for method in ("full", "cap2", "unweighted")
            if alternatives[method][2] >= 0.65
        )
        for method, (law, mix, ess, region_mass) in alternatives.items():
            p = posterior(law)
            row = {
                "profile": name,
                "policy": method,
                "source_rows": size,
                "metadata_flip": noise,
                "weighted_true_region_prevalence": region_mass,
                "activity_outside_region": mix[0],
                "activity_in_region": mix[1],
                "ess_fraction": ess,
                "asymptotic_ess_at_source_n": size * ess,
                "max_abs_difference_from_s50_posterior": np.max(
                    np.abs(p - posterior(reference))
                ),
                **endpoint_metrics(p, endpoints),
            }
            if method == "unweighted":
                native = row
            row["delta_max_regret_vs_unweighted"] = (
                row["max_regret"] - native["max_regret"]
            )
            rows.append(row)
            # All eight focal profiles have symmetric environment representation.
            np.testing.assert_allclose(region_mass, 0.5, rtol=0, atol=2e-15)

            if method == "full" and noise == 0 and name != "no_shortcut":
                np.testing.assert_allclose(law, reference, rtol=0, atol=2e-15)
            if name in {"case_mix", "homogeneous", "varied", "no_shortcut"}:
                np.testing.assert_allclose(
                    row["delta_max_regret_vs_unweighted"], 0, atol=2e-15
                )
            if name == "no_shortcut":
                np.testing.assert_allclose(
                    p, posterior(endpoints["s0"]), rtol=0, atol=2e-15
                )
    keyed = {(r["profile"], r["policy"]): r for r in rows}
    for method in METHODS:
        m, rev = [keyed[(k, method)] for k in ("mixed", "reversed")]
        np.testing.assert_allclose(
            m["activity_in_region"], rev["activity_outside_region"], atol=2e-15
        )
        np.testing.assert_allclose(m["ess_fraction"], rev["ess_fraction"], atol=2e-15)
    full = keyed[("mixed", "full")]
    np.testing.assert_allclose(full["ess_fraction"], 4 * 0.15 * 0.85, atol=2e-15)
    return rows


if __name__ == "__main__":
    rows = calculate()
    out = HERE / "results" / "population.csv"
    out.parent.mkdir(exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, lineterminator="\n", fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

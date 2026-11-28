"""Distinguish source sample size, prevalence and native family choice on exact target support."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from fusionbench.eval.suite_point_gp import predict_tree
from fusionbench.methods.averaging import SimpleAverage, TopN
from fusionbench.methods.native_pattern import RegularizedNativePatternOKAPI
from fusionbench.methods.stacking import StackingLR
from fusionbench.scenarios.rare_shortcut_noise import RareShortcutNoise

HERE = Path(__file__).resolve().parent
SIZES = (512, 5120)
PREVALENCES = (0.05, 0.5)
FAMILIES = ("flat", "augmented", "gp")


def make_scenario(pi):
    return RareShortcutNoise(
        region_fit=pi,
        region_test=0.5,
        shortcut_fit=0.0,
        shortcut_test=0.0,
        provider_noise_fit=0.0,
        provider_noise_test=0.0,
        label_noise_fit=0.0,
        label_noise_test=0.0,
    )


def support_risks(positive, mass, q):
    excess = float(2 * mass @ np.square(positive - q))
    irreducible = float(2 * mass @ (q * (1 - q)))
    return {"brier": excess + irreducible, "excess": excess}


def evaluate_fit(x, y, region, scenario, meta):
    # Only source data enter the four fitting calls. Native family selection finishes here.
    native = RegularizedNativePatternOKAPI(seed=0, split_seed=0)
    models = {
        "mean": SimpleAverage(),
        "top_n": TopN(metric="brier"),
        "prob_lr": StackingLR(c_value=1.0),
        "regularized": native,
    }
    timings = {}
    for name, model in models.items():
        start = perf_counter()
        model.fit(x, y, task="binary", n_classes=2)
        timings[name] = perf_counter() - start
    val = native.validation_indices_
    counts = native.counts_
    support, pooled_mass, pooled_q = scenario.support("test")
    domains = {"overall": (pooled_mass, pooled_q)}
    for key, value in (("common", 0), ("rare", 1)):
        other, mass, q = scenario.support("test", region=value)
        np.testing.assert_array_equal(other, support)
        domains[key] = (mass, q)
    assert len(support) == 16
    for mass, q in domains.values():
        np.testing.assert_allclose(mass.sum(), 1.0, atol=1e-12)
        assert np.all(mass >= 0) and np.all((0 <= q) & (q <= 1))

    scores, state_predictions, choices = [], [], []

    def record(method, stage, family, prediction):
        np.testing.assert_allclose(prediction.sum(axis=1), 1.0, atol=1e-12)
        assert prediction.shape == (16, 2) and np.all(
            (0 <= prediction) & (prediction <= 1)
        )
        p = prediction[:, 1]
        identity = dict(**meta, method=method, stage=stage, family=family)
        state_predictions.extend(
            dict(**identity, state=i, positive=v) for i, v in enumerate(p)
        )
        risks = {
            group: support_risks(p, mass, q) for group, (mass, q) in domains.items()
        }
        for metric in ("brier", "excess"):
            np.testing.assert_allclose(
                risks["overall"][metric],
                0.5 * (risks["common"][metric] + risks["rare"][metric]),
                rtol=0,
                atol=2e-14,
            )
        for group, risk in risks.items():
            scores.append(dict(**identity, group=group, **risk))

    for method, model in models.items():
        family = native.chosen_ if method == "regularized" else method
        record(method, "deployed", family, model.predict(support))
    rare_count = int(region.sum())
    costs = dict(
        **meta,
        rare_total=rare_count,
        rare_validation=int(region[val].sum()),
        rare_discovery=rare_count - int(region[val].sum()),
        discovery_rows=counts["discovery_rows"],
        validation_rows=len(val),
        chosen=native.chosen_,
        top_n=models["top_n"].n_selected_,
        gp_calls=counts["score_calls"]["gp_discovery"],
        gp_unique=counts["gp_unique_programs"],
        gp_path_digest=counts["gp_path_digest"],
        native_calls=counts["total_score_calls"],
        **{f"{name}_seconds": t for name, t in timings.items()},
    )
    assert native.trees_["gp"] is native.discovery_trees_["gp"]
    for stage, trees in (
        ("discovery", native.discovery_trees_),
        ("refit", native.trees_),
    ):
        for family in FAMILIES:
            tree = trees[family]
            prediction = predict_tree(tree, support)
            if stage == "refit" and family == native.chosen_:
                np.testing.assert_array_equal(prediction, native.predict(support))
            record("regularized", stage, family, prediction)
            if stage == "discovery":
                vpred = predict_tree(tree, x[val])[:, 1]
                losses = 2 * (vpred - y[val]) ** 2
                vscore = counts["validation_scores"][family]
                np.testing.assert_allclose(losses.mean(), 2 * (1 - vscore), atol=2e-12)
                choices.append(
                    dict(
                        **meta,
                        family=family,
                        chosen=family == native.chosen_,
                        validation_brier=losses.mean(),
                        validation_common_brier=losses[~region[val]].mean(),
                        validation_rare_brier=(
                            losses[region[val]].mean() if region[val].any() else np.nan
                        ),
                    )
                )
    return scores, state_predictions, choices, costs


def summarize(scores, states, costs, directory, bootstrap_seed):
    n_banks = costs.seed.nunique()
    rng = np.random.default_rng(bootstrap_seed)
    indices = rng.integers(0, n_banks, (10000, n_banks))
    summaries, contrasts, regrets, paired_regrets = [], [], [], []

    def estimate(vector):
        values = np.asarray(vector)
        assert values.shape == (n_banks,) and np.isfinite(values).all()
        low, high = np.quantile(values[indices].mean(axis=1), [0.025, 0.975])
        return {"mean": values.mean(), "low": low, "high": high}

    joined = scores.merge(
        costs[["seed", "prevalence", "n", "chosen"]],
        on=["seed", "prevalence", "n"],
        validate="many_to_one",
    )
    deployed = joined[joined.stage == "deployed"]
    for (pi, n, method, group), block in deployed.groupby(
        ["prevalence", "n", "method", "group"]
    ):
        for metric in ("brier", "excess"):
            summaries.append(
                dict(
                    prevalence=pi,
                    n=n,
                    method=method,
                    group=group,
                    metric=metric,
                    **estimate(block.sort_values("seed")[metric]),
                )
            )
    comparisons = {
        "larger_n_rare_source": ((0.05, 5120), (0.05, 512)),
        "same_expected_rare_count": ((0.05, 5120), (0.5, 512)),
        "larger_n_balanced_source": ((0.5, 5120), (0.5, 512)),
        "lower_prevalence_small_n": ((0.05, 512), (0.5, 512)),
        "lower_prevalence_large_n": ((0.05, 5120), (0.5, 5120)),
    }
    for (method, group), block in deployed.groupby(["method", "group"]):
        pivot = block.pivot(
            index="seed", columns=["prevalence", "n"], values="brier"
        ).sort_index()
        for name, (a, b) in comparisons.items():
            contrasts.append(
                dict(
                    method=method,
                    group=group,
                    contrast=name,
                    **estimate(pivot[a] - pivot[b]),
                )
            )

    candidates = joined[joined.stage.isin(["discovery", "refit"])]
    for (seed, pi, n, stage), block in candidates.groupby(
        ["seed", "prevalence", "n", "stage"]
    ):
        chosen = block.chosen.iat[0]
        matrix = block.pivot(index="group", columns="family", values="brier").loc[
            :, list(FAMILIES)
        ]
        best_overall = matrix.loc["overall"].idxmin()
        for group, row in matrix.iterrows():
            best_group = row.idxmin()
            regrets.append(
                {
                    "seed": seed,
                    "prevalence": pi,
                    "n": n,
                    "stage": stage,
                    "group": group,
                    "chosen": chosen,
                    "overall_best": best_overall,
                    "group_best": best_group,
                    "chosen_brier": row[chosen],
                    "overall_best_brier": row[best_overall],
                    "group_best_brier": row[best_group],
                    "regret_to_overall_best": row[chosen] - row[best_overall],
                    "regret_to_group_best": row[chosen] - row[best_group],
                }
            )
    regrets = pd.DataFrame(regrets)
    for (pi, n, stage, group), block in regrets.groupby(
        ["prevalence", "n", "stage", "group"]
    ):
        for metric in (
            "chosen_brier",
            "group_best_brier",
            "regret_to_group_best",
            "regret_to_overall_best",
        ):
            paired_regrets.append(
                dict(
                    prevalence=pi,
                    n=n,
                    stage=stage,
                    group=group,
                    metric=metric,
                    **estimate(block.sort_values("seed")[metric]),
                )
            )
    for (pi, n, group), block in regrets.groupby(["prevalence", "n", "group"]):
        pivot = block.pivot(
            index="seed", columns="stage", values="chosen_brier"
        ).sort_index()
        paired_regrets.append(
            dict(
                prevalence=pi,
                n=n,
                stage="refit_minus_discovery",
                group=group,
                metric="chosen_brier",
                **estimate(pivot.refit - pivot.discovery),
            )
        )
    # Additional descriptive comparisons use the complete set of observable inputs.
    stage_states = states[states.stage.isin(["discovery", "refit"])]
    keys = ["seed", "prevalence", "n", "stage"]
    surface = stage_states.pivot(
        index=keys + ["state"], columns="family", values="positive"
    )
    differences = (surface.flat - surface.augmented).abs().groupby(level=keys).max()
    equivalence = differences.rename("max_absolute_difference").reset_index()
    equivalence["equivalent_at_1e_12"] = equivalence.max_absolute_difference <= 1e-12
    overall_refits = regrets[(regrets.stage == "refit") & (regrets.group == "overall")]
    attainable = overall_refits.pivot(
        index="seed", columns=["prevalence", "n"], values="group_best_brier"
    ).sort_index()
    contrasts.append(
        dict(
            method="regularized_hindsight_best_refit",
            group="overall",
            contrast="same_expected_rare_count",
            **estimate(attainable[0.05, 5120] - attainable[0.5, 512]),
        )
    )
    for name, table in (
        ("summary", summaries),
        ("contrasts", contrasts),
        ("hindsight", regrets),
        ("family-equivalence", equivalence),
        ("family-diagnostics", paired_regrets),
    ):
        pd.DataFrame(table).to_csv(directory / f"rare-{name}.csv", index=False)


def run(directory, bootstrap_seed):
    directory.mkdir(parents=True, exist_ok=True)
    scores, states, choices, costs = [], [], [], []
    for seed in range(252100, 252116):
        for pi in PREVALENCES:
            scenario = make_scenario(pi)
            # Reset the stream across source prevalences, then derive both sizes by prefix.
            x, y, latent = scenario.sample_split(
                max(SIZES), np.random.default_rng([seed, 0]), "fit"
            )
            region = np.asarray(latent["region"], dtype=bool)
            for n in SIZES:
                data = evaluate_fit(
                    x[:n],
                    np.asarray(y[:n]).reshape(-1),
                    region[:n],
                    scenario,
                    {"seed": seed, "prevalence": pi, "n": n},
                )
                scores.extend(data[0])
                states.extend(data[1])
                choices.extend(data[2])
                costs.append(data[3])
        print(f"Finished bank {seed}", flush=True)
    scores, states, costs = (
        pd.DataFrame(scores),
        pd.DataFrame(states),
        pd.DataFrame(costs),
    )
    for name, table in (
        ("risks", scores),
        ("probabilities", states),
        ("validation", pd.DataFrame(choices)),
        ("costs", costs),
    ):
        table.to_csv(
            directory / f"rare-{name}.csv.gz",
            index=False,
            compression={"method": "gzip", "mtime": 0},
        )
    summarize(scores, states, costs, directory, bootstrap_seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "results")
    parser.add_argument("--bootstrap-seed", type=int, default=97942)
    args = parser.parse_args()
    run(args.out, args.bootstrap_seed)

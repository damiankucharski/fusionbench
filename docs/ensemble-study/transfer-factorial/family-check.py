"""Replay all native families in the post-hoc weak-specialization check."""

import hashlib
import json
from pathlib import Path
from runpy import run_path
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from fusionbench.eval.suite_point_gp import predict_tree
from fusionbench.methods.averaging import SimpleAverage, TopN
from fusionbench.methods.native_pattern import RegularizedNativePatternOKAPI
from fusionbench.methods.stacking import StackingLR
from fusionbench.scenarios.stratified_pool import StratifiedPool

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
training_bank = run_path(str(HERE / "seeds.py"))["training_bank"]
SEEDS = range(258100, 258116)
FAMILIES = ("flat", "augmented", "gp")


def risks(fit_p, support_p, y, val, mass, q):
    loss = 2 * (fit_p[:, 1] - y) ** 2
    p = support_p[:, 1]
    excess = 2 * np.dot(mass, (p - q) ** 2)
    oracle = 2 * np.dot(mass, q * (1 - q))
    population = 2 * np.dot(mass, q * (1 - p) ** 2 + (1 - q) * p**2)
    np.testing.assert_allclose(population, oracle + excess, rtol=0, atol=1e-14)
    return {
        "fit_brier": float(loss.mean()),
        "validation_rows_brier": float(loss[val].mean()),
        "population_brier": float(population),
        "excess_brier": float(excess),
        "forecast_sha256": hashlib.sha256(p.copy().tobytes()).hexdigest(),
    }


def bank(seed):
    sc = StratifiedPool(0.25)
    x, y = training_bank(sc, seed)
    native = RegularizedNativePatternOKAPI(seed=0, split_seed=0).fit(
        x, y, task="binary", n_classes=2
    )
    refs = {
        "mean": SimpleAverage(),
        "top_n": TopN(metric="brier"),
        "prob_lr": StackingLR(c_value=1.0),
    }
    for model in refs.values():
        model.fit(x, y, task="binary", n_classes=2)
    fixed = native.chosen_
    scores = native.counts_["validation_scores"]
    assert fixed == max(FAMILIES, key=scores.get)
    val = native.validation_indices_
    support, mass, q = sc.support()
    records = []
    common = {
        "seed": seed,
        "chosen_family": fixed,
        "discovery_rows": native.counts_["discovery_rows"],
        "validation_rows": len(val),
        "gp_programs_validated": native.counts_["score_calls"]["gp_validation"],
        "total_score_calls": native.counts_["total_score_calls"],
    }
    for stage, trees in (
        ("discovery", native.discovery_trees_),
        ("refit", native.trees_),
    ):
        for family in FAMILIES:
            tree = trees[family]
            fit_pred = predict_tree(tree, x)
            sup_pred = predict_tree(tree, support)
            row = risks(fit_pred, sup_pred, y, val, mass, q)
            if stage == "discovery":
                np.testing.assert_allclose(
                    row["validation_rows_brier"],
                    2 * (1 - scores[family]),
                    rtol=0,
                    atol=2e-14,
                )
            if stage == "refit" and family == fixed:
                np.testing.assert_array_equal(sup_pred, native.predict(support))
            records.append(
                {
                    **common,
                    "stage": stage,
                    "family": family,
                    "is_chosen": family == fixed,
                    "discovery_validation_score": scores[family],
                    **row,
                }
            )
    if native.discovery_trees_["gp"] is not native.trees_["gp"]:
        raise RuntimeError(
            "Expected the original GP validation champion at both stages"
        )
    for name, model in refs.items():
        records.append(
            {
                **common,
                "stage": "reference",
                "family": name,
                "is_chosen": False,
                "discovery_validation_score": None,
                **risks(model.predict(x), model.predict(support), y, val, mass, q),
            }
        )
    return records


def main():
    started = perf_counter()
    with threadpool_limits(limits=1):
        frame = pd.DataFrame(row for s in SEEDS for row in bank(s))
    assert (
        len(frame) == 16 * 9 and not frame.duplicated(["seed", "stage", "family"]).any()
    )
    draws = np.random.default_rng(97947).integers(16, size=(10000, 16))

    def paired(a, b):
        v = a.to_numpy() - b.to_numpy()
        lo, hi = np.quantile(v[draws].mean(axis=1), [0.025, 0.975])
        return {
            "mean": float(v.mean()),
            "low": float(lo),
            "high": float(hi),
            "lower": int((v < -1e-12).sum()),
            "tied": int((np.abs(v) <= 1e-12).sum()),
            "higher": int((v > 1e-12).sum()),
        }

    controls = (
        frame[frame.stage == "reference"]
        .pivot(index="seed", columns="family", values="population_brier")
        .loc[list(SEEDS)]
    )
    results = {}
    for stage in ("discovery", "refit"):
        block = frame[frame.stage == stage]
        risks_by_family = block.pivot(
            index="seed", columns="family", values="population_brier"
        ).loc[list(SEEDS), list(FAMILIES)]
        chosen = (
            block[block.is_chosen]
            .set_index("seed")
            .loc[list(SEEDS), "population_brier"]
        )
        best = risks_by_family.min(axis=1)
        results[stage] = {
            "family_mean_population_brier": risks_by_family.mean().to_dict(),
            "family_mean_validation_rows_brier": block.groupby("family")
            .validation_rows_brier.mean()
            .to_dict(),
            "chosen_mean_population_brier": float(chosen.mean()),
            "hindsight_minimum_mean_population_brier": float(best.mean()),
            "chosen_population_best_within_tolerance": int(
                (abs(chosen - best) <= 1e-12).sum()
            ),
            "flat_population_best_within_tolerance": int(
                (abs(risks_by_family.flat - best) <= 1e-12).sum()
            ),
            "chosen_minus_flat": paired(chosen, risks_by_family.flat),
            "chosen_minus_mean": paired(chosen, controls["mean"]),
            "flat_minus_mean": paired(risks_by_family.flat, controls["mean"]),
            "chosen_minus_hindsight_minimum": paired(chosen, best),
        }
    distinct = frame.drop_duplicates("seed")
    summary = {
        "elapsed_seconds": perf_counter() - started,
        "banks": len(SEEDS),
        "numerical_threads": 1,
        "split_seed": 0,
        "search_seed": 0,
        "bootstrap_seed": 97947,
        "chosen_family_counts": distinct.chosen_family.value_counts().to_dict(),
        "validated_gp_programs": {
            "min": int(distinct.gp_programs_validated.min()),
            "mean": distinct.gp_programs_validated.mean(),
            "max": int(distinct.gp_programs_validated.max()),
        },
        "reference_means": controls.mean().to_dict(),
        "stage_results": results,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / "family-check.csv", index=False)
    (OUT / "family-check.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

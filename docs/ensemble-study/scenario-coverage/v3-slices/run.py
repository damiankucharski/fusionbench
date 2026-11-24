"""Score held-out subgroups for the coverage generators, using fit-only model selection."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from threadpoolctl import threadpool_limits

from fusionbench import get_scenario
from fusionbench.methods.averaging import SimpleAverage, TopN
from fusionbench.methods.native_pattern import RegularizedNativePatternOKAPI
from fusionbench.methods.stacking import StackingLR

HERE = Path(__file__).resolve().parent
SCENARIOS = ("rare_specialist", "redundant_pool", "environment_shortcut", "selective_fallback", "similar_classes_38")
DATA_SEED = 4821000
SEARCH_SEED = 7169000


def settings(scenario_id, data_seed):
    yield "default", {}
    if scenario_id == "rare_specialist":
        yield "no_gain", {"specialist_gain": 0.0}
    elif scenario_id == "redundant_pool":
        yield "permuted_null", {"role_seed": data_seed}
        yield "permuted_signal", {"role_seed": data_seed, "positive_control": True}
    elif scenario_id == "environment_shortcut":
        yield "stationary", {"test_mix": (0.8, 0.1, 0.1)}
    elif scenario_id == "selective_fallback":
        yield "random_fallback", {"mechanism": "random"}


@dataclass
class Split:
    x: np.ndarray
    y: np.ndarray
    oracle: np.ndarray | None
    masks: dict


def draw(scenario_id, scenario, n, rng, split):
    x, y, latent = scenario.sample_split(n, rng, split=split)
    masks = {"all": np.ones(n, dtype=bool)}
    if scenario_id == "rare_specialist":
        rare = latent["region"].astype(bool)
        masks.update(rare=rare, common=~rare)
    elif scenario_id == "environment_shortcut":
        for i, name in enumerate(("aligned", "independent", "reversed")):
            masks[name] = latent["environment"] == i
    elif scenario_id == "selective_fallback":
        fallback = latent["fallback"].astype(bool)
        hard = latent["hard"].astype(bool)
        masks.update(fallback=fallback, available=~fallback, hard=hard, easy=~hard)
    elif scenario_id == "similar_classes_38":
        affected = np.isin(y, [3, 8])
        hard = latent["hard"].astype(bool)
        masks.update(label_3=y == 3, label_8=y == 8, pair_38=affected, other_labels=~affected,
                     hard=hard, easy=~hard, hard_pair_38=hard & affected)
    return Split(x, y, scenario.bayes_oracle(x, latent), masks)


def provider_control(scenario_id, scenario):
    if scenario_id == "redundant_pool":
        return "anchor", scenario.anchor_index
    names = {"rare_specialist": "generalist", "environment_shortcut": "stable", "selective_fallback": "reference"}
    if scenario_id in names:
        name = names[scenario_id]
        return name, scenario.model_ids().index(name)
    return None


def row_losses(predictions, y):
    return np.square(predictions - np.eye(predictions.shape[1])[y]).sum(axis=1)


def run_cell(inputs):
    scenario_id, arm, params, data_seed, gp_seed, n_fit, n_test = inputs
    logger.disable("okapi")
    with threadpool_limits(limits=1):
        scenario = get_scenario(scenario_id, **params)
        # Separate fit/test streams permit changing one split size without changing the other.
        rngs = [np.random.default_rng(seed) for seed in np.random.SeedSequence(data_seed).spawn(2)]
        fit = draw(scenario_id, scenario, n_fit, rngs[0], "fit")
        test = draw(scenario_id, scenario, n_test, rngs[1], "test")
        native = RegularizedNativePatternOKAPI(seed=gp_seed)
        predictions = {}
        for method in (SimpleAverage(), TopN(metric="brier"), StackingLR(), native):
            method.fit(fit.x, fit.y, task=scenario.task, n_classes=scenario.n_classes)
            predictions[method.name] = (method.predict(fit.x), method.predict(test.x))
        control = provider_control(scenario_id, scenario)
        if control is not None:
            name, idx = control
            predictions[name] = (fit.x[:, idx, :], test.x[:, idx, :])
        if test.oracle is not None:
            predictions["oracle"] = (fit.oracle, test.oracle)
        rows = []
        for method, (fit_pred, test_pred) in predictions.items():
            fit_brier = row_losses(fit_pred, fit.y)
            test_brier = row_losses(test_pred, test.y)
            oracle_distance = None if test.oracle is None else np.square(test_pred - test.oracle).sum(axis=1)
            oracle_loss = None if test.oracle is None else row_losses(test.oracle, test.y)
            for name, mask in test.masks.items():
                fmask = fit.masks[name]
                rows.append({
                    "scenario": scenario_id, "arm": arm, "data_seed": data_seed, "gp_seed": gp_seed,
                    "slice": name, "method": method, "n_fit": int(fmask.sum()), "n_test": int(mask.sum()),
                    "n_native_validation": int(fmask[native.validation_indices_].sum()),
                    "fit_brier": fit_brier[fmask].mean() if fmask.any() else np.nan,
                    "test_brier": test_brier[mask].mean() if mask.any() else np.nan,
                    "test_minus_oracle": (test_brier[mask] - oracle_loss[mask]).mean() if mask.any() and oracle_loss is not None else np.nan,
                    "sqdist_observed_posterior": oracle_distance[mask].mean() if mask.any() and oracle_distance is not None else np.nan,
                })
        return rows


def comparisons(frame):
    keys = ["scenario", "arm", "slice", "data_seed", "gp_seed"]
    paired = frame.pivot(index=keys, columns="method", values="test_brier").reset_index()
    methods = sorted(set(frame.method) - {"okapi_native_pattern_l2"})
    contrasts = []
    for method in methods:
        differences = (paired.okapi_native_pattern_l2 - paired[method]).rename("delta")
        values = paired.assign(delta=differences).dropna(subset=["delta"])
        for key, group in values.groupby(keys[:3]):
            contrasts.append(dict(zip(keys[:3], key, strict=True)) | {
                "comparator": method, "n_banks": len(group), "mean_delta": group.delta.mean(),
                "sd_between_banks": group.delta.std(), "min_delta": group.delta.min(),
                "max_delta": group.delta.max(), "native_lower": int((group.delta < -1e-12).sum()),
            })
    return pd.DataFrame(contrasts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=SCENARIOS)
    parser.add_argument("--n-fit", type=int, default=512)
    parser.add_argument("--n-test", type=int, default=4096)
    parser.add_argument("--replicates", type=int, default=6)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out", type=Path, default=HERE)
    args = parser.parse_args()
    cells = [(scenario, arm, params, DATA_SEED + i, SEARCH_SEED + i, args.n_fit, args.n_test)
             for i in range(args.replicates) for scenario in args.scenarios
             for arm, params in settings(scenario, DATA_SEED + i)]
    with ProcessPoolExecutor(args.workers) as pool:
        frame = pd.DataFrame([row for rows in pool.map(run_cell, cells) for row in rows])
    args.out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out / "banks.csv", index=False, float_format="%.10g")
    comparisons(frame).to_csv(args.out / "paired.csv", index=False, float_format="%.10g")


if __name__ == "__main__":
    main()

"""Run reproducible, mechanism-specific comparisons over the standard scenarios."""

import argparse
import json
from copy import deepcopy
from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

from fusionbench import __version__, get_scenario
from fusionbench.eval.transfer_grid import brier
from fusionbench.methods.averaging import SimpleAverage, TopN
from fusionbench.methods.native_pattern import RegularizedNativePatternOKAPI
from fusionbench.methods.stacking import StackingLR
from fusionbench.scenarios.registry import STANDARD_SCENARIOS

DEFAULT_METHODS = ("okapi_native_pattern_l2", "simple_average", "top_n", "stacking_lr")
PROFILE_VERSION = "stress-v1"

# Settings are compared with the constructor defaults in the reference profile.
_VARIANTS = {
    "committee": {"weaker_providers": {"accuracy": 0.55}, "fewer_providers": {"k": 3}},
    "deadweight": {"no_irrelevant": {"n_bad": 0}, "many_irrelevant": {"n_bad": 24}},
    "needle": {"few_irrelevant": {"n_bad": 7}, "more_irrelevant": {"n_bad": 197}},
    "saturated": {"independent_noise": {"corr": 0}, "identical_providers": {"corr": 1}},
    "correlated": {
        "independent_noise": {"corr": 0},
        "strong_dependence": {"corr": 0.98},
    },
    "calibrated_suppressor": {
        "positive_weak_weight": {"corr": 0.15},
        "zero_weak_weight": {"corr": 0.5},
    },
    "polarity_fault": {
        "no_flips": {"fault_rate": 0},
        "frequent_flips": {"fault_rate": 0.30},
    },
    "hardveto": {
        "rare_decisive": {"decisive_fraction": 0.10},
        "frequent_decisive": {"decisive_fraction": 0.70},
    },
    "hardrescue": {
        "rare_decisive": {"decisive_fraction": 0.10},
        "frequent_decisive": {"decisive_fraction": 0.75},
    },
    "weakest_link": {
        "deeper_matched_prevalence": {
            "m": 6,
            "positive_prevalence": 1 / 16,
            "decisive_fraction": 4 / 15,
        },
        "rarer_matched_depth": {
            "m": 4,
            "positive_prevalence": 0.02,
            "decisive_fraction": 4 / 15,
        },
    },
    "two_pathways": {
        "rare_second_path": {"decisive_fraction": 0.10},
        "frequent_second_path": {"decisive_fraction": 0.75},
    },
    "parity": {
        "weak_second_bit": {"second_accuracy": 0.70},
        "uninformative_second_bit": {"second_accuracy": 0.5},
    },
    "compound": {
        "rare_rescue": {"decisive_fraction": 0.10},
        "frequent_rescue": {"decisive_fraction": 0.75},
    },
    "gated_experts": {
        "uninformative_selector": {"acc_select": 0.5},
        "weaker_experts": {"acc_expert": 0.65},
    },
    "native_switch": {
        "few_material_rows": {"material_fraction": 0.10},
        "many_material_rows": {"material_fraction": 0.85},
    },
    "compositional_switch": {
        "interior_grid": {"corner_probability": 0},
        "corners_only": {"corner_probability": 1},
    },
    "native_close_switch": {
        "few_material_rows": {"material_fraction": 0.10},
        "many_material_rows": {"material_fraction": 0.85},
    },
    "miscalibration": {
        "undistorted": {"temperature_range": 0},
        "partial_distortion": {"temperature_range": 0.5},
    },
    "multiclass_committee": {
        "weaker_providers": {"accuracy": 0.51},
        "fewer_providers": {"k": 3},
    },
    "similar_classes_38": {
        "mixed_destinations": {"deviation_mix": 0.5},
        "private_destinations": {"deviation_mix": 1},
    },
    "rare_specialist": {
        "rarer_region": {"region_prevalence": 0.025},
        "uninformative_cue": {"cue_accuracy": 0.5},
    },
    "redundant_pool": {
        "larger_null_pool": {"n_candidates": 63},
        "informative_candidate": {"positive_control": True},
    },
    "environment_shortcut": {
        "stationary_mix": {"test_mix": [0.8, 0.1, 0.1]},
        "stable_only": {"stable_only": True},
    },
    "cloned_shortcut": {
        "independent_shortcuts": {"clone_probability": 0},
        "aligned_test": {"test_reversal_probability": 0},
    },
    "stratified_pool": {
        "equal_pool": {"specialization": 0},
        "strong_specialization": {"specialization": 0.75},
    },
    "rare_shortcut_noise": {
        "source_like_test": {"region_test": 0.05, "shortcut_test": 1.0, "provider_noise_test": 0.0},
        "clean_test_providers": {"provider_noise_test": 0.0},
    },
    "selective_fallback": {
        "random_missingness": {"mechanism": "random"},
        "always_available": {"fallback_rate": 0},
    },
}


def list_profiles(scenario=None):
    """Return independent dictionaries of named constructor overrides."""
    if scenario is None:
        return {name: list_profiles(name) for name in STANDARD_SCENARIOS}
    if scenario not in _VARIANTS or scenario not in STANDARD_SCENARIOS:
        raise ValueError(f"Unknown standard scenario {scenario!r}")
    return {"reference": {}, **deepcopy(_VARIANTS[scenario])}


def _selection(requested, options, label):
    if requested is None:
        return list(options)
    values = [requested] if isinstance(requested, str) else list(requested)
    if not values or len(set(values)) != len(values) or not set(values) <= set(options):
        raise ValueError(f"Choose distinct {label} from {', '.join(options)}")
    return values


def _integer(value, minimum, name):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Integral)
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _draw(scenario, scenario_id, seed, split, n):
    rng = np.random.default_rng([seed, 0 if split == "fit" else 1])
    x, y, latent = scenario.sample_split(n, rng, split=split)
    y = np.asarray(y).reshape(-1)
    oracle = scenario.bayes_oracle(x, latent)
    masks = {}
    for key in ("decisive", "material", "region", "fallback", "hard", "flipped",
                "shortcut_cloned", "shortcut_reversed", "high_amplitude"):
        if key in latent:
            mask = np.asarray(latent[key])
            if mask.shape == (n,) and mask.dtype.kind == "b":
                masks[key] = mask
                masks["outside_" + key] = ~mask
    if "environment" in latent:
        for i, environment in enumerate(("aligned", "independent", "reversed")):
            masks[environment] = np.asarray(latent["environment"]) == i
    if scenario_id == "similar_classes_38":
        masks["affected_3_8"] = np.isin(y, [3, 8])
        masks["other_classes"] = ~masks["affected_3_8"]
    return x, y, oracle, masks


def _method(name, search_seed, split_seed):
    if name == "okapi_native_pattern_l2":
        if not RegularizedNativePatternOKAPI.available():
            raise ImportError(
                "Install the matching OKAPI optional dependency or select the three control methods."
            )
        return RegularizedNativePatternOKAPI(seed=search_seed, split_seed=split_seed)
    return {
        "simple_average": SimpleAverage,
        "top_n": lambda: TopN("brier"),
        "stacking_lr": StackingLR,
    }[name]()


def _one_bank(name, profile, settings, seed, search_seed, fit_rows, test_rows, methods):
    scenario = get_scenario(name, **settings)
    x, y, _, fit_masks = _draw(scenario, name, seed, "fit", fit_rows)
    test, target, oracle, test_masks = _draw(scenario, name, seed, "test", test_rows)
    records = []
    groups = []
    for name_method in methods:
        model = _method(name_method, search_seed, seed)
        if name_method == "stacking_lr" and len(np.unique(y)) < 2:
            raise ValueError(
                f"stacking_lr requires two observed fit labels: {name}/{profile}, seed={seed}; increase fit_rows"
            )
        model.fit(x, y, task=scenario.task, n_classes=scenario.n_classes)
        p = model.predict(test)
        key = {
            "scenario": name,
            "profile": profile,
            "data_seed": seed,
            "search_seed": search_seed,
            "split_seed": seed,
            "method": name_method,
        }
        test_loss = brier(p, target)
        records.append(
            {
                **key,
                "fit_rows": len(y),
                "test_rows": len(target),
                "providers": x.shape[1],
                "classes": x.shape[2],
                "profile_parameters": json.dumps(settings, sort_keys=True),
                "fit_class_counts": json.dumps(
                    np.bincount(y, minlength=x.shape[2]).tolist()
                ),
                "test_class_counts": json.dumps(
                    np.bincount(target, minlength=x.shape[2]).tolist()
                ),
                "fit_brier": brier(model.predict(x), y),
                "test_brier": test_loss,
                "oracle_test_brier": brier(oracle, target)
                if oracle is not None
                else None,
                "top_n_selected": getattr(model, "n_selected_", None),
            }
        )
        for group, mask in test_masks.items():
            groups.append(
                {
                    **key,
                    "group": group,
                    "fit_rows": int(fit_masks[group].sum()),
                    "test_rows": int(mask.sum()),
                    "test_brier": brier(p[mask], target[mask]) if mask.any() else None,
                    "oracle_test_brier": brier(oracle[mask], target[mask])
                    if oracle is not None and mask.any()
                    else None,
                }
            )
    return records, groups


def summarize_profiles(results):
    """Average seeds within each setting, settings within family, then families equally."""
    keys = ["scenario", "profile", "data_seed"]
    coverage = results.pivot(index=keys, columns="method", values="test_brier")
    if coverage.empty or not np.isfinite(coverage.values).all():
        raise ValueError("Each bank must contain one finite Brier per selected method")
    if not (
        coverage.groupby(["scenario", "profile"]).size() == results.data_seed.nunique()
    ).all():
        raise ValueError(
            "Every scenario/profile combination must have the same data seeds"
        )
    by_profile = (
        results.groupby(["scenario", "profile", "method"], sort=False)
        .agg(mean_brier=("test_brier", "mean"), banks=("data_seed", "size"))
        .reset_index()
    )
    panels = {
        "all_profiles": results,
        "reference": results[results.profile == "reference"],
        "stress": results[results.profile != "reference"],
    }
    scenarios = []
    macro = []
    paired = []
    for panel, frame in panels.items():
        if frame.empty:
            continue
        cells = by_profile.merge(
            frame[["scenario", "profile"]].drop_duplicates(), on=["scenario", "profile"]
        )
        families = (
            cells.groupby(["scenario", "method"], sort=False)
            .agg(mean_brier=("mean_brier", "mean"), profiles=("profile", "size"))
            .reset_index()
        )
        scenarios.append(families.assign(panel=panel))
        macro.append(
            families.groupby("method", sort=False)
            .agg(mean_brier=("mean_brier", "mean"), scenarios=("scenario", "size"))
            .reset_index()
            .assign(panel=panel)
        )
        pivot = frame.pivot(
            index=["data_seed", "scenario", "profile"],
            columns="method",
            values="test_brier",
        )
        if DEFAULT_METHODS[0] in pivot:
            for ref in pivot.columns.drop(DEFAULT_METHODS[0]):
                delta = pivot[DEFAULT_METHODS[0]] - pivot[ref]
                per_seed = (
                    delta.groupby(["data_seed", "scenario"])
                    .mean()
                    .groupby("data_seed")
                    .mean()
                )
                estimate = per_seed.mean()
                margin = (
                    t.ppf(0.975, len(per_seed) - 1) * per_seed.sem()
                    if len(per_seed) > 1
                    else np.nan
                )
                paired.append(
                    {
                        "panel": panel,
                        "reference": ref,
                        "seeds": len(per_seed),
                        "scenarios": frame.scenario.nunique(),
                        "okapi_minus_reference": estimate,
                        "lower_95": estimate - margin,
                        "upper_95": estimate + margin,
                    }
                )
    return {
        "by_profile": by_profile,
        "by_scenario": pd.concat(scenarios, ignore_index=True),
        "macro": pd.concat(macro, ignore_index=True),
        "paired": pd.DataFrame(
            paired,
            columns=[
                "panel",
                "reference",
                "seeds",
                "scenarios",
                "okapi_minus_reference",
                "lower_95",
                "upper_95",
            ],
        ),
    }


def run_profiles(
    *,
    scenarios=None,
    profiles=None,
    seeds=(0,),
    search_seed=0,
    fit_rows=384,
    test_rows=1024,
    methods=DEFAULT_METHODS,
    out=None,
):
    """Fit and evaluate selected matched profiles. Optionally save tidy CSVs and a manifest."""
    names = _selection(scenarios, STANDARD_SCENARIOS, "scenarios")
    catalog = {n: list_profiles(n) for n in names}
    profile_names = _selection(
        profiles, dict.fromkeys(p for v in catalog.values() for p in v), "profiles"
    )
    selected = [
        (name, p, settings)
        for name, variants in catalog.items()
        for p, settings in variants.items()
        if p in profile_names
    ]
    if not all(any(row[0] == n for row in selected) for n in names):
        raise ValueError(
            "Each selected scenario needs at least one matching profile; narrow --scenario or include reference"
        )
    methods = _selection(methods, DEFAULT_METHODS, "methods")
    if DEFAULT_METHODS[0] in methods:
        methods = list(dict.fromkeys([*methods, *DEFAULT_METHODS[1:]]))
    seeds = [_integer(s, 0, "data seed") for s in seeds]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Supply at least one distinct data seed")
    search_seed = _integer(search_seed, 0, "search_seed")
    fit_rows = _integer(fit_rows, 2, "fit_rows")
    test_rows = _integer(test_rows, 1, "test_rows")
    records = []
    groups = []
    for name, p, settings in selected:
        for seed in seeds:
            record, group = _one_bank(
                name, p, settings, seed, search_seed, fit_rows, test_rows, methods
            )
            records.extend(record)
            groups.extend(group)
    raw = pd.DataFrame(records)
    result = {
        "cells": raw,
        "groups": pd.DataFrame(
            groups,
            columns=[
                "scenario",
                "profile",
                "data_seed",
                "search_seed",
                "split_seed",
                "method",
                "group",
                "fit_rows",
                "test_rows",
                "test_brier",
                "oracle_test_brier",
            ],
        ),
        **summarize_profiles(raw),
    }
    if out is not None:
        folder = Path(out)
        folder.mkdir(parents=True, exist_ok=True)
        for filename, frame in result.items():
            frame.to_csv(folder / (filename + ".csv"), index=False)
        description = {
            "profile_version": PROFILE_VERSION,
            "fusionbench_version": __version__,
            "fit_rows": fit_rows,
            "test_rows": test_rows,
            "data_seeds": seeds,
            "search_seed": search_seed,
            "methods": methods,
            "rng_streams": {"fit": 0, "test": 1},
            "native_split_seed": "data_seed",
            "profiles": [
                {"scenario": n, "name": p, "parameters": s} for n, p, s in selected
            ],
        }
        (folder / "manifest.json").write_text(
            json.dumps(description, indent=2) + "\n", encoding="utf-8"
        )
    return result


def _profile_options(parser):
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print available profiles and their constructor overrides",
    )
    parser.add_argument("--scenario", nargs="+")
    parser.add_argument("--profile", nargs="+")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--fit-rows", type=int, default=384)
    parser.add_argument("--test-rows", type=int, default=1024)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--search-seed", type=int, default=0)
    parser.add_argument(
        "--methods", nargs="+", choices=DEFAULT_METHODS, default=DEFAULT_METHODS
    )
    parser.set_defaults(func=_run_cli)


def add_profile_parser(subparsers):
    parser = subparsers.add_parser(
        "profiles", help="Evaluate matched stress profiles", description=__doc__
    )
    _profile_options(parser)


def _run_cli(args):
    if args.list:
        names = _selection(args.scenario, STANDARD_SCENARIOS, "scenarios")
        print(json.dumps({n: list_profiles(n) for n in names}, indent=2))
        return
    if args.out is None:
        raise SystemExit("--out is required when evaluating profiles")
    result = run_profiles(
        scenarios=args.scenario,
        profiles=args.profile,
        fit_rows=args.fit_rows,
        test_rows=args.test_rows,
        seeds=args.seeds,
        search_seed=args.search_seed,
        methods=args.methods,
        out=args.out,
    )
    print(result["macro"].to_string(index=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    _profile_options(parser)
    _run_cli(parser.parse_args(argv))


if __name__ == "__main__":
    main()

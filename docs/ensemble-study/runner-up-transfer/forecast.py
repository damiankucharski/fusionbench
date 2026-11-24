"""Forecast five fusion losses from archived synthetic scores and target fit banks.

Run this file without arguments from a checkout with FusionBench installed.
Historical fits are regenerated only to measure runner-up coordinates and
verify the six archived coordinates. All five method losses are read from the
archived synthetic study. This runner performs no fusion method training.
"""

import argparse
import json
import math
import sys
from importlib import import_module
from pathlib import Path
from statistics import fmean

import numpy as np

from fusionbench import get_scenario
from fusionbench.eval.confusion_profile import runner_up_profile

STUDY = Path(__file__).resolve().parent
sys.path.insert(0, str(STUDY.parent))
previous = import_module("external-coverage.forecast")
profile_fit = import_module("external-coverage.common").profile_fit

TARGETS = {
    "digits": ("digits", 3_300_000),
    "iris": ("wine", 3_310_000),
    "seeds": ("wine", 3_320_000),
}
FIELDS = (*previous.FIELDS, "top_two_margin", "wrong_class_repeatability")
MODES = {"base": len(FIELDS) - 2, "margin_only": len(FIELDS) - 1, "augmented": len(FIELDS)}
ATOL = 1e-12


def checked_profile(inputs, labels, record, bank_id, max_errors):
    """Verify raw coordinates per split, and scale archived Briers in the centroid."""
    if inputs.dtype != np.float32:
        raise ValueError(f"{bank_id}: expected original float32 provider predictions")
    actual = profile_fit(inputs, labels)
    for field in previous.FIELDS:
        error = abs(actual[field] - float(record[field]))
        if not error <= ATOL:
            raise ValueError(f"{bank_id}: regenerated {field} differs by {error!r}")
        max_errors[field] = max(error, max_errors[field])
    return {**record, **runner_up_profile(inputs, labels)}


def centroid(rows):
    return (
        *previous.profile(rows),
        *(fmean(previous.bounded(row[f], 1.0) for row in rows) for f in FIELDS[-2:]),
    )


def synthetic_centers(source, archived, settings, max_errors):
    rows = previous.read_index(source / "profiles.csv", previous.synth_key)
    centers = {}
    for stratum, config in archived:
        classes, n_fit, seed_root = settings["strata_classes_fit_rows_seed_root"][stratum]
        scenario_id, kwargs = settings["scenarios"][config]
        scenario = get_scenario(scenario_id, n_classes=classes, **kwargs)
        profiles = []
        for replicate in previous.REPLICATES:
            seed = seed_root + replicate
            rng = np.random.default_rng([seed, settings["rng_streams"]["fit"]])
            inputs, labels, _ = scenario.sample(n_fit, rng)
            key = stratum, config, replicate
            if int(rows[key]["seed"]) != seed or inputs.shape != (n_fit, 3, classes):
                raise ValueError(f"{key}: archived seed or expected shape mismatch")
            profiles.append(checked_profile(inputs.astype(np.float32), labels, rows[key], key, max_errors))
        centers[stratum, config] = centroid(profiles)
        if centers[stratum, config][:-2] != tuple(archived[stratum, config]):
            raise ValueError(f"{stratum} {config}: six-coordinate centroid differs from original rule")
    return centers


def target_centers(source, settings, max_errors):
    rows = previous.read_index(source / "fit_profiles.csv", lambda r: (r["suite"], int(r["seed"])))
    expected = {(s, root + i) for s, (_, root) in TARGETS.items() for i in previous.REPLICATES}
    previous.check_keys(rows, expected, "external fit profiles")
    centers = {}
    with np.load(source / "fit_banks.npz", allow_pickle=False) as fit:
        keys = {f"{s}_{seed}_{part}" for s, seed in expected for part in ("inputs", "labels", "rows")}
        previous.check_keys(dict.fromkeys(fit.files), keys, "external fit arrays")
        for suite, (stratum, root) in TARGETS.items():
            classes, n_fit, _ = settings["strata_classes_fit_rows_seed_root"][stratum]
            profiles = []
            for replicate in previous.REPLICATES:
                key = suite, root + replicate
                prefix = f"{suite}_{root + replicate}_"
                inputs, labels, indices = (fit[prefix + k] for k in ("inputs", "labels", "rows"))
                record = rows[key]
                shape = n_fit, 3, classes
                if tuple(int(record[k]) for k in ("n_fit", "n_providers", "n_classes")) != shape:
                    raise ValueError(f"{key}: CSV shape does not match the synthetic stratum")
                if inputs.shape != shape or indices.shape != (n_fit,) or len(np.unique(indices)) != n_fit:
                    raise ValueError(f"{key}: incorrect fit shape or duplicate row identifiers")
                profiles.append(checked_profile(inputs, labels, record, key, max_errors))
            centers[suite] = centroid(profiles)
    return centers


def forecasts(centers, losses, targets):
    distances, predictions, methods = [], [], []
    for suite, target in targets.items():
        stratum = TARGETS[suite][0]
        for mode, size in MODES.items():
            key = {"suite": suite, "stratum": stratum, "profile": mode}
            candidates = sorted(
                (math.dist(target[:size], centers[stratum, config][:size]), config)
                for config in previous.EXPANDED
            )
            distance, chosen = candidates[0]
            distances.extend(
                {**key, "config": config, "distance": d, "selected": config == chosen}
                for d, config in candidates
            )
            scores = losses[stratum, chosen]
            ranks = {m: previous.average_rank(x, scores.values()) for m, x in scores.items()}
            gap = scores["native"] - scores["prob_lr"]
            selected = {**key, "config": chosen, "distance": distance}
            predictions.append({
                **selected,
                "native_minus_prob_lr": gap,
                "direction": "tie" if abs(gap) <= 0.001 else ("native_lower" if gap < 0 else "prob_lr_lower"),
                **{f"mean_brier_{m}": scores[m] for m in previous.METHODS},
                **{f"rank_{m}": ranks[m] for m in previous.METHODS},
            })
            for method in sorted(scores, key=lambda m: (ranks[m], m)):
                methods.append({**selected, "method": method, "mean_brier": scores[method], "rank": ranks[method]})
    return distances, predictions, methods


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", type=Path, default=STUDY.parent / "external-coverage/synthetic-results")
    parser.add_argument("--fit", type=Path, default=STUDY / "external-fit")
    parser.add_argument("--out", type=Path, default=STUDY / "predictions")
    args = parser.parse_args()
    settings = json.loads((args.synthetic / "settings.json").read_text())
    archived, losses = previous.synthetic_tables(args.synthetic)
    synthetic_errors = dict.fromkeys(previous.FIELDS, 0.0)
    external_errors = synthetic_errors.copy()
    centers = synthetic_centers(args.synthetic, archived, settings, synthetic_errors)
    targets = target_centers(args.fit, settings, external_errors)
    distances, predictions, methods = forecasts(centers, losses, targets)
    centroids = [
        {"source": source, "stratum": stratum, "id": name, **{f"{f}_scaled": x for f, x in zip(FIELDS, center, strict=True)}}
        for source, stratum, name, center in (
            *(("synthetic", s, cfg, center) for (s, cfg), center in centers.items()),
            *(("external_fit", TARGETS[s][0], s, center) for s, center in targets.items()),
        )
    ]
    args.out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("centroids", centroids), ("distances", distances), ("forecasts", predictions), ("method_losses", methods)):
        previous.write_rows(args.out / f"{name}.csv", rows)
    (args.out / "profile_verification.json").write_text(json.dumps({
        "absolute_tolerance": ATOL,
        "synthetic_fit_banks_verified": len(centers) * len(previous.REPLICATES),
        "synthetic_max_absolute_error_by_field": synthetic_errors,
        "external_fit_banks_verified": len(targets) * len(previous.REPLICATES),
        "external_max_absolute_error_by_field": external_errors,
        "synthetic_score_source": "external-coverage/synthetic-results/scores.csv",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()

"""Forecast external Brier gaps and method ranks from synthetic results and fit profiles."""

import argparse
import csv
import math
from pathlib import Path
from statistics import fmean

FIELDS = (
    "mean_provider_brier",
    "best_provider_brier",
    "mean_pairwise_top1_disagreement",
    "zero_fraction",
    "mean_max_probability",
    "unanimous_wrong_fraction",
)
METHODS = ("native", "mean", "top_n", "prob_lr", "logit")
EXISTING = ("committee_60", "committee_95", "saturated_50", "saturated_95")
EXPANDED = (*EXISTING, "panel_A", "panel_B", "panel_C", "panel_D", "panel_E")
LIBRARIES = {"existing": EXISTING, "expanded": EXPANDED}
TARGETS = {
    "digits": ("digits", 3_140_000, 10, 449),
    "digits_lr_bag": ("digits", 3_140_000, 10, 449),
    "wine": ("wine", 3_150_000, 3, 44),
}
REPLICATES = range(8)


def read_index(path, key, keep=lambda r: True):
    with path.open(newline="", encoding="utf-8") as stream:
        table = {}
        for row in filter(keep, csv.DictReader(stream)):
            identity = key(row)
            if identity in table:
                raise ValueError(f"{path}: duplicate record for {identity}")
            table[identity] = row
        return table


def write_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def bounded(value, ceiling):
    value = float(value)
    if not 0 <= value <= ceiling:
        raise ValueError(f"Expected a finite value in [0, {ceiling}], got {value}")
    return value


def profile(rows):
    coordinates = []
    for name in FIELDS:
        limit = 2 if name.endswith("brier") else 1
        coordinates.append(fmean(bounded(row[name], limit) / limit for row in rows))
    return coordinates


def check_keys(table, expected, source):
    if table.keys() != expected:
        missing = sorted(expected - table.keys())[:3]
        extra = sorted(table.keys() - expected)[:3]
        raise ValueError(
            f"{source} has missing {missing} or unexpected {extra} records"
        )


def synth_key(row):
    return row["stratum"], row["config"], int(row["replicate"])


def synthetic_tables(folder):
    profiles = read_index(folder / "profiles.csv", synth_key)
    scores = read_index(
        folder / "scores.csv",
        lambda r: (*synth_key(r), r["method"]),
        lambda r: r["method"] in METHODS,
    )
    strata = {target[0] for target in TARGETS.values()}
    keys = {(s, c, i) for s in strata for c in EXPANDED for i in REPLICATES}
    check_keys(profiles, keys, "synthetic profiles")
    check_keys(scores, {(*key, m) for key in keys for m in METHODS}, "synthetic scores")
    centers = {}
    losses = {}
    for stratum in sorted(strata):
        for config in EXPANDED:
            pair = stratum, config
            centers[pair] = profile([profiles[*pair, i] for i in REPLICATES])
            losses[pair] = {
                method: fmean(
                    bounded(scores[*pair, i, method]["brier"], 2) for i in REPLICATES
                )
                for method in METHODS
            }
    return centers, losses


def external_centers(path):
    rows = read_index(path, lambda r: (r["suite"], int(r["seed"])))
    expected = {
        (suite, root + i)
        for suite, (_, root, _, _) in TARGETS.items()
        for i in REPLICATES
    }
    check_keys(rows, expected, "external fit profiles")
    centers = {}
    for suite, (_, root, classes, fit_rows) in TARGETS.items():
        subset = [rows[suite, root + i] for i in REPLICATES]
        if any(
            (int(r["n_classes"]), int(r["n_fit"]), int(r["n_providers"]))
            != (classes, fit_rows, 3)
            for r in subset
        ):
            raise ValueError(f"Input shape does not match the frozen {suite} stratum")
        centers[suite] = profile(subset)
    return centers


def average_rank(value, losses):
    return (
        1 + sum(x < value for x in losses) + (sum(x == value for x in losses) - 1) / 2
    )


def forecast(centers, losses, targets):
    distances, forecasts, methods = [], [], []
    for suite, target in targets.items():
        stratum = TARGETS[suite][0]
        for library, configs in LIBRARIES.items():
            common = {"suite": suite, "stratum": stratum, "library": library}
            candidates = sorted(
                (math.dist(target, centers[stratum, c]), c) for c in configs
            )
            best_distance, chosen = candidates[0]
            for distance, config in candidates:
                distances.append(
                    {
                        **common,
                        "config": config,
                        "distance": distance,
                        "selected": config == chosen,
                        **{
                            f"target_{name}_scaled": v
                            for name, v in zip(FIELDS, target, strict=True)
                        },
                        **{
                            f"synthetic_{name}_scaled": v
                            for name, v in zip(
                                FIELDS, centers[stratum, config], strict=True
                            )
                        },
                    }
                )
            scores = losses[stratum, chosen]
            gap = scores["native"] - scores["prob_lr"]
            direction = (
                "tie"
                if abs(gap) <= 0.001
                else ("native_lower" if gap < 0 else "prob_lr_lower")
            )
            ranks = {k: average_rank(v, scores.values()) for k, v in scores.items()}
            selected = {**common, "config": chosen, "distance": best_distance}
            forecasts.append(
                {
                    **selected,
                    "native_minus_prob_lr": gap,
                    "direction": direction,
                    **{f"rank_{m}": ranks[m] for m in METHODS},
                }
            )
            for method in sorted(METHODS, key=lambda m: (ranks[m], m)):
                methods.append(
                    {
                        **selected,
                        "method": method,
                        "mean_brier": scores[method],
                        "rank": ranks[method],
                    }
                )
    return distances, forecasts, methods


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--synthetic",
        type=Path,
        required=True,
        help="Directory with synthetic profiles.csv and scores.csv",
    )
    parser.add_argument(
        "--profiles", type=Path, required=True, help="External fusion-fit profile CSV"
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    centers, losses = synthetic_tables(args.synthetic)
    distances, forecasts, methods = forecast(
        centers, losses, external_centers(args.profiles)
    )
    args.out.mkdir(parents=True, exist_ok=True)
    for name, rows in [
        ("distances", distances),
        ("forecasts", forecasts),
        ("method_losses", methods),
    ]:
        write_rows(args.out / (name + ".csv"), rows)


if __name__ == "__main__":
    main()

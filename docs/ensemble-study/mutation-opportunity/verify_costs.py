#!/usr/bin/env python3
"""Reconcile the mutation-opportunity CSV ledgers using the Python standard library.

    python verify_costs.py --output costs-verified.json

Without file arguments the checker reads the original and supplemental CSVs
beside this script. Optional --per-run writes every seed, policy and endpoint.

Counts refer to generated candidate occurrences and observed objective calls.
The intermediate native proposal in identity/refresh is counted through the
point event. The CSV records selected operator sites, which also makes it
possible to count how frequently a parameter refresh could change a program.
The files provide cumulative totals and no lineage-level survival records.
"""

import argparse
import csv
import gzip
import hashlib
import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

SCENARIOS = {
    "committee",
    "deadweight",
    "needle",
    "saturated",
    "correlated",
    "calibrated_suppressor",
    "polarity_fault",
    "hardveto",
    "hardrescue",
    "weakest_link",
    "two_pathways",
    "parity",
    "compound",
    "gated_experts",
    "native_switch",
    "compositional_switch",
    "native_close_switch",
    "miscalibration",
    "multiclass_committee",
}
SHIFT = {
    "two_pathways",
    "parity",
    "weakest_link",
    "correlated",
    "calibrated_suppressor",
    "polarity_fault",
    "native_switch",
    "native_close_switch",
    "compositional_switch",
}
ARMS = ("stock", "identity", "refresh", "retype")
SHADOW = "refresh_independent"
SEEDS = (661, 673)
ENDPOINTS = ("calls", "generation")
FAMILIES = {
    "MeanNode",
    "MinNode",
    "MaxNode",
    "WeightedMeanNode",
    "CloseThresholdNode",
    "FarThresholdNode",
}
PARAMETRIC = {"WeightedMeanNode", "CloseThresholdNode", "FarThresholdNode"}
COUNTS = [
    "fit_calls",
    "validation_calls",
    "initial_trees",
    "native_fitness_batches",
    "completed_generations",
    "crossover_events",
    "crossover_children",
    "point_eligible_crossover_children",
    "structural_mutations",
    "point_eligible_structural_mutations",
    "mutation_calls",
    "mutation_children",
    "joined_trees",
    "dedup_removed",
    "point_opportunities",
    "point_family_changes",
    "point_program_changes",
    "point_original_parametric",
]
META = COUNTS + [
    "calls",
    "generation",
    "archive_trees",
    "providers",
    "classes",
    "path_digest",
    "search_seconds",
    "family_transitions",
]
DECISION_META = ("nodes", "program", "archive_selection_brier")
WANTED = {"fit", "validation"}


def require(condition, context, reason):
    if not condition:
        raise ValueError(f"{context}: {reason}")


def number(value):
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0 or not numeric.is_integer():
        raise ValueError(f"Expected a nonnegative integer, got {value!r}")
    return int(numeric)


def digest(value):
    return len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def proportion(a, b):
    return a / b if b else None


def inspect_endpoint(key, rows):
    scenario, _, _, arm, endpoint = key
    first = next(iter(rows.values()))
    domains = {"fit", "validation", "iid"} | ({"shift"} if scenario in SHIFT else set())
    require(
        set(rows) == set(itertools.product(WANTED, domains)),
        key,
        "incomplete decisions or split coverage",
    )
    require(
        all(all(row[s] == first[s] for s in META) for row in rows.values()),
        key,
        "endpoint metadata differs by row",
    )
    for decision in WANTED:
        example = rows[decision, decision]
        require(
            all(
                all(rows[decision, s][col] == example[col] for col in DECISION_META)
                for s in domains
            ),
            key,
            "winners or archive scores differ between splits",
        )
        require(
            abs(float(example["brier"]) - 2 * float(example["archive_selection_brier"]))
            < 2e-11,
            key,
            f"{decision} raw Brier disagrees with published full-class Brier",
        )
        competitor = "fit" if decision == "validation" else "validation"
        require(
            float(example["brier"])
            <= float(rows[competitor, decision]["brier"]) + 2e-11,
            key,
            f"{decision} archive winner loses to alternate selection on its objective",
        )
    require(
        all(digest(r["prediction_sha256"]) for r in rows.values())
        and digest(first["path_digest"]),
        key,
        "invalid digest",
    )
    counts = {name: number(first[name]) for name in COUNTS}
    fit_calls, batches, generation = (
        counts["fit_calls"],
        counts["native_fitness_batches"],
        number(first["generation"]),
    )
    initial, joined, removed = (
        counts["initial_trees"],
        counts["joined_trees"],
        counts["dedup_removed"],
    )
    crossovers, crossover_children, mutation_children = (
        counts["crossover_events"],
        counts["crossover_children"],
        counts["mutation_children"],
    )
    structural, mutation_calls = (
        counts["structural_mutations"],
        counts["mutation_calls"],
    )
    eligible_children = counts["point_eligible_crossover_children"]
    eligible_structural = counts["point_eligible_structural_mutations"]
    opportunities, changed_families, changes, parametric = (
        counts[s]
        for s in (
            "point_opportunities",
            "point_family_changes",
            "point_program_changes",
            "point_original_parametric",
        )
    )
    require(
        fit_calls == number(first["calls"])
        and 0
        < counts["validation_calls"]
        == number(first["archive_trees"])
        <= fit_calls,
        key,
        "archive or fit-call count mismatch",
    )
    require(
        (fit_calls == 1200 and counts["completed_generations"] == generation - 1)
        if endpoint == "calls"
        else (generation == 10 == counts["completed_generations"]),
        key,
        "wrong endpoint clock",
    )
    require(
        initial == min(number(first["providers"]), 30) + 30
        and batches == generation + 1,
        key,
        "unaccounted singleton, initialization seeds or native cohort",
    )
    target = min(40, initial)
    require(
        crossover_children
        == 2 * crossovers
        == 2 * math.ceil(3 * target / 2) * generation,
        key,
        "crossover construction budget mismatch",
    )
    require(
        0 <= structural <= mutation_children <= crossover_children
        and mutation_children <= mutation_calls <= 2 * mutation_children
        and mutation_calls - structural <= mutation_children,
        key,
        "structural/parameter calls inconsistent with distinct child occurrences",
    )
    require(
        eligible_structural <= min(structural, eligible_children)
        and eligible_children <= crossover_children
        and opportunities <= eligible_structural,
        key,
        "eligible child/event nesting violated",
    )
    parent_instances = joined - crossover_children - mutation_children
    require(
        initial + generation - 1
        <= parent_instances
        <= initial + (generation - 1) * target
        and removed < joined,
        key,
        "invalid joined cohort",
    )
    # All generated cohorts have undergone deduplication. At the calls endpoint,
    # the last such cohort can contain proposals still waiting for evaluation.
    pending = initial + joined - removed - fit_calls
    max_final_batch = target + 2 * math.ceil(3 * target / 2) * 2
    require(
        0 <= pending < max_final_batch and (endpoint == "calls" or pending == 0),
        key,
        "scored, pending and discarded occurrences do not conserve constructions",
    )
    transitions = Counter()
    for transition, count in json.loads(first["family_transitions"]).items():
        old, proposed, returned = transition.split("/")
        require(
            {old, proposed, returned} <= FAMILIES
            and old != proposed
            and isinstance(count, int)
            and count > 0,
            key,
            "invalid native alternative",
        )
        require(
            arm != "stock"
            and (returned == proposed if arm == "retype" else returned == old),
            key,
            "returned family inconsistent with policy",
        )
        transitions[old, proposed, returned] = count
    require(
        sum(transitions.values()) == opportunities
        and sum(n for (old, _, ret), n in transitions.items() if old != ret)
        == changed_families
        and sum(n for (old, _, _), n in transitions.items() if old in PARAMETRIC)
        == parametric,
        key,
        "transition table disagrees with opportunities, old parametric family or returned family",
    )
    require(
        (changes == changed_families == opportunities)
        if arm == "retype"
        else (
            changed_families == 0
            and (changes <= parametric if arm in ("refresh", SHADOW) else changes == 0)
        ),
        key,
        "changed offspring programs contradict point policy",
    )
    # Structural and later numerical mutation can act on the same child.
    # The native loop adds one returned child for their union.
    derived = {
        "generated_candidate_instances": initial
        + crossover_children
        + mutation_children,
        "joined_parent_instances": parent_instances,
        "structural_and_parameter_mutation_children": mutation_calls
        - mutation_children,
        "structural_only_mutation_children": structural
        - (mutation_calls - mutation_children),
        "parameter_only_mutation_children": mutation_children - structural,
        "native_alternative_constructions": opportunities,
        "full_cohort_fit_evaluations": initial + joined - removed,
        "pending_fit_in_generated_cohort": pending,
        "repeated_fit_evaluations": fit_calls - counts["validation_calls"],
        "intra_cohort_duplicate_rate": removed / joined,
        "repeated_fit_rate": 1 - counts["validation_calls"] / fit_calls,
        "selected_per_eligible_structural": proportion(
            opportunities, eligible_structural
        ),
        "program_changes_per_selected": proportion(changes, opportunities),
    }
    record = dict(
        zip(("scenario", "data_seed", "gp_seed", "arm", "checkpoint"), key, strict=True)
    )
    record.update(
        counts,
        generation=generation,
        parent_population_target=target,
        path_digest=first["path_digest"],
        family_transitions=first["family_transitions"],
        **derived,
    )
    for selection in sorted(WANTED):
        for split in domains:
            record[f"{selection}_winner_{split}_brier"] = float(
                rows[selection, split]["brier"]
            )
    return record, transitions


def check_pair(key, earlier, later):
    early, es = earlier
    late, ls = later
    for name in COUNTS:
        require(
            early[name] <= late[name], key, f"{name} decreases along same trajectory"
        )
    require(all(ls[k] >= n for k, n in es.items()), key, "transition counts decrease")
    for selection in WANTED:
        name = f"{selection}_winner_{selection}_brier"
        require(
            late[name] <= early[name] + 2e-11,
            key,
            f"{selection} archive loss increases",
        )
    if early["fit_calls"] == late["fit_calls"]:
        require(
            early["path_digest"] == late["path_digest"],
            key,
            "same objective count, divergent path",
        )
    else:
        require(
            early["path_digest"] != late["path_digest"],
            key,
            "different prefixes, same path",
        )
    if early["generation"] == late["generation"]:
        for name in set(COUNTS) - {
            "fit_calls",
            "validation_calls",
            "completed_generations",
        }:
            require(
                early[name] == late[name],
                key,
                f"{name} changes after proposals already generated",
            )
        require(
            late["fit_calls"] - early["fit_calls"]
            == early["pending_fit_in_generated_cohort"],
            key,
            "missing last part of generated cohort",
        )


def add_rows(path, groups, arms):
    all_names = set()
    with gzip.open(path, "rt") if path.suffix == ".gz" else path.open() as f:
        for row in csv.DictReader(f):
            all_names.add(row["scenario"])
            if row["method"] not in arms:
                require(
                    row["method"] in {"mean", "top_n", "stack_logit"},
                    row["method"],
                    "unexpected method",
                )
                continue
            key = (
                row["scenario"],
                number(row["data_seed"]),
                number(row["gp_seed"]),
                row["method"],
                row["checkpoint"],
            )
            pair = (row["selection"], row["split"])
            require(pair not in groups[key], key, "duplicate decision/split row")
            groups[key][pair] = row
    return all_names


def read_layer(path, sensitivity=None):
    groups = defaultdict(dict)
    all_names = add_rows(path, groups, ARMS)
    arms = ARMS
    if sensitivity:
        require(
            add_rows(sensitivity, groups, (SHADOW,)) == all_names,
            sensitivity,
            "supplemental and original scenario sets differ",
        )
        arms = (*ARMS, SHADOW)
    wine = all_names == {"wine"}
    require(wine or all_names == SCENARIOS, path, "unexpected or missing scenarios")
    cases = itertools.product(
        sorted(all_names), range(5) if wine else range(5100, 5106), SEEDS, arms
    )
    expected = {(*case, endpoint) for case in cases for endpoint in ENDPOINTS}
    require(
        set(groups) == expected,
        path,
        "missing or unexpected bank, GP seed, arm or endpoint",
    )
    outputs = {key: inspect_endpoint(key, rows) for key, rows in groups.items()}
    overlap = defaultdict(Counter)
    refresh_comparison = defaultdict(Counter)
    unique_fit, unique_val = 0, 0
    for key, (refresh, _) in outputs.items():
        if key[3] not in ("refresh", SHADOW):
            continue
        sham_key = (*key[:3], "identity", key[4])
        sham, _ = outputs[sham_key]
        same_path = refresh["path_digest"] == sham["path_digest"]
        zero_change = refresh["point_program_changes"] == 0
        common = refresh_comparison[f"{key[3]}_{key[4]}"]
        no_opportunity_on_sham = sham["point_original_parametric"] == 0
        common["identity_runs_without_selected_parametric_site"] += int(
            no_opportunity_on_sham
        )
        common["same_path_without_selected_parametric_site_on_identity"] += int(
            no_opportunity_on_sham and same_path
        )
        if no_opportunity_on_sham:
            require(
                zero_change and same_path,
                key,
                "refresh diverges before identity selects a parametric family",
            )
        common["runs"] += 1
        common["no_returned_program_change"] += int(zero_change)
        common["same_path_as_identity"] += int(same_path)
        if zero_change:
            require(
                same_path
                and all(
                    refresh[name] == sham[name]
                    for name in [*COUNTS, "family_transitions"]
                ),
                key,
                "identity and zero-change refresh do not reproduce the same ledger",
            )
            for label, row in groups[key].items():
                reference = groups[sham_key][label]
                require(
                    number(row["nodes"]) == number(reference["nodes"])
                    and float(row["archive_selection_brier"])
                    == float(reference["archive_selection_brier"])
                    and row["program"] == reference["program"]
                    and row["prediction_sha256"] == reference["prediction_sha256"],
                    key,
                    "zero-change refresh differs from identity in an archived winner or prediction",
                )
    for key in sorted(outputs):
        if key[-1] != "calls":
            continue
        fixed, gen = outputs[key], outputs[(*key[:-1], "generation")]
        fc, gc = fixed[0]["fit_calls"], gen[0]["fit_calls"]
        if fixed[0]["generation"] <= 10:
            check_pair(key, fixed, gen)
            require(fc <= gc, key, "clock order mismatch")
            last = gen[0]
        else:
            check_pair(key, gen, fixed)
            require(gc < fc, key, "clock order mismatch")
            last = fixed[0]
        unique_fit += last["fit_calls"]
        unique_val += last["validation_calls"]
        position = "earlier" if fc < gc else "same_fit_count" if fc == gc else "later"
        overlap[key[3]][position] += 1
    summary = {}
    for arm, ep in itertools.product(arms, ENDPOINTS):
        subset = [v[0] for k, v in outputs.items() if k[-2:] == (arm, ep)]

        def total(s, records=subset):
            return sum(r[s] for r in records)

        summary[f"{arm}_{ep}"] = {
            "trajectories": len(subset),
            **{f"total_{s}": total(s) for s in COUNTS},
            **{
                f"total_{s}": total(s)
                for s in (
                    "pending_fit_in_generated_cohort",
                    "repeated_fit_evaluations",
                    "generated_candidate_instances",
                    "native_alternative_constructions",
                    "structural_and_parameter_mutation_children",
                    "structural_only_mutation_children",
                    "parameter_only_mutation_children",
                )
            },
            "cohorts_with_computed_proposals_still_to_score": sum(
                r["pending_fit_in_generated_cohort"] > 0 for r in subset
            ),
            "pooled_intra_cohort_duplicate_rate": total("dedup_removed")
            / total("joined_trees"),
            "pooled_repeated_fit_rate": total("repeated_fit_evaluations")
            / total("fit_calls"),
            "pooled_selected_per_eligible_structural": proportion(
                total("point_opportunities"),
                total("point_eligible_structural_mutations"),
            ),
            "pooled_genotype_change_per_opportunity": proportion(
                total("point_program_changes"), total("point_opportunities")
            ),
        }
    coverage = {}
    refresh_controls = (
        ("identity", "refresh", SHADOW) if sensitivity else ("identity", "refresh")
    )
    for arm, ep in itertools.product(refresh_controls, ENDPOINTS):
        per_case = defaultdict(Counter)
        for k, (r, _) in outputs.items():
            if k[-2:] == (arm, ep):
                c = per_case[k[0]]
                c["runs"] += 1
                c["selected_parametric_sites"] += r["point_original_parametric"]
                c["runs_with_selected_parametric_site"] += int(
                    r["point_original_parametric"] > 0
                )
        coverage[f"{arm}_{ep}"] = {
            "scenarios_with_any_selected_parametric_site": sum(
                v["selected_parametric_sites"] > 0 for v in per_case.values()
            ),
            "runs_with_selected_parametric_site": sum(
                v["runs_with_selected_parametric_site"] for v in per_case.values()
            ),
            "exposed_runs_by_scenario": {
                scenario: counts["runs_with_selected_parametric_site"]
                for scenario, counts in per_case.items()
            },
        }
    return {
        "file": path.name,
        "sensitivity_file": sensitivity.name if sensitivity else None,
        "sensitivity_sha256": hashlib.sha256(sensitivity.read_bytes()).hexdigest()
        if sensitivity
        else None,
        "parametric_exposure_coverage": coverage,
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "trajectory_count": len(groups) // 2,
        "checkpoint_count": len(groups),
        "GP_output_rows": sum(len(rows) for rows in groups.values()),
        "fit_calls_through_both_endpoints_once_per_trajectory": unique_fit,
        "validation_calls_through_both_endpoints_once_per_trajectory": unique_val,
        "calls_1200_before_or_after_generation_end": dict(overlap),
        "refresh_compared_to_identity": dict(refresh_comparison),
        "arms": summary,
    }, [v[0] for _, v in sorted(outputs.items())]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", type=Path, nargs="*")
    parser.add_argument("--per-run", type=Path)
    parser.add_argument(
        "--output", type=Path, help="Write the verified JSON here instead of stdout."
    )
    parser.add_argument(
        "--refresh-independent",
        nargs="+",
        type=Path,
        metavar="CSV",
        help="Optional independently seeded sensitivity results, ordered to match base files.",
    )
    args = parser.parse_args()
    if not args.files:
        directory = Path(__file__).resolve().parent
        args.files = [directory / "runs.csv.gz", directory / "wine-runs.csv.gz"]
        if not args.refresh_independent:
            args.refresh_independent = [
                directory / "refresh-independent-runs.csv.gz",
                directory / "wine-refresh-independent-runs.csv.gz",
            ]
    if args.refresh_independent and len(args.refresh_independent) != len(args.files):
        parser.error("provide one --refresh-independent CSV per base file (same order)")
    summaries, rows = [], []
    supplements = args.refresh_independent or [None] * len(args.files)
    for path, supplement in zip(args.files, supplements, strict=True):
        summary, layer_rows = read_layer(path, supplement)
        summaries.append(summary)
        rows.extend(layer_rows)
    if args.per_run:
        fields = list(dict.fromkeys(k for r in rows for k in r))
        with args.per_run.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    result = json.dumps(summaries, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(result + "\n", encoding="utf-8")
    else:
        print(result)


if __name__ == "__main__":
    main()

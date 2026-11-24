"""Describe logit-head coefficients in saved native/challenger sample trees.

Run weights.py ROUND_DIR for a JSON summary, or add --view banks / --view
providers for CSV. The sample file alone supplies both stages and all weights.
Absolute weights exclude the binary intercept and use the original log-evidence
scale. Summary medians weight sampled banks equally. The selected-head
comparison requires a flat or augmented choice in both arms; their feature
sets can differ. Flat-head coefficients are paired separately by provider ID.
"""

import argparse
import csv
import gzip
import json
import math
import sys
from collections import Counter
from pathlib import Path
from statistics import fmean, median

KEY = ("round", "scenario", "replicate", "data_seed", "gp_seed")
ARMS = ("native", "challenger")
STAGES = {"discovery": "discovery_trees", "refit": "trees"}


def head(tree, *, flat=False):
    operators = tree["children"]
    if tree["type"] != "ValueNode" or not operators:
        raise ValueError("Expected a native tree with a fitted final head")
    node = operators[-1]
    fields = {"BinaryAffineNode": "coefficients", "WeightedLogitMeanNode": "weights"}
    if node["type"] not in fields:
        raise ValueError(f"Unexpected fitted head {node['type']}")
    weights = node[fields[node["type"]]]
    ids = [tree["id"], *(child["id"] for child in node["children"])]
    if len(weights) != len(ids) or not all(math.isfinite(w) for w in weights):
        raise ValueError("Expected one finite weight for each native head slot")
    if flat and (
        len(operators) != 1
        or any(c["children"] for c in node["children"])
        or len(ids) != len(set(ids))
    ):
        raise ValueError("Flat head must contain one slot per distinct raw provider")
    return ids, weights, node.get("intercept")


def paired_records(path):
    groups = {}
    with gzip.open(path, "rt") as source:
        for line in source:
            row = json.loads(line)
            key = tuple(row[field] for field in KEY)
            arms = groups.setdefault(key, {})
            if row["method"] in arms:
                raise ValueError(f"Duplicate {row['method']} sample for {key}")
            arms[row["method"]] = row
    if not groups:
        raise ValueError("No sampled trees")
    for key, arms in groups.items():
        if set(arms) != set(ARMS):
            raise ValueError(f"Expected a native/challenger pair for {key}")
        yield dict(zip(KEY, key, strict=True)), arms


def compare(path):
    banks, providers = [], []
    for identity, arms in paired_records(path):
        for stage, field in STAGES.items():
            key = {**identity, "stage": stage}
            bank = key.copy()
            flat = {}
            for arm in ARMS:
                sample = arms[arm]
                ids, weights, _ = head(sample[field]["flat"], flat=True)
                flat[arm] = dict(zip(ids, weights, strict=True))
                bank[f"{arm}_flat_slots"] = len(ids)
                choice = sample["chosen"]
                bank[f"{arm}_choice"] = choice
                if choice == "gp":
                    selected = []
                    intercept = None
                elif choice in ("flat", "augmented"):
                    _, selected, intercept = head(sample[field][choice])
                else:
                    raise ValueError(f"Unknown selected candidate {choice}")
                bank[f"{arm}_selected_slots"] = len(selected)
                bank[f"{arm}_selected_mean_abs"] = (
                    fmean(map(abs, selected)) if selected else None
                )
                bank[f"{arm}_selected_intercept"] = intercept
            shared = [
                provider
                for provider in flat["native"]
                if provider in flat["challenger"]
            ]
            if not shared:
                raise ValueError(f"No common raw providers for {key}")
            pairs = [(flat["native"][p], flat["challenger"][p]) for p in shared]
            bank["shared_providers"] = len(pairs)
            for arm in ARMS:
                bank[f"{arm}_flat_shared_mean_abs"] = fmean(
                    abs(flat[arm][p]) for p in shared
                )
            bank["flat_mean_abs_paired_difference"] = fmean(
                abs(b - a) for a, b in pairs
            )
            for provider, (a, b) in zip(shared, pairs, strict=True):
                providers.append(
                    {
                        **key,
                        "provider": provider,
                        "native": a,
                        "challenger": b,
                        "difference": b - a,
                        "abs_weight_change": abs(b) - abs(a),
                    }
                )
            banks.append(bank)
    return banks, providers


def paired_sizes(rows, measure):
    pairs = [(r["native_" + measure], r["challenger_" + measure]) for r in rows]
    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
    return {
        "paired_banks": len(pairs),
        "smaller_challenger_banks": sum(b < a for a, b in pairs),
        "equal_banks": sum(b == a for a, b in pairs),
        "median_native_bank_mean_abs": median(a for a, _ in pairs) if pairs else None,
        "median_challenger_bank_mean_abs": median(b for _, b in pairs)
        if pairs
        else None,
        "median_paired_change": median(b - a for a, b in pairs) if pairs else None,
    }


def summarize(banks):
    result = {}
    for stage in STAGES:
        rows = [r for r in banks if r["stage"] == stage]
        result[stage] = {
            "sampled_banks": len(rows),
            "choices": {
                arm: dict(Counter(r[arm + "_choice"] for r in rows)) for arm in ARMS
            },
            "shared_flat_provider_slots": sum(r["shared_providers"] for r in rows),
            "flat_head_on_shared_providers": paired_sizes(rows, "flat_shared_mean_abs"),
            "median_bank_mean_abs_paired_difference": median(
                r["flat_mean_abs_paired_difference"] for r in rows
            ),
            "selected_flat_or_augmented_heads": paired_sizes(rows, "selected_mean_abs"),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "round_dir", type=Path, help="Round directory containing sample_trees.jsonl.gz"
    )
    parser.add_argument(
        "--view", choices=("summary", "banks", "providers"), default="summary"
    )
    args = parser.parse_args()
    banks, providers = compare(args.round_dir / "sample_trees.jsonl.gz")
    if args.view == "summary":
        print(json.dumps(summarize(banks), indent=2, allow_nan=False))
    else:
        rows = banks if args.view == "banks" else providers
        writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

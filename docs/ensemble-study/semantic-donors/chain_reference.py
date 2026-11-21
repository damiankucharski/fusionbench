"""Exhaust nonconstant monotone functions and one join of two raw-input chains."""

import argparse
import itertools
import json
from pathlib import Path


def lattice(n):
    # Shannon expansion: low and high cofactors are monotone and low <= high.
    family = {0, 1}
    for depth in range(n):
        shift = 1 << depth
        family = {
            low | (high << shift)
            for low, high in itertools.product(family, repeat=2)
            if not low & ~high
        }
    return family - {0, (1 << (1 << n)) - 1}


def classify(n):
    columns = tuple(
        sum(1 << i for i in range(1 << n) if i & (1 << j)) for j in range(n)
    )
    rules = lattice(n)
    chains = set(columns)
    while True:
        wider = (
            chains
            | {f & g for f, g in itertools.product(chains, columns)}
            | {f | g for f, g in itertools.product(chains, columns)}
        )
        if wider == chains:
            break
        chains = wider
    canal = {f for f in rules if any((f & ~c) == 0 or (c & ~f) == 0 for c in columns)}
    two = (
        chains
        | {f & g for f, g in itertools.product(chains, repeat=2)}
        | {f | g for f, g in itertools.product(chains, repeat=2)}
    )
    assert chains <= canal <= rules and two <= rules
    return rules, chains, canal, two, columns


def save(folder):
    rules, chains, canal, two, columns = classify(4)
    # Independent check of all four-input truth tables by testing every Boolean-cube edge.
    assert rules == {
        f
        for f in range(1, 65535)
        if all(
            not ((f >> a) & 1) or (f >> (a | 1 << j)) & 1
            for a in range(16)
            for j in range(4)
        )
    }
    target = (columns[0] & columns[1]) | (columns[2] & columns[3])
    assert target not in canal and target in two
    five, chain5, canal5, two5, _ = classify(5)
    maj5 = sum(1 << i for i in range(32) if i.bit_count() >= 3)
    assert maj5 in five - two5
    report = {
        "all_nonconstant_monotone": len(rules),
        "chain_closure": len(chains),
        "with_necessary_or_sufficient_input": len(canal),
        "without_either": len(rules - canal),
        "two_chains_joined": len(two),
        "matched_rule_bits": target,
        "chain_rule_bits": sorted(chains),
        "canalizing_rule_bits": sorted(canal),
        "join_of_two_chains_rule_bits": sorted(two),
        "exploratory_five_inputs": {
            "all_nonconstant_monotone": len(five),
            "chain_closure": len(chain5),
            "with_necessary_or_sufficient_input": len(canal5),
            "two_chains_joined": len(two5),
            "unreachable_by_one_join": len(five - two5),
            "majority_not_in_two_chains": maj5 not in two5,
        },
    }
    (folder / "chain-reference.json").write_text(json.dumps(report, indent=2) + "\n")
    print({k: v for k, v in report.items() if not isinstance(v, list)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    save(parser.parse_args().folder)

"""Measure organic and supplied routes through two neutral native subtrees."""

import argparse
import json
from itertools import product
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from okapi.globals import set_backend
from okapi.node import MaxNode, MinNode, ValueNode
from okapi.paired_fitness import arithmetic_mean_tree
from okapi.replacement import select_brier_replacement
from okapi.tree import Tree

from fusionbench.eval.suite_point_gp import predict_tree
from fusionbench.methods.averaging import SimpleAverage, TopN
from fusionbench.methods.stacking import StackingLR

HERE = Path(__file__).resolve().parent
POLICIES = ("ordinary", "open_raw", "open_kl", "strict_kl")


def bank_inputs(bank):
    def generator(stream):
        return np.random.default_rng([5910000, bank, stream])

    a, b = np.array(list(product((0, 1), repeat=2))).T
    permutation = generator(8).permutation(4)
    p = 0.125 + 0.75 * np.column_stack([a, 1 - a, b, 1 - b])[:, permutation]
    cube = np.stack([1 - p, p], axis=-1)
    q = 0.125 + 0.75 * (a != b)
    blocks = []
    for stream in range(8):
        rng = generator(stream)
        indices = rng.integers(4, size=2048 if stream == 7 else 128)
        labels = (rng.random(len(indices)) < q[indices]).astype(int)
        blocks.append((cube[indices], labels))
    return cube, q, permutation, blocks


def witnesses(cube, permutation):
    set_backend("numpy")
    tensors = {f"m{i}.npy": cube[:, i, 1:2] for i in range(4)}

    def leaf(role):
        key = f"m{np.flatnonzero(permutation == role).item()}.npy"
        return ValueNode(None, tensors[key], key)

    def conjunction(first, second):
        root = leaf(first)
        root.add_child(MinNode([leaf(second)]))
        return root

    union = conjunction(0, 3)
    union.add_child(MaxNode([conjunction(1, 2)]))
    return {
        "mean": arithmetic_mean_tree(tensors),
        "u": Tree(conjunction(0, 3)),
        "v": Tree(conjunction(1, 2)),
        "xor": Tree(union),
    }


def full_brier(p, y):
    return np.square(p - np.eye(2)[y]).sum(axis=1).mean()


def barrier(bank, cube, q, permutation, blocks):
    trees = witnesses(cube, permutation)
    losses = {
        key: 2 * np.square(predict_tree(tree, cube)[:, 1] - q).mean()
        for key, tree in trees.items()
    }
    assert losses == {"mean": 0.28125, "u": 0.28125, "v": 0.28125, "xor": 0}
    output = []
    for policy in ("open_kl", "strict_kl"):
        champion = "mean"
        can_compose = False
        for generation, (x, y) in enumerate(blocks[1:7], start=1):
            offered = [champion] * 12
            if generation == 1:
                offered[:2] = ["u", "v"]
            elif generation == 2 and can_compose:
                offered[0] = "xor"
            predictions = {
                key: predict_tree(trees[key], x) for key in set(offered + [champion])
            }
            decision = select_brier_replacement(
                predictions[champion],
                [predictions[key] for key in offered],
                y,
                alpha=0.05 / 6,
                rule="kl",
            )
            if generation == 1:
                can_compose = policy == "open_kl" or {0, 1}.issubset(decision.eligible)
            previous = champion
            if decision.index is not None:
                champion = offered[decision.index]
            output.append(
                {
                    "bank": bank,
                    "generation": generation,
                    "policy": policy,
                    "previous": previous,
                    "offered_1": offered[0],
                    "offered_2": offered[1],
                    "eligible_1": 0 in decision.eligible,
                    "eligible_2": 1 in decision.eligible,
                    "lower_1": decision.lower_bounds[0],
                    "champion": champion,
                    "excess": losses[champion],
                }
            )
    return output


def live(bank, cube, q, blocks):
    from fusionbench.eval.replacement_search import run_replacement_search

    fit_x, fit_y = blocks[0]
    test_x, test_y = blocks[-1]
    data = []
    for policy in POLICIES:
        result = run_replacement_search(
            fit_x,
            fit_y,
            blocks[1:7],
            seed=7910000 + bank,
            policy=policy,
            incumbent="mean",
        )
        p = predict_tree(result.tree, cube)
        observed = [
            2 * np.square(predict_tree(r.tree, cube)[:, 1] - q).mean()
            for r in result.records
            if r.generation > 0
        ]
        data.append(
            {
                "bank": bank,
                "method": policy,
                "exact_excess": 2 * np.square(p[:, 1] - q).mean(),
                "test_brier": full_brier(predict_tree(result.tree, test_x), test_y),
                "best_offered_excess": min(observed),
                "found_exact": min(observed) < 1e-12,
                "unique_providers": result.unique_providers,
                "unconfirmed_mating_ancestor": result.records[
                    result.champion_id
                ].has_unconfirmed_ancestor,
                "proposal_count": result.proposal_count,
                "path": result.path_digest,
            }
        )
    assert len({r["path"] for r in data[:3]}) == 1
    for source, source_blocks in (("discovery", blocks[:1]), ("all", blocks[:7])):
        x, y = (np.concatenate(parts) for parts in zip(*source_blocks, strict=True))
        for name, model in (
            ("mean", SimpleAverage()),
            ("top_n", TopN("brier")),
            ("logistic", StackingLR()),
        ):
            model.fit(x, y, task="binary", n_classes=2)
            p = model.predict(cube)
            data.append(
                {
                    "bank": bank,
                    "method": f"{name}_{source}",
                    "exact_excess": 2 * np.square(p[:, 1] - q).mean(),
                    "test_brier": full_brier(model.predict(test_x), test_y),
                }
            )
    return data


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("experiment", choices=["live", "supplied"])
    ap.add_argument("--out", type=Path, default=HERE)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    rows = []
    for bank in range(32):
        cube, q, permutation, blocks = bank_inputs(bank)
        if args.experiment == "supplied":
            rows.extend(barrier(bank, cube, q, permutation, blocks))
        else:
            rows.extend(live(bank, cube, q, blocks))
    pd.DataFrame(rows).to_csv(args.out / f"{args.experiment}.csv", index=False)
    (args.out / f"{args.experiment}-time.json").write_text(
        json.dumps({"seconds": perf_counter() - start}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()

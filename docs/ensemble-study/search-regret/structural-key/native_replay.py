"""Replay native OKAPI without overriding the deduplication or selection code.

Set PYTHONPATH to the requested OKAPI checkout and FusionBench checkout.
Run each policy in a fresh interpreter. These checkpoints count objective calls.
"""

import argparse
import csv
import subprocess
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import okapi
from loguru import logger
from okapi import Okapi
from okapi.operators import MAX, MIN

from fusionbench.scenarios.gates import TwoPathways

CHECKPOINTS = (300, 1200)
SEARCH_SEEDS = (151, 157, 163, 167)
COMMITS = {
    "legacy": "ec1ba021cb82f13f8b364eb31d812788666aad06",
    "structural": "86f21cdd580564f7e84bed9b326caad977ee2325",
}


class ReplayComplete(Exception):
    pass


def recursive_key(node):
    return node.code, tuple(recursive_key(child) for child in node.children)


class NativeReplay(Okapi):
    """Fingerprint evaluations around the native batched objective."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, objective_functions=(self._fitness,), **kwargs)
        self.calls = 0
        self.sequence = sha256()
        self.best_loss = np.inf
        self.winner = None
        self.snapshots = []

    def _fitness(self, prediction, truth):
        errors = prediction.ravel() - truth.ravel()
        loss = float(np.mean(np.square(errors)))
        self.batch_losses.append(loss)
        return 1.0 - loss

    def _calculate_fitnesses(self, trees=None):
        trees = self.population if trees is None else trees
        allowed = CHECKPOINTS[-1] - self.calls
        batch = trees[:allowed]
        self.batch_losses = []
        scores = super()._calculate_fitnesses(batch)
        for tree, loss in zip(batch, self.batch_losses, strict=True):
            record = (repr(recursive_key(tree.root)) + "\n").encode("utf-8")
            self.sequence.update(record)
            self.calls += 1
            if loss < self.best_loss:
                self.best_loss = loss
                self.winner = sha256(record).hexdigest()
            if self.calls in CHECKPOINTS:
                self.snapshots.append(
                    {
                        "calls": self.calls,
                        "sequence_sha256": self.sequence.hexdigest(),
                        "fit": self.best_loss,
                        "fit_champion_sha256": self.winner,
                    }
                )
        if self.calls == CHECKPOINTS[-1]:
            raise ReplayComplete
        return scores


def replay(panel_seed, search_seeds):
    x, labels, _ = TwoPathways(rep=2, accuracy=0.9)._generate(
        128, np.random.default_rng([panel_seed, 0])
    )
    with TemporaryDirectory() as directory:
        root = Path(directory)
        paths = [root / f"p{i}.npy" for i in range(x.shape[1])]
        for i, path in enumerate(paths):
            np.save(path, x[:, i, 1:2])
        label_path = root / "labels.npy"
        np.save(label_path, labels)
        for seed in search_seeds:
            engine = NativeReplay(
                paths,
                label_path,
                8,
                2,
                2,
                backend="numpy",
                seed=seed,
                allowed_ops=(MIN, MAX),
                reuse_survivor_fitness=False,
                postprocessing_function=np.asarray,
            )
            try:
                while True:
                    engine.run_iteration()
            except ReplayComplete:
                for row in engine.snapshots:
                    yield {"data_seed": panel_seed, "gp_seed": seed, **row}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True, choices=COMMITS)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--data-start", type=int, default=3500)
    parser.add_argument("--data-count", type=int, default=12)
    parser.add_argument("--search-seeds", type=int, nargs="+", default=SEARCH_SEEDS)
    args = parser.parse_args()
    checkout = Path(okapi.__file__).resolve().parent
    commit = subprocess.check_output(
        ["git", "-C", checkout, "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != COMMITS[args.policy]:
        raise RuntimeError(
            f"Expected {COMMITS[args.policy]} for {args.policy}, got {commit}"
        )
    tracked_changes = subprocess.check_output(
        ["git", "-C", checkout, "status", "--short", "--untracked-files=no"], text=True
    )
    if tracked_changes:
        raise RuntimeError(f"The native checkout has local changes: {tracked_changes}")
    logger.remove()
    with args.out.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "policy",
                "commit",
                "data_seed",
                "gp_seed",
                "calls",
                "sequence_sha256",
                "fit",
                "fit_champion_sha256",
            ],
        )
        writer.writeheader()
        for data_seed in range(args.data_start, args.data_start + args.data_count):
            writer.writerows(
                {"policy": args.policy, "commit": commit, **row}
                for row in replay(data_seed, args.search_seeds)
            )
            stream.flush()


if __name__ == "__main__":
    main()

"""Replay archived decisions and forecasts from the public adapter on selected banks."""

import argparse
import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from fusionbench import generate_dataset
from fusionbench.methods.environment_screen import EnvironmentScreenedOKAPI
from fusionbench.scenarios.source_environments import (
    SOURCE_ENVIRONMENT_PRESETS,
    source_environment_bank,
)

OUT = Path(__file__).resolve().parent / "results"


def digest(values):
    return sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def test_bank(family, seed, worlds):
    if family == "grouped_rare_event":
        bank = generate_dataset(family, seed=seed, n_per_split=512)
        x, y = bank.fit_preds, bank.fit_y
        scene = bank.meta["evaluation_groups"]["fit"]["scene_id"]
        rng = np.random.default_rng([seed, 217])
        rows = np.sort(
            [rng.choice(np.flatnonzero(scene == s)) for s in np.unique(scene)]
        )
        # Recompute proxies on the same diagnostic rows as the study. The adapter
        # accepts IDs aligned to the full source and reads only these positions.
        p = x[rows].mean(axis=1)
        disagreement = np.square(x[rows] - p[:, None, :]).sum(2).mean(1)
        env = np.zeros(len(x), dtype=int)
        ctx = np.zeros(len(x), dtype=int)
        env[rows] = disagreement > np.median(disagreement)
        ctx[rows] = p.max(1) > np.median(p.max(1))
        return x, y, env, ctx, rows, {"test": bank.test_preds}
    region, shortcut = SOURCE_ENVIRONMENT_PRESETS[family]
    b = source_environment_bank(region=region, shortcut=shortcut, n_rows=512, seed=seed)
    laws = {
        "s0": (0.5, 0),
        "s50": (0.5, 0.5),
        "s100": (0.5, 1),
        "specialist80": (0.8, 0.5),
    }
    queries = {name: b.target(*laws[name]).x for name in worlds}
    return b.x, b.y, b.env, b.context, np.arange(len(b.y)), queries


def replay(r):
    path = OUT / f"r{r:02}.csv"
    data = pd.read_csv(path)
    base = 840000 + 1000 * r
    cases = (
        ("mixed", base + 100, "adjusted"),
        ("case_mix", base + 101, "raw"),
        ("grouped_rare_event", base + 400, "adjusted"),
    )
    verified = []
    for family, seed, mode in cases:
        rows = data[
            (data.family == family) & (data.seed == seed) & (data.method == mode)
        ]
        if len(rows) == 0:
            raise ValueError(f"Missing {family} seed {seed} in {path}")
        x, y, env, ctx, idx, queries = test_bank(family, seed, rows.world)
        model = EnvironmentScreenedOKAPI(mode=mode).fit(
            x,
            y,
            task="binary",
            n_classes=2,
            environments=env,
            contexts=ctx,
            screen_indices=idx,
        )
        for row in rows.itertuples():
            assert model.drop_ == (None if pd.isna(row.drop) else int(row.drop))
            assert model.screen_info_["reason"] == row.reason
            assert len(idx) == row.n_screen
            assert digest(x) == row.source_x_hash and digest(y) == row.source_y_hash
            assert digest(model.predict(x)) == row.fit_hash
            query = queries[row.world]
            assert digest(query) == row.test_x_hash
            assert digest(model.predict(query)) == row.test_hash
            assert (
                digest(model.predict_retained(query[:, model.kept_indices_, :]))
                == row.test_hash
            )
        verified.append(
            {
                "family": family,
                "seed": seed,
                "mode": mode,
                "diagnostic_rows": len(idx),
                "all_training_rows": len(y),
                "drop": model.drop_,
                "reason": model.screen_info_["reason"],
                "matching_test_hashes": len(rows),
            }
        )
    return {
        "round": r,
        "scores_sha256": sha256(path.read_bytes()).hexdigest(),
        "replays": verified,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("round", type=int, choices=(2, 9))
    r = parser.parse_args().round
    with threadpool_limits(limits=1):
        data = replay(r)
    (OUT / f"r{r:02}-adapter-replay.json").write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(data))

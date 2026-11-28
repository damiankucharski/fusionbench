"""Train the registered cost-sensitive configuration rule only on allowed round-1 families."""

import argparse
import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
from okapi.configuration import POOL_FEATURES, fit_configuration_policy, recommend

from fusionbench.transfer import load_preset

OUT = Path(__file__).resolve().parent / "results"
ACTIONS = ("mean", "top_n", "cal_mean", "flat", "cal_pull")
EXCLUDED = frozenset(
    {
        "grouped_rare_event",
        "shared_error_bloc",
        "rare_scene_specialist",
        "cloned_shortcut",
        "parity",
        "similar_classes_38",
    }
)
KEY = ["family", "seed"]


def train_frame():
    allowed = sorted(set(load_preset("standard-v6")["scenarios"]) - EXCLUDED)
    if len(allowed) != 24:
        raise ValueError(
            "The development protocol includes exactly 24 standard-v6 families"
        )
    fps = pd.read_csv(OUT / "r1-fingerprints.csv", float_precision="round_trip")
    scores = pd.read_csv(OUT / "r1-scores.csv", float_precision="round_trip")
    features = fps.loc[
        (fps.cohort == "standard") & fps.family.isin(allowed), [*KEY, *POOL_FEATURES]
    ]
    costs = scores[
        (scores.cohort == "standard")
        & scores.family.isin(allowed)
        & scores.method.isin(ACTIONS)
    ]
    pivot = costs.pivot(index=KEY, columns="method", values="brier").loc[
        :, list(ACTIONS)
    ]
    frame = (
        features.set_index(KEY)
        .join(pivot, how="outer", validate="one_to_one")
        .reset_index()
        .sort_values(KEY)
    )
    expected = pd.MultiIndex.from_product([allowed, range(611400, 611404)], names=KEY)
    if not frame.set_index(KEY).index.equals(expected) or frame.isna().any().any():
        raise ValueError("Missing development features or costs")
    return frame, allowed


def fit_frame(frame):
    return fit_configuration_policy(
        frame[list(POOL_FEATURES)].to_dict(orient="records"),
        {a: frame[a].to_numpy() for a in ACTIONS},
        frame.family.tolist(),
    )


def train():
    frame, families = train_frame()
    policy = fit_frame(frame)
    lock = {
        "training_round": 1,
        "training_families": families,
        "excluded_families": sorted(EXCLUDED),
        "row_count": len(frame),
        "training_sha256": sha256(frame.to_csv(index=False).encode()).hexdigest(),
        "input_sha256": {
            name: sha256((OUT / name).read_bytes()).hexdigest()
            for name in ("r1-scores.csv", "r1-fingerprints.csv")
        },
        "constant": policy["tree"]["action"],
        "policy": policy,
    }
    loo = []
    for held in families:
        fitted = fit_frame(frame[frame.family != held])
        constant = fitted["tree"]["action"]
        for _, row in frame[frame.family == held].iterrows():
            name = recommend(row, fitted)["action"]
            loo.append(
                {
                    "held_family": held,
                    "seed": int(row.seed),
                    "action": name,
                    "constant": constant,
                    "tree_brier": row[name],
                    "constant_brier": row[constant],
                    "oracle_brier": np.min([row[a] for a in ACTIONS]),
                }
            )
    OUT.mkdir(exist_ok=True)
    name = OUT / "r1-policy.json"
    data = json.dumps(lock, indent=2, allow_nan=False) + "\n"
    if name.exists() and name.read_text() != data:
        raise RuntimeError("The frozen policy differs from the recomputed policy")
    name.write_text(data)
    pd.DataFrame(loo).to_csv(OUT / "r1-leave-family-out.csv", index=False)
    print(
        f"Trained on {len(frame)} banks in {len(families)} families; constant={lock['constant']}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["train"])
    parser.parse_args()
    train()

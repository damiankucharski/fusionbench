"""Replay decisions independently from archived per-scene or per-row cross-fitted losses."""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
ACTIONS = ("mean", "top_n", "native")
FIELDS = ["family", "seed", "variant"]


def reconstruct(e):
    if e["fallback_reason"] is not None:
        raise AssertionError(
            "The registered study does not contain a fallback-sized bank"
        )
    loss = np.column_stack([e["unit_loss"][name] for name in ACTIONS])
    fold = np.asarray(e["unit_fold"])
    difficulty = np.asarray(e["unit_disagreement"])
    assert loss.shape[1] == 3 and len(loss) == e["n_units"] and set(fold) == {0, 1, 2}

    def gains_vs(reference):
        return loss[:, reference, None] - loss

    def se(gain):
        return np.sqrt(
            np.sum((gain - gain.mean()) ** 2) / ((len(gain) - 1) * len(gain))
        )

    # Use individual action means for exact ties; the flat memory order of a
    # stacked array can otherwise introduce sub-ulp differences in reductions.
    risk = np.array([np.mean(e["unit_loss"][a]) for a in ACTIONS])
    best, simple = int(risk.argmin()), int(risk[:2].argmin())
    simple_gain = gains_vs(2)[:, simple]
    guard = simple_gain.mean() >= se(simple_gain)
    high = difficulty > np.median(difficulty)
    return {
        "cv": ACTIONS[best],
        "one_se": next(
            a
            for j, a in enumerate(ACTIONS)
            if risk[j] - risk[best] <= se(gains_vs(best)[:, j])
        ),
        "paired": ACTIONS[simple] if guard else "native",
        "sign": ACTIONS[simple]
        if guard and all(simple_gain[fold == i].mean() > 0 for i in range(3))
        else "native",
        "stress": ACTIONS[simple]
        if guard and all(simple_gain[high == i].mean() >= 0 for i in np.unique(high))
        else "native",
    }


def audit(r):
    raw = pd.read_csv(OUT / f"r{r:02}.csv", float_precision="round_trip")
    rows = raw.set_index(FIELDS + ["method"]).sort_index()
    with gzip.open(OUT / f"r{r:02}-source.jsonl.gz", "rt") as file:
        traces = [json.loads(line) for line in file]
    checked = 0
    for trace in traces:
        key = tuple(trace[c] for c in FIELDS)
        evidence = trace["evidence"]
        if evidence:
            computed = reconstruct(evidence)
            for name, action in trace["decisions"].items():
                if name in computed:
                    assert (
                        action == computed[name] == rows.at[(*key, name), "action"]
                    ), (key, name, action, computed[name])
                    checked += 1
        if r == 1:
            action = min(ACTIONS, key=lambda a: rows.at[(*key, a), "fit_brier"])
            assert action == trace["decisions"]["apparent"]
            checked += 1
    shortcut_count = 0
    if r in (5, 10):
        triplets = raw[
            raw.family.isin(["shortcut_000", "shortcut_050", "shortcut_100"])
        ]
        for _, block in triplets.groupby(["seed", "method"]):
            assert len(block) == 3 and block.target_hash.nunique() == 3
            assert (
                block[
                    [
                        "source_x_hash",
                        "source_y_hash",
                        "test_x_hash",
                        "action",
                        "fit_hash",
                        "test_hash",
                    ]
                ]
                .nunique()
                .eq(1)
                .all()
            )
            shortcut_count += 1
    out = dict(
        conditions=len(traces),
        decisions_from_raw_unit_or_apparent_losses=checked,
        matched_identical_shortcut_triplets=shortcut_count,
    )
    (OUT / f"r{r:02}-choice-audit.json").write_text(json.dumps(out, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("round", type=int, choices=range(1, 11))
    audit(parser.parse_args().round)

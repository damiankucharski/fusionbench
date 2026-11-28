"""Measure fixed witnesses and three unprivileged references after the protocol lock."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from inputs import POOLS, SUPPORT, generate, score, support_targets
from threadpoolctl import threadpool_limits

from fusionbench.methods.averaging import SimpleAverage, TopN
from fusionbench.methods.stacking import StackingLR

OUT = Path(__file__).resolve().parent / "results"


def references(bank):
    from okapi.replacement import select_brier_replacement

    targets, _, _ = support_targets(bank)
    fit = generate(bank, 21, 128)
    blocks = [generate(bank, i, 128) for i in range(31, 37)]
    test = generate(bank, 41, 2048)
    power = generate(bank, 51, 128)
    mean, witness = (targets[k][power.indices] for k in POOLS)
    rows, control = [], []
    for law in POOLS:
        for slots in (12, 1):
            for gate in ("kl", "raw"):
                check = select_brier_replacement(
                    mean,
                    [witness] + [mean] * (slots - 1),
                    power.labels[law],
                    alpha=0.05 / 6,
                    rule=gate,
                )
                control.append(
                    {
                        "bank": bank,
                        "law": law,
                        "gate": gate,
                        "slots": slots,
                        "empirical_gain": check.gains[0],
                        "lower_bound": check.lower_bounds[0],
                        "accept": check.index == 0,
                        "true_gain": float(
                            np.mean(
                                2
                                * (
                                    (targets["null"] - targets[law]) ** 2
                                    - (targets["signal"] - targets[law]) ** 2
                                )
                            )
                        ),
                    }
                )
        for name, pieces in (("discovery", [fit]), ("all_development", [fit, *blocks])):
            x = np.concatenate([d.predictions for d in pieces])
            y = np.concatenate([d.labels[law] for d in pieces])
            for method, model in (
                ("mean", SimpleAverage()),
                ("top_n", TopN("brier")),
                ("prob_lr", StackingLR()),
            ):
                model.fit(x, y, task="binary", n_classes=2)
                p = model.predict(SUPPORT)[:, 1]
                rows.append(
                    {
                        "bank": bank,
                        "law": law,
                        "split": name,
                        "n_labels": len(y),
                        "method": method,
                        **score(p, targets[law], test, law),
                        "top_n": model.n_selected_ if method == "top_n" else None,
                        "ranking": json.dumps(model.ranking_.tolist())
                        if method == "top_n"
                        else None,
                        "prefix_brier": json.dumps(model.prefix_scores_.tolist())
                        if method == "top_n"
                        else None,
                        "p1": json.dumps(p.tolist(), separators=(",", ":")),
                    }
                )
    return rows, control


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    rows, checks = [], []
    with threadpool_limits(limits=1):
        for bank in range(48):
            base, power = references(bank)
            rows.extend(base)
            checks.extend(power)
    pd.DataFrame(rows).to_csv(OUT / "baselines.csv.gz", index=False)
    pd.DataFrame(checks).to_csv(OUT / "direct-witness.csv", index=False)

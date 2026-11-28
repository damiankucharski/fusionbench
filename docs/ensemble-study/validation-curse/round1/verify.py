"""Recalculate every fixed-menu selection using exact integer losses on the grid."""

import argparse
import csv
from itertools import product
from pathlib import Path
from statistics import fmean, stdev

import numpy as np
from scipy.stats import t

HERE = Path(__file__).resolve().parent


def read(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def verify(folder):
    cells = read(folder / "selections.csv")
    table = {
        (
            r["law"],
            int(r["replicate"]),
            int(r["validation_rows"]),
            int(r["menu_size"]),
        ): r
        for r in cells
    }
    assert len(cells) == len(table) == 768
    ledgers = read(folder / "banks.csv")
    bank = {(r["law"], int(r["replicate"]), r["split"]): r for r in ledgers}
    assert len(ledgers) == len(bank) == 320
    controls = read(folder / "references.csv")
    control = {(r["law"], int(r["replicate"]), r["method"]): r for r in controls}
    assert len(control) == len(controls) == 208
    for rep in range(32):
        seed = 4_100_000 + rep
        risks = {}
        for key, positive in [("mean_null", 0), ("information", 1)]:
            risks[key] = (
                np.r_[
                    0.4,
                    0.4 - 0.02 * positive if positive else 0.42,
                    np.repeat(0.42, 62),
                ],
                np.r_[0.352, 0.412 if positive else 0.372, np.repeat(0.372, 62)],
            )
        split_losses = {}
        for j, (split, n) in enumerate(
            [
                ("fit", 640),
                ("validation", 512),
                ("audit", 2048),
                ("iid", 4096),
                ("shift", 4096),
            ]
        ):
            masses = [0.4, 0.1, 0.1, 0.4] if j == 4 else [0.25] * 4
            rng = np.random.default_rng([seed, j])
            anchor = rng.choice([2, 4, 6, 8], n, p=masses)
            code = rng.integers(0, 64, n)
            u = rng.random(n)
            # The six-bit state selects one of two parities for each nonzero mask.
            sign = np.array([1 if k.bit_count() % 2 == 0 else -1 for k in range(64)])
            ticks = anchor[:, None] + sign[np.bitwise_and(code[:, None], np.arange(64))]
            ticks[:, 0] = anchor
            null, info = [bank[k, rep, split] for k in risks]
            assert all(
                null[k] == info[k]
                for k in [
                    "seed",
                    "stream",
                    "n",
                    "features_sha256",
                    "label_uniforms_sha256",
                ]
            )
            for law, signal in [("mean_null", 0), ("information", 1)]:
                eta = (
                    anchor + signal * (-1 if split == "shift" else 1) * sign[code & 1]
                ) / 10
                y = (u < eta).astype(int)
                # Each per-row full Brier is an integer multiple of 1/50.
                sq = (ticks - 10 * y[:, None]) ** 2
                np.testing.assert_allclose(
                    sq[:, 0].mean() / 50,
                    float(control[law, rep, "mean"][split]),
                    atol=1e-13,
                    rtol=0,
                )
                if split != "fit":
                    split_losses[law, split] = sq
        for law, v, k in product(risks, (32, 128, 512), (1, 4, 16, 64)):
            r = table[law, rep, v, k]
            assert int(r["seed"]) == seed
            sums = split_losses[law, "validation"][:v, :k].sum(0)
            w = int(np.argmin(sums))
            assert w == int(r["winner"])
            val, base = sums[w] / (50 * v), sums[0] / (50 * v)
            expected = {"validation": val, "validation_anchor": base}
            for split in ["audit", "iid", "shift"]:
                test = split_losses[law, split]
                expected[split] = test[:, w].mean() / 50
                expected[split + "_anchor"] = test[:, 0].mean() / 50
            expected["optimism"] = expected["audit"] - val
            expected["extra_optimism"] = expected["optimism"] - (
                expected["audit_anchor"] - base
            )
            for idx, tag in enumerate(["source", "shift"]):
                risk = risks[law][idx]
                expected[tag + "_risk"] = risk[w]
                expected[tag + "_selection_regret"] = risk[w] - min(risk[:k])
                expected[tag + "_menu_penalty"] = min(risk[:k]) - min(risk)
                assert (r[tag + "_best_hit"] == "True") == (risk[w] == min(risk[:k]))
            expected["exact_optimism"] = expected["source_risk"] - val
            expected["exact_extra_optimism"] = expected["exact_optimism"] - (
                risks[law][0][0] - base
            )
            np.testing.assert_allclose(
                [float(r[m]) for m in expected],
                list(expected.values()),
                rtol=0,
                atol=1e-12,
            )
    for row in read(folder / "selection-means.csv"):
        metric = row["metric"]
        sample = [
            table[row["law"], i, int(row["validation_rows"]), int(row["menu_size"])][
                metric
            ]
            for i in range(32)
        ]
        vals = [
            float(v == "True") if metric == "source_best_hit" else float(v)
            for v in sample
        ]
        mean = fmean(vals)
        width = t.ppf(0.975, 31) * stdev(vals) / np.sqrt(32)
        np.testing.assert_allclose(
            [float(row[k]) for k in ("mean", "low", "high")],
            [mean, mean - width, mean + width],
            atol=1e-12,
            rtol=0,
        )
    print(
        f"{len(cells)} chosen rules, held-out losses and exact regrets reproduced on integer probability ticks; {len(ledgers)} split records and {len(control)} comparator records."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "results")
    verify(parser.parse_args().out)

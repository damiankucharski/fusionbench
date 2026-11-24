"""Score cross-half wrong-class predictions on earlier fusion-fit banks."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from fusionbench import get_scenario

HERE = Path(__file__).resolve().parent
HISTORICAL = HERE.parent / "external-coverage"
POPULATIONS = ("all", "top1_errors")
BASELINES = ("uniform", "popularity")
N_PERMUTATIONS = 100


def historical_banks():
    with np.load(HISTORICAL / "external-results/fit_banks.npz") as saved:
        for seed in range(3_140_000, 3_140_008):
            key = f"digits_{seed}"
            yield "digits", seed, saved[key + "_inputs"], saved[key + "_labels"]
    for panel, bias in (("panel_B", 0.0), ("panel_E", 3.0)):
        generator = get_scenario(
            "multiclass_provider_panel",
            n_classes=10,
            separation=6,
            votes=[31, 7],
            confusion_bias=bias,
        )
        for seed in range(3_160_000, 3_160_008):
            x, y, _ = generator.sample(449, np.random.default_rng([seed, 0]))
            yield panel, seed, x.astype(np.float32), y


def make_bin_weights(x, y):
    eligible = np.arange(x.shape[2])[None, :] != y[:, None]
    wrong = np.where(eligible[:, None, :], x, -np.inf)
    tied = wrong == wrong.max(2, keepdims=True)
    mass = tied / tied.sum(2, keepdims=True)
    np.testing.assert_allclose(mass.sum(2), 1, atol=1e-15)
    return mass, eligible


def class_stratified_half(y):
    half = np.zeros(len(y), dtype=bool)
    for label in np.unique(y):
        half[np.flatnonzero(y == label)[1::2]] = True
    return half


def gains(mass, y, half, included, eligible):
    """Return event-averaged gains versus uniform and pooled popularity, by provider."""
    classes = mass.shape[-1]
    weighted = mass * included[:, :, None]
    sums = np.zeros((mass.shape[1], 2))
    allowed = ~np.eye(classes, dtype=bool)[:, None, :]
    for training in (half, ~half):
        counts = np.zeros((classes, *mass.shape[1:]))
        np.add.at(counts, y[training], weighted[training])
        conditional = counts + allowed * (2 / (classes - 1))
        conditional /= conditional.sum(2, keepdims=True)
        pooled = (counts.sum(0) + 2 / classes) * allowed
        pooled /= pooled.sum(2, keepdims=True)
        heldout = ~training
        valid = eligible[heldout, None, :]
        cond_log = np.log(np.where(valid, conditional[y[heldout]], 1))
        pooled_log = np.log(np.where(valid, pooled[y[heldout]], 1))
        observed = weighted[heldout]
        sums[:, 0] += (observed * (cond_log + np.log(classes - 1))).sum((0, 2))
        sums[:, 1] += (observed * (cond_log - pooled_log)).sum((0, 2))
    counts = included.sum(0)[:, None]
    return np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)


def shuffled(rng, mass, slots, values):
    order = np.argsort(rng.random(values.shape), axis=2)
    out = np.zeros_like(mass)
    np.put_along_axis(out, slots, np.take_along_axis(values, order, axis=2), axis=2)
    return out


def bank(source, seed, x, y):
    n, m, c = x.shape
    assert (n, m, c) == (449, 3, 10) and y.shape == (n,)
    mass, eligible = make_bin_weights(x, y)
    masks = (np.ones((n, m), dtype=bool), x.argmax(2) != y[:, None])
    half = class_stratified_half(y)
    rows = []
    observed = np.empty((len(masks), len(BASELINES)))
    for j, (population, mask) in enumerate(zip(POPULATIONS, masks, strict=True)):
        result = gains(mass, y, half, mask, eligible)
        observed[j] = np.nanmean(result, axis=0)
        for provider in range(m):
            included = mask[:, provider]
            all_tied = np.count_nonzero(mass[:, provider], axis=1) == c - 1
            rows.append(
                {
                    "source": source,
                    "seed": seed,
                    "population": population,
                    "provider": provider,
                    "events": int(included.sum()),
                    "all_false_tied_events": int((all_tied & included).sum()),
                    **{
                        f"gain_vs_{b}": v
                        for b, v in zip(BASELINES, result[provider], strict=True)
                    },
                }
            )
    # Put permuted mass only into the eligible classes, keeping the query truth empty.
    slots = np.broadcast_to(np.arange(c), (n, c))[eligible].reshape(n, 1, c - 1)
    slots = np.broadcast_to(slots, (n, m, c - 1))
    values = np.take_along_axis(mass, slots, axis=2)
    reference = np.empty((N_PERMUTATIONS, len(masks), len(BASELINES)))
    for perm in range(N_PERMUTATIONS):
        rng = np.random.default_rng([seed, 62017, perm])
        randomized = shuffled(rng, mass, slots, values)
        for j, mask in enumerate(masks):
            reference[perm, j] = np.nanmean(
                gains(randomized, y, half, mask, eligible), axis=0
            )
    return rows, observed, reference


def write(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run(out):
    banks = []
    results = defaultdict(list)
    for source, seed, x, y in historical_banks():
        records, observed, reference = bank(source, seed, x, np.asarray(y).ravel())
        banks.extend(records)
        results[source].append((observed, reference))
    summaries = []
    permutations = []
    for source, rows in results.items():
        assert len(rows) == 8
        observed = np.mean([r[0] for r in rows], axis=0)
        ref = np.mean([r[1] for r in rows], axis=0)
        for i, pop in enumerate(POPULATIONS):
            stats = [
                r for r in banks if r["source"] == source and r["population"] == pop
            ]
            for j, baseline in enumerate(BASELINES):
                key = {"source": source, "population": pop, "baseline": baseline}
                null = ref[:, i, j]
                low, high = np.quantile(null, [0.025, 0.975])
                summaries.append(
                    {
                        **key,
                        "banks": len(rows),
                        "nonempty_provider_banks": sum(r["events"] > 0 for r in stats),
                        "events": sum(r["events"] for r in stats),
                        "all_false_tied_events": sum(
                            r["all_false_tied_events"] for r in stats
                        ),
                        "observed_gain": observed[i, j],
                        "null_mean": null.mean(),
                        "null_low": low,
                        "null_high": high,
                    }
                )
                permutations.extend(
                    {**key, "permutation": k, "gain": v} for k, v in enumerate(null)
                )
    out.mkdir(parents=True, exist_ok=True)
    for filename, rows in [
        ("banks.csv", banks),
        ("summary.csv", summaries),
        ("permutations.csv", permutations),
    ]:
        write(out / filename, rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "mechanism")
    run(parser.parse_args().out)

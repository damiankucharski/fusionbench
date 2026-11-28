"""Decompose archived R10 forecasts on three fixed target laws, without refitting."""

import gzip
import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
from banks import PREVIOUS, PROFILES
from threadpoolctl import threadpool_limits

from fusionbench.scenarios.rare_shortcut_noise import RareShortcutNoise
from fusionbench.scenarios.source_environments import source_environment_bank

OUT = Path(__file__).resolve().parent / "results"
METHODS = ("native", "full", "cap2", "guarded")
WORLDS = ("s0", "s100", "specialist80")
COMPONENTS = ("excess", "squared_bias", "variance")


def digest(values):
    return sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def endpoint(world):
    region, shortcut = PREVIOUS["TARGETS"][world]
    return RareShortcutNoise(
        region_test=region, shortcut_test=shortcut, provider_noise_test=0
    ).support("test")


def read_forecasts():
    score_path, trace_path = OUT / "r10.csv", OUT / "r10-source.jsonl.gz"
    frame = pd.read_csv(score_path, float_precision="round_trip")
    with gzip.open(trace_path, "rt") as stream:
        traces = [t for line in stream if (t := json.loads(line))["cohort"] == "focus"]
    if len({(t["family"], t["seed"]) for t in traces}) != len(traces):
        raise ValueError("Duplicated focal source")
    source = frame[
        (frame.cohort == "focus")
        & frame.method.isin(METHODS)
        & frame.world.isin(WORLDS)
    ]
    expected = pd.MultiIndex.from_tuples(
        [
            (t["family"], t["seed"], world, method)
            for t in traces
            for world in WORLDS
            for method in METHODS
        ],
        names=["family", "seed", "world", "method"],
    )
    scores = source.set_index(expected.names).sort_index()
    if not scores.index.sort_values().equals(expected.sort_values()):
        raise ValueError(
            "Archived forecasts and scores must cover the same focal banks"
        )
    laws = {world: endpoint(world) for world in WORLDS}
    aligned = {}
    max_error = 0.0
    for t in traces:
        family, seed = t["family"], t["seed"]
        region, shortcut = PROFILES[family]
        bank = source_environment_bank(
            region=region,
            shortcut=shortcut,
            seed=seed,
            n_rows=128 if family == "small128" else 512,
            context_noise=0.25 if family == "noisy_context" else 0.0,
        )
        if t["source_x_hash"] != digest(bank.x) or t["source_y_hash"] != digest(bank.y):
            raise ValueError("Regenerated source differs from archived source")
        forecasts = np.asarray(
            [t["exact_forecasts"][method] for method in METHODS], dtype=float
        )
        if forecasts.shape != (len(METHODS), 16, 2) or not np.isfinite(forecasts).all():
            raise ValueError("Expected 16 full-class forecasts for every method")
        np.testing.assert_allclose(forecasts.sum(axis=2), 1, atol=1e-6, rtol=0)
        for world in WORLDS:
            target = bank.target(*PREVIOUS["TARGETS"][world])
            votes, mass, q = laws[world]
            # Restore generator role order only in this post-fit audit. The
            # runner presents anonymous, permuted provider columns to OKAPI.
            restored = target.x[:, np.argsort(bank.provider_order), 1] > 0.5
            np.testing.assert_array_equal(restored, votes[:, :, 1] > 0.5)
            np.testing.assert_array_equal(
                restored @ np.array([8, 4, 2, 1]), np.arange(16)
            )
            np.testing.assert_array_equal(target.mass, mass)
            np.testing.assert_array_equal(target.q, q)
            ref = scores.loc[(family, seed, world)].reindex(METHODS)
            if (
                not ref.test_x_hash.eq(digest(target.x)).all()
                or not ref.target_hash.eq(digest(np.column_stack((mass, q)))).all()
            ):
                raise ValueError(
                    "Target reports or joint law differ from the original evaluation"
                )
            if ref.test_hash.tolist() != [digest(p) for p in forecasts]:
                raise ValueError("Prediction trace differs from its archived test hash")
            truth = np.column_stack([1 - q, q])
            per_method = np.square(forecasts - truth).sum(axis=2) @ mass
            archive_excess = (ref.brier - ref.oracle_brier).to_numpy()
            max_error = max(
                max_error, float(np.max(np.abs(per_method - archive_excess)))
            )
            np.testing.assert_allclose(per_method, archive_excess, rtol=0, atol=2e-12)
        aligned.setdefault(family, {})[seed] = forecasts
    evidence = {
        "sources": len(traces),
        "verified_forecasts": len(scores),
        "max_abs_error_against_archived_excess": max_error,
        "scores_sha256": sha256(score_path.read_bytes()).hexdigest(),
        "trace_sha256": sha256(trace_path.read_bytes()).hexdigest(),
    }
    return aligned, laws, evidence


def decompose(forecasts, mass, q, bootstrap):
    """Return absolute components and paired deltas with the same source resamples."""
    truth = np.column_stack([1 - q, q])
    avg = forecasts.mean(axis=0)
    per_seed = np.square(forecasts - truth).sum(axis=-1) @ mass
    bias = np.square(avg - truth).sum(axis=-1) @ mass
    var = np.square(forecasts - avg).sum(axis=-1).mean(axis=0) @ mass
    excess = per_seed.mean(axis=0)
    np.testing.assert_allclose(excess, bias + var, atol=2e-15, rtol=0)
    n, k, patterns, classes = forecasts.shape
    boot_mean = (bootstrap @ forecasts.reshape(n, -1)).reshape(-1, k, patterns, classes)
    boot_excess = bootstrap @ per_seed
    boot_bias = np.square(boot_mean - truth).sum(axis=-1) @ mass
    boot_var = boot_excess - boot_bias
    result = []
    for j, method in enumerate(METHODS):
        row = {"method": method, "sources": n}
        for name, point, reps in zip(
            COMPONENTS,
            (excess, bias, var),
            (boot_excess, boot_bias, boot_var),
            strict=True,
        ):
            delta = reps[:, j] - reps[:, 0]
            lo, hi = np.quantile(delta, [0.025, 0.975])
            row.update(
                {
                    name: point[j],
                    f"delta_{name}": point[j] - point[0],
                    f"lo_{name}": lo,
                    f"hi_{name}": hi,
                }
            )
        result.append(row)
    return result


def main():
    aligned, laws, evidence = read_forecasts()
    # Each resample takes whole training draws and reuses the same indices for
    # every method and world. The 16 support points are not sampling units.
    seeds = sorted(next(iter(aligned.values())))
    n = len(seeds)
    if n != 20 or any(sorted(s) != seeds for s in aligned.values()):
        raise ValueError(
            "Expected the registered 20 matched seeds in every focal design"
        )
    bootstrap_seed = 103010
    ids = np.random.default_rng(bootstrap_seed).integers(n, size=(10_000, n))
    bootstrap = np.column_stack([(ids == i).mean(axis=1) for i in range(n)])
    rows = []
    for family, bank in sorted(aligned.items()):
        p = np.stack([bank[s] for s in seeds])
        for world, (_, mass, q) in laws.items():
            rows.extend(
                {"family": family, "world": world, **v}
                for v in decompose(p, mass, q, bootstrap)
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "r10-decomposition.csv", index=False)
    evidence.update(
        rows=len(frame),
        independent_seeds=n,
        bootstrap_draws=len(ids),
        bootstrap_seed=bootstrap_seed,
        max_abs_identity_error=float(
            np.max(np.abs(frame.excess - frame.squared_bias - frame.variance))
        ),
    )
    (OUT / "r10-decomposition.json").write_text(
        json.dumps(evidence, indent=2, allow_nan=False) + "\n"
    )
    print(evidence)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()

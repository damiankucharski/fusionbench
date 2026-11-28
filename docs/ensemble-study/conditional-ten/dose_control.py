"""Secondary confirmation contrasts isolating the source trigger at fixed weighting strength."""

import json
from collections import Counter
from time import perf_counter

import numpy as np
import pandas as pd
from banks import _focus
from run import OUT, decisions, digest, fit_model, measures
from threadpoolctl import threadpool_limits

METHODS = ("native", "half", "guarded", "single")
WORLDS = ("s0", "s50", "s100", "specialist80")


def _key(xhash, yhash, whash, x_test, target):
    return xhash, yhash, whash, x_test, target


def _stored_cache(raw):
    bank = {}
    for row in raw[raw.method.isin((*METHODS, "replicated"))].itertuples():
        key = _key(
            row.source_x_hash,
            row.source_y_hash,
            row.weight_hash,
            row.test_x_hash,
            row.target_hash,
        )
        value = {
            "brier": row.brier,
            "test_hash": row.test_hash,
            "fit_hash": row.fit_hash,
            "tree_hash": row.tree_hash,
            "path_hash": row.path_hash,
        }
        if key in bank:
            if bank[key][0] != value:
                raise ValueError(
                    "Identical fit inputs and target law disagree in registered benchmark rows"
                )
        else:
            bank[key] = value, f"archived/{row.family}/{row.method}"
    return bank


def _weights(b, rows):
    _, plans, _ = decisions(b, 10)
    full, _ = plans["guarded"]
    half, evidence = plans["half"]
    if (
        digest(full) != rows.loc[("s0", "guarded"), "weight_hash"]
        or digest(half) != rows.loc[("s0", "half"), "weight_hash"]
    ):
        raise ValueError(
            "Reconstructed training plan differs from the registered predictions"
        )
    # Same mathematical interpolation and summation order as the registered half.
    always = 0.5 + 0.5 * full
    np.testing.assert_allclose(always, 1 + 0.5 * (full - 1), rtol=0, atol=4e-16)
    if evidence["gate_active"]:
        np.testing.assert_array_equal(always, half)
    return full, always, half, evidence


def _fit(b, weights):
    queries = {digest(x): x for _, x, _, _, _ in b.worlds}
    fields, predictions = fit_model("always_half", b, weights, queries)
    values = {}
    for world, x, labels, mass, q in b.worlds:
        p = predictions[digest(x)]
        values[world] = {
            "brier": measures(p, labels, mass, q)["brier"],
            "test_hash": digest(p),
            **{name: fields[name] for name in ("fit_hash", "tree_hash", "path_hash")},
        }
    return values


def _verify_controls(b, matched, half):
    checks = []
    for method, weights in (("native", np.ones(len(b.y))), ("half", half)):
        fitted = _fit(b, weights)
        for world in WORLDS:
            archive = matched.loc[(world, method)]
            values = fitted[world]
            if any(
                values[field] != archive[field]
                for field in ("test_hash", "fit_hash", "tree_hash", "path_hash")
            ):
                raise ValueError(f"Control replay mismatch: {method}/{world}")
            np.testing.assert_allclose(
                values["brier"], archive.brier, rtol=0, atol=2e-15
            )
        checks.append(
            {
                "family": b.family,
                "seed": int(b.seed),
                "method": method,
                "world_forecasts_checked": len(WORLDS),
                "fit_hash": fitted["s0"]["fit_hash"],
            }
        )
    return checks


def collect(raw):
    cache = _stored_cache(raw)
    fit_count = 0
    reuse = Counter()
    controls = []
    rows = []
    keyed = raw.set_index(["family", "seed", "world", "method"], verify_integrity=True)
    seeds = sorted(raw.seed.unique())
    if (
        raw.family.nunique() != 9
        or len(seeds) != 24
        or not raw.groupby("family").seed.nunique().eq(24).all()
    ):
        raise ValueError("Expected all 216 registered focal fits")
    seen = set()
    for b in _focus(10, seeds):
        if (b.family, b.seed) in seen:
            raise ValueError("Repeated focal fit")
        seen.add((b.family, b.seed))
        b.screen_idx = b.idx
        src = keyed.loc[(b.family, b.seed)].sort_index()
        xh, yh = digest(b.x), digest(b.y)
        if not (src.source_x_hash.eq(xh).all() and src.source_y_hash.eq(yh).all()):
            raise ValueError("Input differs from archived source")
        full, weights, half, ev = _weights(b, src)
        active = bool(ev["gate_active"])
        uniform = np.ptp(full) == 0
        wh = digest(weights)
        if not controls and active and not uniform:
            controls = _verify_controls(b, src, half)
        keys = {}
        for world, x, labels, mass, q in b.worlds:
            xtest, target = (
                digest(x),
                digest(labels if q is None else np.column_stack((mass, q))),
            )
            original = src.loc[(world, "native")]
            if (xtest, target) != (original.test_x_hash, original.target_hash):
                raise ValueError("A registered target differs from the replay")
            keys[world] = _key(xh, yh, wh, xtest, target)
        if set(keys) != set(WORLDS):
            raise ValueError("Expected the four registered targets")
        if any(key not in cache for key in keys.values()):
            fitted = _fit(b, weights)
            fit_count += 1
            kind = "new_fit"
            for world, key in keys.items():
                cache[key] = fitted[world], f"secondary/{b.family}/{b.seed}"
        elif active and not uniform:
            kind = "registered_half"
        elif uniform:
            kind = "registered_uniform"
        else:
            kind = "other_identical_fit"
        reuse[kind] += 1
        for world, key in keys.items():
            observed, donor = cache[key]
            history = src.xs(world, level="world")
            for field in ("oracle_brier", "test_x_hash", "target_hash"):
                if history[field].nunique() != 1:
                    raise ValueError("Methods must be scored on matching targets")
            # The trigger never makes a difference when it fires or all weights are equal.
            if (active or uniform) and (
                observed["test_hash"] != history.at["half", "test_hash"]
                or history.at["single", "test_hash"]
                != history.at["guarded", "test_hash"]
            ):
                raise ValueError(
                    "Matched trigger control changed an active or uniform source"
                )
            rows.append(
                {
                    "family": b.family,
                    "seed": int(b.seed),
                    "world": world,
                    "trigger": active,
                    "guarded_weights_nonuniform": not uniform,
                    "guarded_policy": history.at["guarded", "policy"],
                    "always_half_source": kind,
                    "reused_key_origin": donor,
                    "source_x_hash": xh,
                    "source_y_hash": yh,
                    "unconditional_weight_hash": digest(full),
                    "always_half_weight_hash": wh,
                    "always_half_fit_hash": observed["fit_hash"],
                    "always_half_test_hash": observed["test_hash"],
                    "target_hash": key[-1],
                    "oracle_brier": history.oracle_brier.iloc[0],
                    **{m: history.at[m, "brier"] for m in METHODS},
                    "always_half": observed["brier"],
                }
            )
    if len(seen) != 216 or len(rows) != 864 or not controls:
        raise ValueError("Incomplete source, target or control coverage")
    return pd.DataFrame(rows), {
        "unique_missing_fits": fit_count,
        "source_reuse": dict(reuse),
        "reference_fits": controls,
    }


def summarize(raw):
    methods = [*METHODS, "always_half"]
    endpoint = raw[raw.world.isin(("s0", "s100"))].copy()
    endpoint[methods] = endpoint[methods].sub(endpoint.oracle_brier, axis=0)
    endpoint = (
        endpoint.groupby(["family", "seed", "trigger", "guarded_weights_nonuniform"])[
            methods
        ]
        .max()
        .reset_index()
    )
    endpoint["world"] = "max_regret"
    all_metrics = pd.concat([raw, endpoint], ignore_index=True)
    out = []
    for (family, metric), table in all_metrics.groupby(["family", "world"]):
        table = table.sort_values("seed")
        if len(table) != 24 or not table.seed.is_unique:
            raise ValueError(
                "Each paired metric requires one observation per focal source"
            )
        row = {
            "family": family,
            "metric": metric,
            "sources": len(table),
            "triggered": int(table.trigger.sum()),
            "untriggered_nonuniform": int(
                (~table.trigger & table.guarded_weights_nonuniform).sum()
            ),
        }
        differences = (
            table[["half", "single"]].to_numpy()
            - table[["always_half", "guarded"]].to_numpy()
        )
        # Match the confirmation's paired bootstrap for a single fixed profile.
        draws = np.random.default_rng(113010).integers(24, size=(10000, 24))
        for label, delta in (
            ("half_trigger_effect", differences[:, 0]),
            ("full_trigger_effect", differences[:, 1]),
            ("half_minus_full_trigger_effect", differences[:, 0] - differences[:, 1]),
        ):
            lo, hi = np.quantile(delta[draws].mean(1), (0.025, 0.975))
            row.update({label: delta.mean(), label + "_lo": lo, label + "_hi": hi})
        row.update({m: table[m].mean() for m in methods})
        out.append(row)
    return pd.DataFrame(out)


def main():
    start = perf_counter()
    raw = (
        pd.read_csv(OUT / "r10.csv", float_precision="round_trip")
        .query("cohort=='focus'")
        .copy()
    )
    with threadpool_limits(limits=1):
        records, metadata = collect(raw)
        summary = summarize(records)
    records.to_csv(OUT / "r10-dose-control-scores.csv", index=False)
    summary.to_csv(OUT / "r10-dose-control.csv", index=False)
    metadata.update(
        round=10, sources=216, worlds_per_source=4, seconds=perf_counter() - start
    )
    (OUT / "r10-dose-control.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

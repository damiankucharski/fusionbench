"""Rebuild source decisions, replay cheap fits, and check paired intervals from raw risks."""

import argparse
import gzip
import json
from hashlib import sha256
from itertools import combinations
from pathlib import Path
from runpy import run_path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from run_env import sources
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
REFERENCE = run_path(str(HERE.parent / "full-data-five" / "verify_full.py"))
CHEAP = ("mean", "top_n", "lr")
EXAMPLES = {
    "multiclass_committee",
    "grouped_rare_event",
    "correlated",
    "mixed",
    "case_mix",
    "reversed",
    "noisy_context",
    "small128",
}
TOL = 4e-12


def digest(x):
    return sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def screen_scores(x, y, env, ctx, adjusted, z):
    if x.shape[1] == 1:
        return None, "single_provider"
    levels = np.unique(env)
    if len(levels) < 2:
        return None, "single_environment"
    loss = np.square(x.astype(float) - np.eye(x.shape[-1])[y, None, :]).sum(2)
    if not adjusted:
        ctx = np.zeros(len(y), dtype=int)
    strata = np.unique(ctx)
    props = np.array([np.mean(ctx == c) for c in strata])
    mean, variances = [], []
    for e in levels:
        cells = [loss[(env == e) & (ctx == c)] for c in strata]
        if min(map(len, cells)) < (8 if adjusted else 2):
            return None, "insufficient_cell_count"
        mean.append(sum(p * vals.mean(0) for p, vals in zip(props, cells, strict=True)))
        variances.append(
            sum(
                p * p * vals.var(0, ddof=1) / len(vals)
                for p, vals in zip(props, cells, strict=True)
            )
        )
    scores = np.max(
        [
            np.abs(mean[i] - mean[j]) - z * np.sqrt(variances[i] + variances[j])
            for i, j in combinations(range(len(levels)), 2)
        ],
        axis=0,
    )
    return scores, None


def check_screen(b, m, stored):
    env = (
        np.random.default_rng([b.seed, 283]).permutation(b.env)
        if m == "sham"
        else b.env
    )
    scores, issue = screen_scores(
        b.x[b.idx], b.y[b.idx], env, b.ctx, m != "raw", 2.58 if m == "strict" else 1.96
    )
    if issue:
        assert stored["reason"] == issue and stored["drop"] is None
        return
    # Recomputing means/variances in a different summation order may move an
    # exact floating-point tie by one ulp; check declared ties in saved precision.
    actual = np.array(stored["scores"])
    np.testing.assert_allclose(scores, actual, atol=TOL, rtol=0)
    highest = actual.max()
    idx = np.flatnonzero(actual == highest)
    if highest <= (0.15 if m == "strict" else 0.10):
        expect = (None, "below_threshold")
    elif len(idx) != 1:
        expect = (None, "tied_maximum")
    else:
        expect = (int(idx[0]), "unique_above_threshold")
    assert (stored["drop"], stored["reason"]) == expect


def check_interval_records(raw, r):
    table = pd.read_csv(
        OUT / f"r{r:02d}-contrasts.csv", float_precision="round_trip"
    ).set_index(["group", "method", "reference"])
    standard = raw[raw.cohort == "standard"]
    groups = {
        "standard": standard.pivot(
            index=["family", "seed"], columns="method", values="brier"
        )
    }
    for family, part in raw[raw.cohort == "focus"].groupby("family"):
        ep = (
            part[part.world.isin(("s0", "s100"))]
            .groupby(["family", "seed", "method"])
            .brier.max()
            .unstack()
        )
        groups[f"{family}/worst_endpoint"] = ep
        for world, rows in part.groupby("world"):
            groups[f"{family}/{world}"] = rows.pivot(
                index=["family", "seed"], columns="method", values="brier"
            )
    checked = 0
    for key, frame in groups.items():
        frame = frame.sort_index()
        nf = len(frame.index.unique("family"))
        n = len(frame) // nf
        idx = np.random.default_rng(83000 + r).integers(n, size=(10_000, nf, n))
        idx += np.arange(nf)[None, :, None] * n
        for m in frame:
            if m == "native":
                continue
            d = (frame[m] - frame.native).to_numpy()
            lo, hi = np.quantile(d[idx].mean((1, 2)), [0.025, 0.975])
            vals = table.loc[(key, m, "native"), ["delta", "lo", "hi"]].to_numpy()
            np.testing.assert_allclose(vals, [d.mean(), lo, hi], atol=2e-13, rtol=0)
            checked += 1
    return checked


def check_target_geometry(frame):
    """Fixed forecasts integrate against affine mixtures of the two shortcut laws."""
    focus = frame[frame.cohort == "focus"]
    by_source = focus.groupby(["family", "seed", "method"])
    if (
        not by_source[["test_hash", "test_x_hash", "drop"]]
        .nunique(dropna=False)
        .eq(1)
        .all()
        .all()
    ):
        raise ValueError(
            "Forecast, input support or exclusion depends on the target world"
        )
    panel = focus.pivot(
        index=["family", "seed", "method"], columns="world", values="brier"
    )
    error = 0.0
    for world, t in (("s25", 0.25), ("s50", 0.5), ("s75", 0.75)):
        if world in panel:
            error = max(
                error,
                float(
                    np.max(np.abs(panel[world] - (1 - t) * panel.s0 - t * panel.s100))
                ),
            )
    removed = focus[focus["drop"].eq(focus.true_shortcut)].pivot(
        index=["family", "seed", "method"], columns="world", values="brier"
    )
    invariant = (
        float(np.max(np.abs(removed.s0 - removed.s100))) if len(removed) else 0.0
    )
    if max(error, invariant) > 3e-12:
        raise ValueError(
            "The exact target risks violate the shortcut mixture or marginal invariance"
        )
    return {
        "max_brier_interpolation_error": error,
        "true_shortcut_omissions": len(removed),
        "max_risk_change_after_shortcut_omission": invariant,
    }


def check(r):
    frame = pd.read_csv(OUT / f"r{r:02d}.csv", float_precision="round_trip")
    keyed = frame.groupby(["cohort", "family", "seed"])
    first = frame.groupby(["cohort", "family"]).seed.min().to_dict()
    with gzip.open(OUT / f"r{r:02d}-source.jsonl.gz", "rt") as f:
        traces = {(t["cohort"], t["family"], t["seed"]): t for t in map(json.loads, f)}
    seen = set()
    counts = {
        "score_rows": len(frame),
        "source_banks": 0,
        "target_banks": 0,
        "evidence_replays": 0,
        "independent_reference_fits": 0,
    }
    for b in sources(r):
        key = (b.cohort, b.family, b.seed)
        if key in seen:
            raise ValueError("Duplicate source during replay")
        seen.add(key)
        rows = keyed.get_group(key)
        trace = traces[key]
        for field, x in [
            ("source_x_hash", b.x),
            ("source_y_hash", b.y),
            ("environment_hash", b.env),
            ("context_hash", b.ctx),
            ("screen_indices_hash", b.idx),
        ]:
            code = digest(x)
            assert trace[field] == code and rows[field].eq(code).all()
        for m, saved in trace["evidence"].items():
            check_screen(b, m, saved)
            choices = rows[rows.method.eq(m)]["drop"].fillna(-1)
            assert choices.eq(-1 if saved["drop"] is None else saved["drop"]).all()
            counts["evidence_replays"] += 1
        for attr, role in (
            ("true_specialist", b.specialist),
            ("true_shortcut", b.shortcut),
        ):
            if role is None:
                assert rows[attr].isna().all()
            else:
                assert rows[attr].eq(role).all()
        counts["source_banks"] += 1
        cheap = b.family in EXAMPLES and b.seed == first[(b.cohort, b.family)]
        if cheap:
            preds = REFERENCE["controls"](b.x, b.y, [v[1] for v in b.worlds])
            counts["independent_reference_fits"] += 3
        for i, (world, test_x, y, mass, q) in enumerate(b.worlds):
            stored = rows[rows.world == world].set_index("method")
            assert stored.test_x_hash.eq(digest(test_x)).all()
            assert stored.target_hash.eq(
                digest(y if y is not None else np.column_stack((mass, q)))
            ).all()
            screen = stored.loc[list(trace["evidence"])].copy()
            screen["mask_index"] = screen["drop"].fillna(-1)
            assert (
                screen.groupby("mask_index")[["fit_hash", "test_hash", "brier"]]
                .nunique()
                .eq(1)
                .all()
                .all()
            )
            counts["target_banks"] += 1
            if cheap:
                for m in CHEAP:
                    p = preds[i][m]
                    data = SimpleNamespace(audit_y=y, mass=mass, q=q)
                    if y is None:
                        bayes = np.column_stack((1 - q, q))
                        risk = float(
                            mass @ (np.square(p - bayes).sum(1) + 2 * q * (1 - q))
                        )
                    else:
                        risk = REFERENCE["empirical"](p, y)
                    f1, _ = REFERENCE["classification_scores"](data, p)
                    np.testing.assert_allclose(
                        stored.loc[m, ["brier", "macro_f1"]].astype(float),
                        [risk, f1],
                        atol=TOL,
                        rtol=0,
                    )
    assert seen == set(traces) == set(keyed.groups)
    counts.update(check_target_geometry(frame))
    counts["independent_paired_intervals"] = check_interval_records(frame, r)
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("round", type=int, choices=range(1, 11))
    r = parser.parse_args().round
    with threadpool_limits(limits=1):
        summary = check(r)
    (OUT / f"r{r:02d}-verification.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

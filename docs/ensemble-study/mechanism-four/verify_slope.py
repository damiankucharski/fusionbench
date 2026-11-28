"""Audit independent slope-stress data, predictions and paired 1.8 vs 1.3 interaction."""

import json
import subprocess
from itertools import product
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from slope_banks import banks_slope
from threadpoolctl import threadpool_limits
from verify_four import OLD, TOL, checked_measures, hash_array, temperature_replay

from fusionbench.transfer import load_preset

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
SLOPES = (1.3, 1.8, 2.3)
SEED = 496000


def prespec():
    root = HERE.parents[2]
    rel = (HERE / "slope-protocol.md").relative_to(root)
    commit = subprocess.check_output(
        ["git", "log", "-1", "--diff-filter=A", "--format=%H", "--", str(rel)],
        cwd=root,
        text=True,
    ).strip()
    assert commit
    assert (
        subprocess.check_output(["git", "show", f"{commit}:{rel}"], cwd=root)
        == (root / rel).read_bytes()
    )
    assert not subprocess.check_output(
        [
            "git",
            "ls-tree",
            commit,
            "--",
            str((OUT / "r5-scores.csv").relative_to(root)),
        ],
        cwd=root,
    )
    return commit


def effect(raw, method, reference):
    rows = raw[raw.cohort == "factorial"]
    p = rows.pivot(
        index=["seed", "slope", "family"], columns="method", values="expected_brier"
    ).sort_index()
    values = (p[method] - p[reference]).groupby(["seed", "slope"]).mean().unstack()
    assert list(values.index) == list(range(SEED, SEED + 12)) and list(values) == list(
        SLOPES
    )
    draws = np.random.default_rng(84105).integers(12, size=(10000, 12))

    def interval(v):
        obs = v.to_numpy()
        return [
            float(x)
            for x in (obs.mean(), *np.quantile(obs[draws].mean(1), [0.025, 0.975]))
        ]

    out = {str(slope): interval(values[slope]) for slope in SLOPES}
    for slope in SLOPES[1:]:
        out[f"{slope}-1.3"] = interval(values[slope] - values[1.3])
    return out


def audit():
    start = perf_counter()
    raw = pd.read_csv(OUT / "r5-scores.csv", float_precision="round_trip")
    assert set(raw["round"]) == {5}
    raw["slope_key"] = raw.slope.fillna(-1.0)
    methods = {
        "cal_pull",
        "cal_mean",
        "full",
        "mean",
        "top_n",
        "lr",
        "pull_mean",
        "previous8",
    }
    keys = ["cohort", "family", "seed", "slope_key"]
    assert not raw.duplicated([*keys, "method"]).any()
    blocks = raw.groupby(keys)
    expected = {
        ("factorial", f"h{h}c{c}s{s}", seed, slope)
        for h, c, s in product((0, 1), repeat=3)
        for seed in range(SEED, SEED + 12)
        for slope in SLOPES
    }
    for cohort, fams, offset, n in [
        ("clustered", ("scene16", "scene128"), 100, 12),
        ("controlled", ("weak", "strong"), 200, 12),
        ("standard", load_preset("standard-v6")["scenarios"], 400, 3),
    ]:
        expected.update(
            (cohort, f, seed, -1.0)
            for f in fams
            for seed in range(SEED + offset, SEED + offset + n)
        )
    assert len(expected) == len(blocks) == 426 and expected == set(blocks.groups)
    for r in range(1, 5):
        previous = pd.read_csv(OUT / f"r{r}-scores.csv", usecols=["seed"])
        assert set(raw.seed).isdisjoint(previous.seed)
    common, refits, seen = {}, [], set()
    references, metric_err = 0, 0.0
    for b in banks_slope():
        h = b.context
        key = (h["cohort"], h["family"], h["seed"], h["slope"] or -1.0)
        rows = blocks.get_group(key).set_index("method", verify_integrity=True)
        assert set(rows.index) == methods and key not in seen
        seen.add(key)
        assert not np.shares_memory(b.x, b.audit_x)
        if b.groups is not None:
            assert set(b.groups).isdisjoint(b.audit_groups)
        for name, arr in [
            ("fit_x", b.x),
            ("audit_x", b.audit_x),
            ("fit_y", b.y),
            ("audit_y", b.audit_y),
            ("q", b.q),
        ]:
            if arr is not None:
                assert (rows[name + "_sha256"] == hash_array(arr)).all()
        if h["cohort"] == "factorial":
            values = tuple(
                hash_array(a)
                for a in (b.y, b.audit_y, b.q, b.groups, b.audit_groups, b.audit_regime)
            )
            assert values == common.setdefault(h["seed"], values)
        src, query = OLD["controls"](b.x, b.y, (b.x, b.audit_x))
        for m in src:
            assert (
                hash_array(src[m]) == rows.at[m, "fit_hash"]
                and hash_array(query[m]) == rows.at[m, "test_hash"]
            )
            observed = np.array(
                [OLD["empirical"](src[m], b.y), *checked_measures(b, query[m])]
            )
            recorded = rows.loc[
                m, ["fit_brier", "brier", "expected_brier", "macro_f1", "ap"]
            ].to_numpy(dtype=float)
            np.testing.assert_allclose(
                observed, recorded, rtol=0, atol=TOL, equal_nan=True
            )
            metric_err = max(metric_err, np.nanmax(np.abs(observed - recorded)))
            references += 1
        if (h["cohort"], h["family"], h["seed"]) == ("factorial", "h1c1s1", SEED):
            refits.append(
                {"slope": h["slope"], **temperature_replay(b, "cal_pull", rows)}
            )
    assert seen == expected and len(refits) == 3 and len(common) == 12
    learned = effect(raw, "cal_pull", "pull_mean")
    naive = effect(raw, "cal_mean", "mean")
    registered = pd.read_csv(OUT / "r5-slope-effects.csv", float_precision="round_trip")
    contrasts = pd.read_csv(OUT / "r5-contrasts.csv", float_precision="round_trip")
    for method, ref, values in (
        ("cal_pull", "pull_mean", learned),
        ("cal_mean", "mean", naive),
    ):
        for slope in SLOPES:
            select = contrasts[
                (contrasts.method == method)
                & (contrasts.reference == ref)
                & (contrasts.slope == slope)
                & (contrasts.group == "factorial")
                & (contrasts.metric == "expected_brier")
            ]
            assert len(select) == 1
            np.testing.assert_allclose(
                select[["delta", "lo", "hi"]].iloc[0],
                values[str(slope)],
                rtol=0,
                atol=TOL,
            )
        for slope in SLOPES[1:]:
            select = registered[
                (registered.method == method)
                & (registered.reference == ref)
                & (registered.slope == slope)
                & (registered.metric == "expected_brier")
            ]
            assert len(select) == 1
            np.testing.assert_allclose(
                select[["difference_in_differences", "lo", "hi"]].iloc[0],
                values[f"{slope}-1.3"],
                rtol=0,
                atol=TOL,
            )
    out = {
        "preregistration_commit": prespec(),
        "banks": len(expected),
        "method_scores": len(raw),
        "paired_slope_worlds": 288,
        "independent_seed_blocks": 12,
        "rows_per_factorial_seed": 24,
        "reference_fits": references,
        "reconstructed_forecast_hashes": 2 * references,
        "max_control_metric_error": float(metric_err),
        "temperature_replays": refits,
        "cal_pull_minus_pull_mean": learned,
        "cal_mean_minus_mean": naive,
        "weaker_at_matched_slope": learned["1.8-1.3"][1] > 0,
        "seconds": perf_counter() - start,
    }
    return out


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        result = audit()
    (OUT / "r5-verification.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )

"""Replay reported Brier scores, baseline fits and two preselected native gates.

Run with this FusionBench checkout and its paired OKAPI checkout on PYTHONPATH.
All checks use the archived source arrays directly and bypass the real-panel driver.
"""

import argparse
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
OLD = HERE.parents[1] / "external-coverage/external-results"
NEW = HERE.parents[1] / "runner-up-transfer"
SOURCES = {
    "digits_hetero_a": (OLD, OLD, "digits", 3140000),
    "digits_lr_bag": (OLD, OLD, "digits_lr_bag", 3140000),
    "wine": (OLD, OLD, "wine", 3150000),
    "digits_hetero_b": (
        NEW / "external-fit",
        NEW / "external-results",
        "digits",
        3300000,
    ),
    "iris": (NEW / "external-fit", NEW / "external-results", "iris", 3310000),
    "seeds": (NEW / "external-fit", NEW / "external-results", "seeds", 3320000),
}
NATIVE = ("incumbent", "search_z0", "search_z1")
FAMILY = ("flat", "augmented", "gp")
OUTER = ("mean", *FAMILY)
GATE_METHODS = ("guarded", "raw_plus_mean", "inner_incumbent")
BASELINES = ("mean", "top_n", "prob_lr", "logit_lr")
METHODS = {
    *BASELINES,
    *NATIVE,
    *GATE_METHODS,
    *(name + "_" + kind for name in NATIVE for kind in FAMILY),
    *("outer_" + kind for kind in OUTER),
}


def near(left, right, atol=2e-12):
    np.testing.assert_allclose(left, right, atol=atol, rtol=0)


def sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def normalize(values):
    values = np.maximum(np.array(values, dtype=np.float64), 1e-12)
    return values / values.sum(1)[:, None]


def briers(pred, labels):
    """Squared distance to one-hot via inner product identity."""
    pred = np.asarray(pred, dtype=np.float64)
    return (
        1 + np.einsum("ij,ij->i", pred, pred) - 2 * pred[np.arange(len(labels)), labels]
    )


def archive_bank(pool, replicate, meta):
    fit_dir, test_dir, suite, first = SOURCES[pool]
    seed = first + replicate
    key = f"{suite}_{seed}_"
    with np.load(fit_dir / "fit_banks.npz") as data:
        y, x, ids = (data[key + field] for field in ("labels", "inputs", "rows"))
    with np.load(test_dir / "test_banks.npz") as data:
        yt, xt, test_ids = (data[key + field] for field in ("labels", "inputs", "rows"))
    assert seed == meta["search_seed"] and len(yt) == meta["n_test"]
    assert sha(test_ids) == meta["test_rows_sha256"]
    if len(y) > 64:
        pos = np.sort(
            np.random.default_rng([seed, 96301]).choice(len(y), 64, replace=False)
        )
        x, y, ids = x[pos], y[pos], ids[pos]
    np.testing.assert_array_equal(ids, meta["fit_rows"])
    order = np.random.default_rng([seed, 159]).permutation(len(ids))
    outer, inner = np.split(order, [int(np.ceil(len(ids) * 0.3))])
    np.testing.assert_array_equal(ids[outer], meta["outer_ids"])
    np.testing.assert_array_equal(ids[inner], meta["inner_ids"])
    order = np.random.default_rng([seed, 51]).permutation(len(ids))
    for name in NATIVE:
        np.testing.assert_array_equal(
            ids[order[7 * len(ids) // 10 :]], meta[name]["selection_row_ids"]
        )
    return seed, x, y, xt, yt, inner, outer


def replay_baselines(x, y, xt, meta):
    scores = [np.mean(briers(normalize(x[:, j]), y)) for j in range(x.shape[1])]
    ranking = np.argsort(scores, kind="stable")
    np.testing.assert_array_equal(ranking, meta["topn_ranking"])
    prefixes = [
        np.mean(briers(normalize(x[:, ranking[:n]].mean(axis=1, dtype=np.float64)), y))
        for n in range(1, len(ranking) + 1)
    ]
    near(prefixes, meta["topn_prefix_brier"])
    n = 1 + int(np.argmin(prefixes))
    decisions = {
        "mean": normalize(xt.mean(axis=1)),
        "top_n": normalize(xt[:, ranking[:n]].mean(axis=1, dtype=np.float64)),
    }
    for name in ("prob_lr", "logit_lr"):

        def features(p, method=name):
            flat = p.reshape(len(p), -1)
            if method == "prob_lr":
                return flat
            value = np.clip(flat, 1e-6, 1 - 1e-6)
            return np.log(value / (1 - value))

        scale = StandardScaler().fit(features(x))
        lr = LogisticRegression(C=1, solver="lbfgs", max_iter=2000)
        lr.fit(scale.transform(features(x)), y)
        p = np.zeros((len(xt), xt.shape[-1]))
        p[:, lr.classes_] = lr.predict_proba(scale.transform(features(xt)))
        decisions[name] = normalize(p)
    return decisions, n


def two_replays(seed, x, y, xt, inner, outer, meta, bank_scores):
    """Train the unguarded inner estimator and compute the gate decision directly."""
    from loguru import logger

    from fusionbench.methods.native_pattern import RegularizedNativePatternOKAPI

    logger.disable("okapi")
    model = RegularizedNativePatternOKAPI(seed=seed, split_seed=seed).fit(
        x[inner], y[inner], task="multiclass", n_classes=x.shape[-1]
    )
    gate = model.component_predictions(x[outer])
    test = model.component_predictions(xt)
    for kind in FAMILY:
        assert sha(test[kind]) == bank_scores.loc["outer_" + kind, "prediction_sha256"]
    # Native binding uses float64 and per-node simplex normalization.
    avg = normalize(x[outer].mean(axis=1, dtype=np.float64))
    reference = briers(avg, y[outer])
    deltas = np.stack([reference - briers(gate[kind], y[outer]) for kind in FAMILY])
    for kind, losses in zip(
        OUTER, [reference, *(reference - v for v in deltas)], strict=True
    ):
        near(losses.mean(), meta["outer_gate_brier"][kind])
    u = len(outer)
    gains = deltas.mean(axis=1)
    centered = deltas - gains[:, None]
    signs = np.where(np.random.default_rng(0).random((4095, u)) < 0.5, -1, 1)
    maxima = np.maximum(0.0, (signs @ centered.T).max(axis=1)) / np.sqrt(u * (u - 1))
    cutoff = np.quantile(maxima, 0.95, method="higher")
    best = int(gains.argmax())
    raw = FAMILY[best] if gains[best] > 0 else "mean"
    guarded = FAMILY[best] if u >= 8 and gains[best] > cutoff else "mean"
    assert guarded == meta["outer_choices"]["guarded"]
    assert raw == meta["outer_choices"]["raw_plus_mean"]
    assert model.chosen_ == meta["outer_choices"]["inner_incumbent"]
    return {
        "n_outer": u,
        "raw": raw,
        "guarded": guarded,
        "max_gain": float(gains[best]),
        "threshold": float(cutoff),
    }


def check_panel(out):
    scores = pd.read_csv(out / "scores.csv.gz", keep_default_na=False)
    gates = pd.read_csv(out / "gate.csv")
    oracles = pd.read_csv(out / "final-family-oracle.csv")
    with gzip.open(out / "banks.jsonl.gz", "rt") as stream:
        metas = [json.loads(line) for line in stream]
    assert len(metas) == 48 and len(scores) == 48 * len(METHODS)
    assert len(gates) == 48 * len(GATE_METHODS) and len(oracles) == 48 * len(NATIVE)
    expected = {(pool, r) for pool in SOURCES for r in range(8)}
    assert {(m["pool"], m["replicate"]) for m in metas} == expected
    group = ["pool", "replicate"]
    scores = scores.set_index(group + ["method"], verify_integrity=True)
    gates = gates.set_index(group + ["method"], verify_integrity=True)
    oracles = oracles.set_index(group + ["search_method"], verify_integrity=True)
    differences = defaultdict(lambda: defaultdict(list))
    counts = Counter()
    gate_choices = {k: Counter() for k in GATE_METHODS}
    replays = {}
    with np.load(out / "test-row-losses.npz", allow_pickle=False) as data:
        assert len(data.files) == len(scores)
        for meta in metas:
            pool, rep = meta["pool"], meta["replicate"]
            seed, x, y, xt, yt, inner, outer = archive_bank(pool, rep, meta)
            rows = scores.xs((pool, rep), level=group)
            assert set(rows.index) == METHODS
            losses = {key: data[f"{pool}_{rep}_{key}"] for key in METHODS}
            for name, v in losses.items():
                assert v.shape == (len(yt),) and np.isfinite(v).all()
                assert v.min() >= -1e-12 and v.max() <= 2 + 1e-12
                near(v.mean(), rows.loc[name, "test_brier"])
                counts["test_row_losses"] += len(v)
            predictions, n = replay_baselines(x, y, xt, meta)
            assert n == int(rows.loc["top_n", "choice"])
            for name, p in predictions.items():
                near(briers(p, yt), losses[name])
                near(np.mean(p.argmax(axis=1) == yt), rows.loc[name, "test_accuracy"])
                counts["baseline_prediction_sha_matches"] += int(
                    sha(p) == rows.loc[name, "prediction_sha256"]
                )
            for name in NATIVE:
                chosen = rows.loc[name, "choice"]
                assert meta[name]["selected"] == chosen
                assert meta[name]["score_calls"] == float(rows.loc[name, "score_calls"])
                assert (
                    rows.loc[name, "prediction_sha256"]
                    == rows.loc[f"{name}_{chosen}", "prediction_sha256"]
                )
                np.testing.assert_array_equal(losses[name], losses[f"{name}_{chosen}"])
                costs = {k: float(losses[f"{name}_{k}"].mean()) for k in FAMILY}
                ref = oracles.loc[pool, rep, name]
                assert ref["selected"] == chosen and ref["oracle_among_three"] == min(
                    costs, key=costs.get
                )
                best = min(costs.values())
                near(
                    [
                        ref["test_brier"],
                        ref["test_three_oracle_brier"],
                        ref["selection_regret"],
                        ref["test_oracle_if_mean_allowed"],
                    ],
                    [
                        costs[chosen],
                        best,
                        costs[chosen] - best,
                        min(best, losses["mean"].mean()),
                    ],
                )
                counts[f"{name}_oracle_matches_selected_loss"] += int(
                    costs[chosen] == best
                )
            for name in NATIVE:
                counts[name + "_score_calls"] += meta[name]["score_calls"]
            # Equality includes each final candidate, the deployed choice, and score calls.
            for tail in ("", *(f"_{k}" for k in FAMILY)):
                a, b = "incumbent" + tail, "search_z0" + tail
                np.testing.assert_array_equal(losses[a], losses[b])
                assert (
                    rows.loc[a, "prediction_sha256"] == rows.loc[b, "prediction_sha256"]
                )
                counts["historical_vs_z0_identical_prediction_digests"] += 1
            assert meta["incumbent"] == meta["search_z0"]
            counts["z0_z1_different_call_count"] += int(
                meta["search_z0"]["score_calls"] != meta["search_z1"]["score_calls"]
            )
            costs = meta["outer_gate_brier"]
            raw = min(OUTER, key=costs.__getitem__)
            assert raw == meta["outer_choices"]["raw_plus_mean"]
            for name in GATE_METHODS:
                k = meta["outer_choices"][name]
                line = gates.loc[pool, rep, name]
                assert line["selected"] == k == rows.loc[name, "choice"] and line[
                    "outer_n"
                ] == len(outer)
                assert (
                    rows.loc[name, "prediction_sha256"]
                    == rows.loc["outer_" + k, "prediction_sha256"]
                )
                np.testing.assert_array_equal(losses[name], losses["outer_" + k])
                d_gate, d_test = (
                    costs[k] - costs["mean"],
                    losses[name].mean() - losses["outer_mean"].mean(),
                )
                near(
                    [
                        line["gate_brier"],
                        line["test_brier"],
                        line["gate_minus_mean"],
                        line["test_minus_mean"],
                        line["relative_optimism"],
                    ],
                    [costs[k], losses[name].mean(), d_gate, d_test, d_test - d_gate],
                )
                gate_choices[name][k] += 1
            assert meta["outer_choices"]["guarded"] in ("mean", raw)
            for kind, a, b in (
                ("primary", "search_z1", "search_z0"),
                ("secondary", "guarded", "raw_plus_mean"),
            ):
                delta = float(np.mean(losses[a] - losses[b]))
                differences[kind][pool].append(delta)
                counts[kind + "_sha_matches"] += int(
                    rows.loc[a, "prediction_sha256"] == rows.loc[b, "prediction_sha256"]
                )
                counts[kind + "_identical_loss_vectors"] += int(
                    np.array_equal(losses[a], losses[b])
                )
            if rep == 0 and pool in ("digits_hetero_a", "iris"):
                replays[pool] = two_replays(seed, x, y, xt, inner, outer, meta, rows)
    assert counts["baseline_prediction_sha_matches"] == 4 * len(metas)
    contrasts = {}
    published = pd.read_csv(out / "pool-differences.csv", index_col="pool")
    summary = json.loads((out / "summary.json").read_text())
    reported_directions = pd.read_csv(out / "paired-direction.csv").set_index(
        ["pool", "contrast"]
    )
    assert summary["z0_z1_prediction_hash_matches"] == counts["primary_sha_matches"]
    assert (
        summary["raw_guarded_prediction_hash_matches"]
        == counts["secondary_sha_matches"]
    )
    assert summary["incumbent_z0_prediction_hash_matches"] == len(metas)
    for method in ("guarded", "raw_plus_mean"):
        assert (
            summary["secondary_terminals_select_mean"][method]
            == gate_choices[method]["mean"]
        )
    for kind, by_pool in differences.items():
        column = (
            "search_z1_minus_search_z0"
            if kind == "primary"
            else "guarded_minus_raw_plus_mean"
        )
        primary_count = Counter()
        for pool, values in by_pool.items():
            direction = {
                "better": sum(v < -1e-12 for v in values),
                "tied": sum(abs(v) <= 1e-12 for v in values),
                "worse": sum(v > 1e-12 for v in values),
            }
            for metric, value in direction.items():
                assert reported_directions.loc[(pool, column), metric] == value
            primary_count.update(direction)
            near(
                reported_directions.loc[
                    (pool, column), ["min_difference", "max_difference"]
                ].to_numpy(float),
                [min(values), max(values)],
            )
        for metric, value in primary_count.items():
            assert value == summary[kind + "_wins_ties_losses"][metric]
        for pool, values in by_pool.items():
            near(published.loc[pool, column], mean(values))
        by_pool = {pool: mean(values) for pool, values in by_pool.items()}
        digits = mean(v for k, v in by_pool.items() if k.startswith("digits"))
        contrasts[kind] = {
            "wins_ties_losses": dict(primary_count),
            "pooled_within_each_provider_pool": by_pool,
            "six_pools_equal": mean(by_pool.values()),
            "four_datasets_equal": mean(
                [digits, by_pool["iris"], by_pool["wine"], by_pool["seeds"]]
            ),
        }
        near(
            [
                contrasts[kind]["six_pools_equal"],
                contrasts[kind]["four_datasets_equal"],
            ],
            [
                published.loc["six_pool_macro", column],
                published.loc["four_dataset_macro", column],
            ],
        )
    return {
        "banks": len(metas),
        "score_rows": len(scores),
        "gate_rows": len(gates),
        "final_family_oracle_rows": len(oracles),
        "checks": dict(counts),
        "selections": gate_choices,
        "contrasts": contrasts,
        "independent_inner_refit_and_multiplier_replays": replays,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=HERE / "results")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        print(json.dumps(check_panel(args.results), indent=2, sort_keys=True))

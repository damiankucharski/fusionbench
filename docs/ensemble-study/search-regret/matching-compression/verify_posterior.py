"""Verify frozen structured ensembles against exact integer marginal likelihoods."""

import argparse
import hashlib
import json
import runpy
from functools import lru_cache
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

from fusionbench.eval.matching_posterior import MatchingFamily

HERE = Path(__file__).resolve().parent
STOCK = runpy.run_path(str(HERE / "verify_stock.py"))
BASE = runpy.run_path(str(HERE.parent / "matched-graft" / "verify.py"))
MODES = ("prior", "conditional", "x_only", "joint")
SPLITS = (
    "fit",
    "validation",
    "two_balanced",
    "two_uniform",
    "three_uniform",
    "four_uniform",
)
EXPERIMENT = ["data_seed", "k", "train_design", "labels", "fit_law"]
LO, SPAN = 1 / 64, 62 / 64


def matchings(vertices):
    # Pair the LAST unmatched vertex, then canonicalize the resulting unordered
    # edge sets. This enumeration is independent of the fitted implementation.
    if not vertices:
        yield ()
        return
    *others, last = vertices
    for j, partner in enumerate(others):
        for rest in matchings(others[:j] + others[j + 1 :]):
            yield tuple(sorted((*rest, (partner, last))))


@lru_cache
def supports(k):
    n = 2 * k
    models = sorted(matchings(list(range(n))))
    grid = {}
    for size in (2, 3, 4):
        sets = list(combinations(range(n), size))
        pair_idx = {edge: j for j, edge in enumerate(combinations(range(n), 2))}
        if size == 2:
            incidence = np.asarray(
                [[edge in m for m in models] for edge in sets], dtype=np.int8
            )
        # Number of true matching edges contained in each query. Its positive
        # indicator is the literal OR in the corresponding hypothesis.
        count = np.stack(
            [incidence[[pair_idx[e] for e in combinations(s, 2)]].sum(0) for s in sets]
        )
        grid[size] = (
            np.asarray(count, dtype=float),
            np.asarray(count > 0, dtype=float),
        )
    return models, pair_idx, grid


def exact_weights(k, design, mode, c, hits):
    # p(y|M) is a product of 63/64 for matches and 1/64 for misses.
    # In a balanced design, every matched input edge is (2k-2) times
    # as probable as any unmatched one; in the uniform design this ratio is 1.
    label = mode in ("conditional", "joint")
    geometry = mode in ("x_only", "joint") and design == "balanced"
    pow_y = [63**i for i in range(int(c.max()) + 1)] if label else [1]
    pow_x = [(2 * k - 2) ** i for i in range(int(hits.max()) + 1)] if geometry else [1]
    y_idx = c if label else np.zeros_like(c)
    x_idx = hits if geometry else np.zeros_like(hits)
    numerators = [pow_y[a] * pow_x[b] for a, b in zip(y_idx, x_idx, strict=True)]
    normal = sum(numerators)
    return np.array([v / normal for v in numerators])


def pair_ids(x, lookup):
    return np.array([lookup[tuple(np.flatnonzero(r > 0.5))] for r in x], dtype=int)


def check_tables(frame, path):
    dims = ["k", "train_design", "labels", "fit_law", "split", "decision"]
    metrics = ["conditional_excess", "observed_brier"]
    groups = frame.groupby([*dims, "method", "data_seed"])[metrics].mean()
    means = groups.groupby([*dims, "method"]).mean()
    saved = pd.read_csv(path / "means.csv", float_precision="round_trip").set_index(
        [*dims, "method"]
    )
    assert (
        means.index.is_unique
        and saved.index.is_unique
        and set(means.index) == set(saved.index)
    )
    np.testing.assert_allclose(
        means, saved.loc[means.index, metrics], atol=2e-15, rtol=0
    )
    paired = pd.read_csv(path / "paired.csv", float_precision="round_trip").set_index(
        [*dims, "method", "comparator", "metric"]
    )
    pairs = [
        ("joint", "conditional"),
        ("x_only", "prior"),
        ("joint", "matching_map"),
        ("joint", "stock_gp"),
        ("joint", "stack_logit"),
        ("joint_pair_network", "joint"),
    ]
    checked = set()
    for cell, rows in groups.reset_index().query("decision=='fit'").groupby(dims):
        for metric in metrics:
            wide = rows.pivot(index="data_seed", columns="method", values=metric)
            for a, b in pairs:
                deltas = wide[a] - wide[b]
                key = (*cell, a, b, metric)
                checked.add(key)
                row = paired.loc[key]
                assert len(deltas) == row.banks == 16
                mid = deltas.mean()
                half = t.ppf(0.975, 15) * deltas.std() / 4
                np.testing.assert_allclose(
                    [row.difference, row.low_95, row.high_95],
                    [mid, mid - half, mid + half],
                    rtol=0,
                    atol=2e-15,
                )
    assert checked == set(paired.index) and len(checked) == 1152
    return len(means), len(checked)


def verify(path):
    frame = pd.read_csv(path / "runs.csv.gz", float_precision="round_trip")
    measured = frame[frame.method != "stock_gp"]
    comp = pd.read_csv(
        path / "compilation.csv.gz", float_precision="round_trip"
    ).set_index([*EXPERIMENT, "mode", "order", "support"])
    assert not comp.index.has_duplicates
    assert len(frame) == 23040 and len(measured) == 16896 and len(comp) == 10240
    assert not frame.duplicated(
        [*EXPERIMENT, "method", "decision", "gp_seed", "split"]
    ).any()
    checks = rows = map_ties = 0
    largest = {
        "float_replay_loss": 0.0,
        "independent_loss": 0.0,
        "integer_posterior_weight": 0.0,
        "independent_program_probability": 0.0,
        "compilation_statistic": 0.0,
    }
    canonical_families = {k: MatchingFamily(k) for k in (4, 6)}
    for experiment, group in measured.groupby(EXPERIMENT):
        seed, k, design, n, law = experiment
        k, n = int(k), int(n)
        bank = STOCK["PairBank"](seed, k)
        splits = STOCK["build_splits"](bank, design, n, law)
        y = splits["fit"].y
        unl, _ = bank.draw(128, 30, design, law)
        models, lookup, grid = supports(k)
        truth = grid[2][1]
        fit_id = pair_ids(splits["fit"].x, lookup)
        val_id = pair_ids(splits["validation"].x, lookup)
        unl_id = pair_ids(unl, lookup)
        f = canonical_families[k]
        assert np.array_equal(f.matchings[0], [lookup[e] for e in models[0]])
        correct = (truth[fit_id].astype(bool) == y[:, None]).sum(0)
        within = (truth[fit_id].sum(0) + truth[unl_id].sum(0)).astype(int)
        best = np.flatnonzero(correct == correct.max())
        records = group.set_index(["method", "split"])
        er = records.loc["matching_map"].iloc[0]
        assert (
            er.ties == len(best)
            and tuple(map(tuple, json.loads(er.model))) == models[best[0]]
        )
        hidden = models.index(tuple(bank.matching))
        map_ties += 1
        methods = {}
        matrices = [splits[s].x for s in SPLITS]
        if len(np.unique(y)) > 1:
            base, (selected, ranking, prefix) = BASE["make_baselines"](
                matrices[0], y, matrices
            )
        else:
            # The logistic baseline uses a constant-label fallback only on
            # this one-class sample. The ordinary Top-N remains fit from y.
            ranking = np.argsort(
                ((matrices[0] - y[:, None]) ** 2).mean(0), kind="stable"
            )
            prefix = [
                2 * np.mean((matrices[0][:, ranking[:j]].mean(1) - y) ** 2)
                for j in range(1, 2 * k + 1)
            ]
            selected = int(np.argmin(prefix)) + 1
            base = {
                "average": [x.mean(1) for x in matrices],
                "top_n": [x[:, ranking[:selected]].mean(1) for x in matrices],
                "stack_logit": [np.full(len(x), y[0], dtype=float) for x in matrices],
            }
        for method, p in base.items():
            methods[method] = dict(zip(SPLITS, p, strict=True))
        row = records.loc["top_n"].iloc[0]
        assert row.n_selected == selected and np.array_equal(
            json.loads(row.ranking), ranking
        )
        np.testing.assert_allclose(
            json.loads(row.prefix_scores), prefix, atol=4e-15, rtol=0
        )
        methods["known_law"] = {s: b.q for s, b in splits.items()}
        groups = {
            "fit": (truth[fit_id], truth[fit_id]),
            "validation": (truth[val_id], truth[val_id]),
            "2": grid[2],
            "3": grid[3],
            "4": grid[4],
        }
        src = {
            "fit": "fit",
            "validation": "validation",
            "two_balanced": "2",
            "two_uniform": "2",
            "three_uniform": "3",
            "four_uniform": "4",
        }
        methods["matching_map"] = {
            s: LO + SPAN * groups[tag][1][:, best[0]] for s, tag in src.items()
        }
        z = {tag: splits[name].x > 0.5 for name, tag in src.items()}
        exact_predictions = {}
        pi = 0.5 if design == "balanced" else 1 / (2 * k - 1)
        for mode in MODES:
            w = exact_weights(k, design, mode, correct, within)
            if mode == "prior":
                canonical = f.fit(z["fit"][:0], y[:0], pi=pi, mode="conditional")
            else:
                canonical = f.fit(z["fit"], y, unl > 0.5, pi=pi, mode=mode)
            weight_err = float(np.max(abs(w - canonical.weights)))
            largest["integer_posterior_weight"] = max(
                largest["integer_posterior_weight"], weight_err
            )
            assert weight_err < 1e-12
            diag = records.loc[mode].iloc[0]
            assert abs(diag.posterior_true - w[hidden]) < 1e-12
            assert np.isclose(
                diag.effective_models, 1 / (w @ w), rtol=1e-12, atol=1e-12
            )
            true_edges = [lookup[edge] for edge in bank.matching]
            assert (
                abs(diag.expected_correct_edges - (truth[true_edges] @ w).sum()) < 1e-12
            )
            methods[mode] = {}
            if mode == "joint":
                methods["joint_pair_network"] = {}
                methods["joint_pair_quad_network"] = {}
            for tag, (count, exists) in groups.items():
                direct = LO + SPAN * (exists @ w)
                if tag in ("fit", "validation"):
                    replay = LO + SPAN * (exists.astype(bool) @ canonical.weights)
                else:
                    replay = LO + SPAN * (exists @ canonical.weights)
                tol = float(np.max(abs(replay - direct)))
                largest["independent_program_probability"] = max(
                    largest["independent_program_probability"], tol
                )
                assert tol < 1e-12
                exact_predictions[mode, tag] = direct
                for s, name in src.items():
                    if name == tag:
                        methods[mode][s] = replay
                for order in (1, 2):
                    # Independent expectation of number of matching pairs for
                    # order one; order two is exactly the OR through four active.
                    target = LO + SPAN * (count @ w) if order == 1 else direct
                    network = canonical.predict(z[tag], order=order)
                    d = float(np.max(abs(target - network)))
                    largest["independent_program_probability"] = max(
                        largest["independent_program_probability"], d
                    )
                    assert d < 1e-12
                    stored = comp.loc[(*experiment, mode, order, tag)]
                    assert stored.states == len(network)
                    raw = float(np.max(abs(replay - network)))
                    clipped = float(np.max(abs(replay - np.clip(network, 0, 1))))
                    assert (
                        abs(raw - stored.max_abs) < 1e-12
                        and abs(clipped - stored.clipped_max_abs) < 1e-12
                    )
                    largest["compilation_statistic"] = max(
                        largest["compilation_statistic"],
                        abs(raw - stored.max_abs),
                        abs(clipped - stored.clipped_max_abs),
                    )
                    assert order == 1 and tag == "4" or raw < 1e-12
                    checks += 1
                    if mode == "joint":
                        name = (
                            "joint_pair_network"
                            if order == 1
                            else "joint_pair_quad_network"
                        )
                        exact_predictions[name, tag] = target
                        for s, tgt in src.items():
                            if tgt == tag:
                                methods[name][s] = network
        if design == "uniform":
            for tag in groups:
                assert np.array_equal(
                    exact_predictions["conditional", tag],
                    exact_predictions["joint", tag],
                )
                assert np.array_equal(
                    exact_predictions["prior", tag], exact_predictions["x_only", tag]
                )
        for row in group.itertuples():
            v = methods[row.method][row.split]
            assert (
                hashlib.sha256(np.asarray(v, dtype="<f8").tobytes()).hexdigest()
                == row.prediction_sha256
            ), (experiment, row.method, row.split, "SHA")
            loss = splits[row.split].losses(v)
            err = max(
                abs(loss["observed_brier"] - row.observed_brier),
                abs(loss["conditional_excess"] - row.conditional_excess),
            )
            largest["float_replay_loss"] = max(err, largest["float_replay_loss"])
            assert err < 1e-13
            independent = exact_predictions.get((row.method, src[row.split]), v)
            independent_losses = splits[row.split].losses(independent)
            numeric_err = max(
                abs(independent_losses[key] - getattr(row, key))
                for key in independent_losses
            )
            largest["independent_loss"] = max(numeric_err, largest["independent_loss"])
            assert numeric_err < 1e-12
            rows += 1
    assert checks == len(comp) and rows == len(measured) and map_ties == 256
    means, contrasts = check_tables(frame, path)
    result = {
        "input_sha256": {
            filename: hashlib.sha256((path / filename).read_bytes()).hexdigest()
            for filename in (
                "runs.csv.gz",
                "compilation.csv.gz",
                "means.csv",
                "paired.csv",
            )
        },
        "structured_and_baseline_prediction_digests": rows,
        "structured_and_baseline_losses": 2 * rows,
        "exact_matching_erm_and_tie_counts": map_ties,
        "independent_compilation_checks": checks,
        "separate_gp_split_rows": len(frame) - rows,
        "bank_means": means,
        "paired_contrasts": contrasts,
        "maximum_absolute_differences": largest,
    }
    (path / "posterior-verified.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("path", type=Path)
    verify(cli.parse_args().path)

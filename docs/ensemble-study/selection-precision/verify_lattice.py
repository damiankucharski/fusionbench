"""Independent graded-lattice audit with a bare native replay on two fixed banks."""

import argparse
import hashlib
import json
import struct
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import t
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

LEVELS = np.array([0.1, 0.35, 0.65, 0.9])
DIGITS = np.array(list(product(range(4), repeat=4)))
X = LEVELS[DIGITS]
FEATURES = np.stack([1 - X, X], axis=-1)
UNIFORM = np.ones(256) / 256
COORD_MASS = np.array([[0.05, 0.1, 0.25, 0.6], [0.6, 0.25, 0.1, 0.05]] * 2)
SHIFT = np.prod([COORD_MASS[j, DIGITS[:, j]] for j in range(4)], axis=0)


def signatures():
    # Generate every Boolean function then reject violations on cube edges.
    truth = (np.arange(65536, dtype=np.int64)[:, None] >> np.arange(16)) & 1
    valid = (truth[:, 0] == 0) & (truth[:, -1] == 1)
    for mask in range(16):
        for bit in range(4):
            valid &= truth[:, mask] <= truth[:, mask | (1 << bit)]
    tables = truth[valid]

    def order(row):
        terms = []
        for mask in np.flatnonzero(row):
            active = tuple(j for j in range(4) if mask & (1 << j))
            if all(not row[mask ^ (1 << j)] for j in active):
                terms.append(active)
        return sum(map(len, terms)), sorted(terms)

    tables = np.asarray(sorted(tables, key=order))
    # A thresholded lattice function is the same Boolean function on thresholded inputs.
    level_counts = np.zeros((len(tables), len(X)), dtype=int)
    for threshold in LEVELS[1:]:
        masks = ((X >= threshold) * (1 << np.arange(4))).sum(axis=1)
        level_counts += tables[:, masks]
    return LEVELS[level_counts]


def data(law, seed):
    role = np.random.default_rng([seed, 903]).permutation(4)
    a, b, c, d = X[:, role].T
    q = (
        np.maximum(np.minimum(a, b), np.minimum(c, d))
        if law == "matched"
        else expit(-0.2 + logit(X[:, role]) @ np.array([0.45, 0.65, 0.35, 0.55]))
    )
    sets = []
    for i, (size, weights) in enumerate(
        [(256, UNIFORM), (256, UNIFORM), (2048, UNIFORM), (2048, SHIFT)]
    ):
        rng = np.random.default_rng([seed, i, 491])
        cells = rng.choice(256, size=size, p=weights)
        sets.append((cells, rng.random(size) < q[cells]))
    return q, sets


def native_values(program):
    result = X[:, int(program["id"][1:-4])]
    for op in program["children"]:
        args = [result] + [native_values(child) for child in op["children"]]
        assert op["type"] in ("MaxNode", "MinNode")
        result = (np.max if op["type"] == "MaxNode" else np.min)(args, axis=0)
    return result


def normalize(p):
    p = np.maximum(p, 1e-12)
    return p / p.sum(axis=1, keepdims=True)


def fit_baselines(cells, y):
    onehot = np.eye(2)[y.astype(int)]

    def loss(p):
        return np.mean(np.sum((normalize(p) - onehot) ** 2, axis=1))

    ranking = np.argsort([loss(FEATURES[cells, j]) for j in range(4)], kind="stable")
    acc = np.zeros((len(cells), 2))
    scores = []
    for k, j in enumerate(ranking, 1):
        acc += FEATURES[cells, j]
        scores.append(loss(acc / k))
    count = 1 + np.argmin(scores)
    totals = np.zeros((len(X), 2))
    for j in ranking[:count]:
        totals += FEATURES[:, j]
    flat = FEATURES.reshape(len(X), -1)
    logs = np.log(flat / (1 - flat))
    model = make_pipeline(
        StandardScaler(), LogisticRegression(solver="lbfgs", C=1.0, max_iter=2000)
    )
    model.fit(logs[cells], y)
    return (
        {
            "stack_logit": normalize(model.predict_proba(logs))[:, 1],
            "mean": np.mean(X, axis=1),
            "top_n": normalize(totals / count)[:, 1],
        },
        ranking,
        scores,
    )


def bare_replay(fit_cells, fit_y, all_rules, fit_cost, validation_cost, rows):
    from okapi import Okapi
    from okapi.operators import MAX, MIN

    idx = {v.tobytes(): i for i, v in enumerate(all_rules)}
    checkpoints = 0
    with TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        paths = [tmp / f"p{i}.npy" for i in range(4)]
        for i, path in enumerate(paths):
            np.save(path, X[:, i : i + 1])
        count = np.bincount(fit_cells, minlength=256)
        pos = np.bincount(fit_cells, weights=fit_y.astype(int), minlength=256)
        gt = tmp / "labels.npy"
        np.save(gt, (pos / np.maximum(count, 1))[:, None])

        def replay_arm(seed, unit, precision, block):
            nonlocal checkpoints
            raw, offered, actual = (hashlib.sha256() for _ in range(3))
            prev = None
            offset = calls = 0
            seen = []
            traces = {}

            def exact_observer(p, _):
                nonlocal calls, offset, prev
                # Public, unmodified native stock evaluates this entire population in order.
                if prev is not native.population:
                    prev = native.population
                    offset = 0
                tree = prev[offset]
                offset += 1
                code = idx[p.ravel().tobytes()]
                if code not in seen:
                    seen.append(code)
                raw.update((repr(tree.structural_key()) + "\n").encode())
                raw.update(struct.pack("<HI", code, int(fit_cost[code])))
                calculated = int(
                    np.sum(
                        (np.rint(20 * p[fit_cells, 0]).astype(int) - 20 * fit_y) ** 2
                    )
                )
                assert calculated == fit_cost[code]
                f = 1.0 - calculated / (400 * len(fit_y))
                if unit == "compressed":
                    f = 1.0 - (1.0 - f) / 4096.0
                g = f if precision == "full" else np.round(f, int(precision[-1]))
                offered.update(struct.pack("<d", f))
                actual.update(struct.pack("<d", g))
                calls += 1
                if calls in (100, 400):
                    traces[calls] = (
                        raw.hexdigest(),
                        offered.hexdigest(),
                        actual.hexdigest(),
                        tuple(seen),
                    )
                return f

            native = Okapi(
                paths,
                gt,
                4,
                2,
                2,
                backend="numpy",
                seed=int(seed),
                objective_functions=(exact_observer,),
                allowed_ops=(MIN, MAX),
                postprocessing_function=np.asarray,
                selection_fitness_decimals=(
                    None if precision == "full" else int(precision[-1])
                ),
            )
            while calls < 400:
                native.run_iteration()
            for row in block.itertuples():
                raw_sha, offer_sha, grid_sha, ids = traces[row.budget]
                assert (raw_sha, offer_sha, grid_sha) == (
                    row.raw_prefix_sha256,
                    row.offered_prefix_sha256,
                    row.selection_prefix_sha256,
                )
                assert len(ids) == row.unique_rules
                key = (
                    (lambda k: fit_cost[k])
                    if row.selector == "fit"
                    else (lambda k: (validation_cost[k], fit_cost[k]))
                )
                assert min(ids, key=key) == row.rule
                checkpoints += int(row.selector == "fit")

        for (seed, unit, precision), block in rows.groupby(
            ["gp_seed", "units", "precision"]
        ):
            replay_arm(seed, unit, precision, block)
    return checkpoints


def audit(folder):
    from loguru import logger

    logger.disable("okapi")
    raw = pd.read_csv(folder / "lattice.csv.gz", float_precision="round_trip")
    tables = signatures()
    assert tables.shape == (166, 256)
    checksum = hashlib.sha256(tables.tobytes()).hexdigest()
    assert (
        checksum == "681603218b3e47141bc7193f36f899097d5b5bd0485134a1ec8959b7592d7cb2"
    )
    assert len(raw) == 1536
    max_error = 0.0
    scalar_checks = bare_checkpoints = 0
    for (law, seed, n), group in raw.groupby(["law", "data_seed", "n_fit"]):
        eta, sets = data(law, seed)
        fit_x, fit_y = sets[0][0][:n], sets[0][1][:n]
        exact_cost = np.sum(
            (np.rint(tables[:, fit_x] * 20).astype(int) - fit_y * 20) ** 2, axis=1
        )
        min_cost, best = int(exact_cost.min()), int(exact_cost.argmin())
        vx, vy = sets[1]
        validation_cost = np.sum(
            (np.rint(tables[:, vx] * 20).astype(int) - vy * 20) ** 2, axis=1
        )
        iid_b, sh_b = (float(w @ (eta * (1 - eta))) for w in [UNIFORM, SHIFT])
        approx = (tables - eta) ** 2
        iid_min, sh_min = (float(np.min(approx @ w)) for w in [UNIFORM, SHIFT])
        base, ranking, prefix_scores = fit_baselines(fit_x, fit_y)
        for row in group.itertuples():
            if row.method == "gp":
                pr = native_values(json.loads(row.program))
                np.testing.assert_array_equal(pr, tables[int(row.rule)])
            elif row.method == "exact_erm":
                pr = tables[best]
                assert row.rule == best
            else:
                pr = base[row.method]
            assert (
                hashlib.sha256(np.asarray(pr, dtype="<f8").tobytes()).hexdigest()
                == row.prediction_sha256
            )
            if row.method == "top_n":
                assert (
                    np.array_equal(json.loads(row.ranking), ranking)
                    and row.n_selected == int(np.argmin(prefix_scores)) + 1
                )
                np.testing.assert_allclose(
                    json.loads(row.prefix_scores), prefix_scores, rtol=0, atol=1e-14
                )
            assert (
                json.loads(row.family_erm_rule_ids)
                == np.flatnonzero(exact_cost == min_cost).tolist()
            )
            for attribute, value in (
                ("family_fit_units", min_cost),
                ("family_erm_ties", (exact_cost == min_cost).sum()),
            ):
                assert getattr(row, attribute) == value
            if pd.notna(row.rule):
                assert row.fit_units == exact_cost[int(row.rule)]
                assert row.fit_hit == int(exact_cost[int(row.rule)] == min_cost)
            dev = pr - eta
            emp = np.mean((pr[fit_x] - fit_y) ** 2)
            expected = {
                "fit_brier": emp,
                "fit_minus_family_erm": emp - min_cost / (400 * n),
                "iid_family_risk": iid_b + iid_min,
                "shift_family_risk": sh_b + sh_min,
                "family_iid_regret_at_erm": (approx[best] @ UNIFORM) - iid_min,
                "iid_bayes_risk": iid_b,
                "shift_bayes_risk": sh_b,
                "iid_excess_bayes": UNIFORM @ dev**2,
                "shift_excess_bayes": SHIFT @ dev**2,
                "fit_input_excess": np.mean(dev[fit_x] ** 2),
                "fit_label_cross_term": -2 * np.mean(dev[fit_x] * (fit_y - eta[fit_x])),
                "fit_generator_noise": np.mean((fit_y - eta[fit_x]) ** 2),
            }
            for attr, (xc, yc) in zip(
                ("validation_brier", "iid_test_brier", "shift_test_brier"), sets[1:]
            ):
                expected[attr] = np.mean((pr[xc] - yc) ** 2)
            expected["iid_risk"] = expected["iid_excess_bayes"] + iid_b
            expected["shift_risk"] = expected["shift_excess_bayes"] + sh_b
            expected["iid_minus_family"] = (
                expected["iid_risk"] - expected["iid_family_risk"]
            )
            expected["shift_minus_family"] = (
                expected["shift_risk"] - expected["shift_family_risk"]
            )
            expected["iid_minus_fit_input_excess"] = (
                expected["iid_excess_bayes"] - expected["fit_input_excess"]
            )
            for attr, value in expected.items():
                delta = abs(getattr(row, attr) - value)
                assert delta < 2e-13, (law, seed, n, row.method, attr, delta)
                max_error = max(max_error, delta)
                scalar_checks += 1
        if seed in (6200, 6201):
            bare_checkpoints += bare_replay(
                fit_x,
                fit_y,
                tables,
                exact_cost,
                validation_cost,
                group[group.method.eq("gp")],
            )
    paired = pd.read_csv(folder / "lattice-paired.csv", float_precision="round_trip")
    gp = raw[raw.method.eq("gp")].copy()
    gp["arm"] = gp.units + "_" + gp.precision
    checks = 0
    for (law, n, sel, budget), rows in paired.groupby(
        ["law", "n_fit", "selector", "budget"]
    ):
        b = gp[
            gp.law.eq(law) & gp.n_fit.eq(n) & gp.selector.eq(sel) & gp.budget.eq(budget)
        ]
        avg = b.groupby(["data_seed", "arm"]).mean(numeric_only=True)
        for row in rows.itertuples():
            difference = np.array(
                [
                    avg.loc[(s, row.left), row.metric]
                    - avg.loc[(s, row.right), row.metric]
                    for s in range(6200, 6212)
                ]
            )
            mean = np.sum(difference) / 12
            half = np.sqrt(np.sum((difference - mean) ** 2) / (11 * 12)) * t.ppf(
                0.975, 11
            )
            np.testing.assert_allclose(
                [mean, mean - half, mean + half],
                [row.mean_difference, row.ci_low, row.ci_high],
                rtol=0,
                atol=1e-12,
            )
            checks += 1
    return {
        "source_sha256": hashlib.sha256(
            (folder / "lattice.csv.gz").read_bytes()
        ).hexdigest(),
        "ordered_family_sha256": checksum,
        "verified_prediction_vectors": len(raw),
        "verified_scalar_metrics": scalar_checks,
        "max_metric_error": max_error,
        "verified_paired_intervals": checks,
        "bare_native_data_seeds": [6200, 6201],
        "bare_native_trajectories": bare_checkpoints // 2,
        "bare_native_checkpoints_with_three_matching_digests": bare_checkpoints,
        "bare_native_archive_winners_verified": 2 * bare_checkpoints,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        answer = audit(args.folder)
    (args.folder / "lattice-verification.json").write_text(
        json.dumps(answer, indent=2) + "\n"
    )

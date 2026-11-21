"""Independently reconstruct frozen forecasts, transported rankings and finite-pool audits."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from functools import cache
from itertools import accumulate, combinations
from pathlib import Path
from runpy import run_path
from statistics import NormalDist
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

P = (0.10, 0.50)
L = np.array([0, 0.25, 0.5, 0.75, 1])
BUDGET = (8, 32, 128)
MODELS = [f"gp_{i}" for i in range(941, 953)] + ["mean", "top_n", "stack_logit"]
PAIRS = list(combinations(range(7), 2))


def observation(bank, split, n):
    rng = np.random.default_rng([bank, split])
    y = rng.integers(2, size=n)
    common = rng.normal(size=n)
    scores = []
    for j, acc in enumerate((0.95, 0.85, 0.82, 0.79, 0.77, 0.73, 0.68)):
        s = NormalDist().inv_cdf(acc)
        noise = rng.standard_normal(n)
        if j:
            noise = math.sqrt(1.0 - 0.65) * noise + math.sqrt(0.65) * common
        prob = 1.0 / (1.0 + np.exp(-(2.0 * s * ((2.0 * y - 1.0) * s + noise))))
        scores.append(prob.clip(1e-6, 1.0 - 1e-6))
    positive = np.array(scores).T
    clean = np.stack((1 - positive, positive), axis=-1)
    other = clean.copy()
    other[:, 0, :] = clean[:, 0, ::-1]
    return clean, other, y


def mixed(a, b, mask):
    x = a.copy()
    x[mask] = b[mask]
    return x


def sha(x, dtype="<f8"):
    return hashlib.sha256(np.asarray(x, dtype=dtype).tobytes()).hexdigest()


def decoded(tree, x):
    out = x[:, int(tree["id"][1:-4]), 1]
    for op in tree["children"]:
        values = [out, *[decoded(v, x) for v in op["children"]]]
        t = op["type"]
        if t == "MeanNode":
            out = sum(values) / len(values)
        elif t == "WeightedMeanNode":
            out = sum(w * v for w, v in zip(op["weights"], values, strict=True))
        elif t in ("MaxNode", "MinNode"):
            out = (np.maximum if t == "MaxNode" else np.minimum).reduce(values)
        else:
            assert t in ("FarThresholdNode", "CloseThresholdNode")
            matrix = np.stack(values)
            ds = np.square(matrix - op["threshold"])
            idx = (np.argmin if t == "CloseThresholdNode" else np.argmax)(ds, axis=0)
            out = matrix[idx, np.arange(len(x))]
    return out


def norm(x):
    x = np.clip(x, 1e-12, None)
    return x / x.sum(axis=1, keepdims=True)


def refs(train, y):
    one = np.eye(2)[y]

    def loss(x):
        return np.mean(np.sum((norm(x) - one) ** 2, axis=1))

    rank = np.argsort([loss(train[:, j]) for j in range(7)], kind="stable")
    full = np.zeros_like(one)
    prefix = []
    for k, j in enumerate(rank, 1):
        full += train[:, j]
        prefix.append(loss(full / k))
    size = 1 + int(np.argmin(prefix))

    def top(x):
        a = np.zeros((len(x), 2))
        for j in rank[:size]:
            a += x[:, j]
        return norm(a / size)[:, 1]

    def feat(x):
        x = np.clip(x.reshape(len(x), -1), 1e-6, 1.0 - 1e-6)
        return np.log(x / (1 - x))

    lr = make_pipeline(
        StandardScaler(), LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000)
    )
    lr.fit(feat(train), y)
    return (
        rank,
        prefix,
        size,
        [
            lambda x: x[:, :, 1].mean(axis=1),
            top,
            lambda x: norm(lr.predict_proba(feat(x)))[:, 1],
        ],
    )


def bare_gp(number, x, y, row):
    from loguru import logger
    from okapi import Okapi
    from okapi.globals import use_postprocessing_function
    from okapi.node import MaxNode, MeanNode, MinNode, ValueNode
    from okapi.tree import Tree

    nfit = 0

    def fitness(p, _):
        nonlocal nfit
        nfit += 1
        return 1.0 - np.mean((p[:, 0] - y) ** 2)

    logger.disable("okapi")
    with TemporaryDirectory() as folder, use_postprocessing_function(np.asarray):
        root = Path(folder)
        paths = [root / f"m{i}.npy" for i in range(7)]
        for i, path in enumerate(paths):
            np.save(path, x[:, i, 1:2])
        gt = root / "labels.npy"
        np.save(gt, y)
        native = Okapi(
            paths,
            gt,
            7,
            2,
            3,
            backend="numpy",
            seed=number,
            objective_functions=(fitness,),
            minimize_node_count=True,
        )
        rng = np.random.default_rng(number)
        for _ in range(8):
            ix = rng.choice(7, int(rng.integers(2, 5)), replace=False)
            members = [ValueNode(None, native.models[j], native.ids[j]) for j in ix]
            op = (MinNode, MaxNode, MeanNode)[int(rng.integers(3))]
            members[0].add_child(op(members[1:]))
            native.population.append(Tree(members[0]))
        native.population_size = 15
        for _ in range(10):
            native.run_iteration()
        winner = native.pareto_trees[int(np.argmax(native.pareto_fitnesses[:, 0]))]
        assert (
            hashlib.sha256(repr(winner.structural_key()).encode()).hexdigest()
            == row.genotype_sha256
        )
        assert nfit == row.fit_calls and winner.nodes_count == row.nodes


def bits(x):
    votes = x[:, :, 1] > 0.5
    return np.stack([votes[:, i] != votes[:, j] for i, j in PAIRS], axis=1)


def check_pool(rows, x, y=None, ctx=None):
    assert len(rows) == 1
    row = rows.iloc[0]
    assert row.inputs_sha256 == sha(x) and row.n == len(x)
    if y is not None:
        assert row.labels_sha256 == sha(y, "u1")
    if ctx is not None:
        assert row.context_sha256 == sha(ctx, "u1")
    return row


def close(a, b, eps=2e-12):
    np.testing.assert_allclose(a, b, rtol=0, atol=eps, equal_nan=True)


def check_source(df, x, y, c, pred, ordered, rank, prefix, nsel):
    assert len(df) == 15
    losses = np.square(pred - y)
    for j, name in enumerate(ordered):
        row = df.loc[name]
        assert row.prediction_sha256 == sha(pred[j])
        assert row.n == len(y) and row.n_changed == c.sum()
        close(
            [row.brier, row.brier_clean, row.brier_changed],
            [losses[j].mean(), losses[j, ~c].mean(), losses[j, c].mean()],
        )
        if name == "top_n":
            assert row.n_selected == nsel and json.loads(row.ranking) == list(rank)
            close(json.loads(row.prefix_scores), prefix)
    return np.stack(
        [
            losses[:12].mean(axis=1),
            losses[:12, ~c].mean(axis=1),
            losses[:12, c].mean(axis=1),
        ]
    )


class ChosenRows:
    """Replay gap-proportional row choices through the exact helper's integer-mass RNG."""

    def __init__(self, gaps, indices):
        denom = math.lcm(*(v.denominator for v in gaps))
        self.weights = [int(abs(v) * denom) for v in gaps]
        self.starts = list(accumulate(self.weights, initial=0))[:-1]
        self.indices = iter(indices)

    def randrange(self, m):
        assert sum(self.weights) == m
        i = next(self.indices)
        assert self.weights[i] > 0
        return int(self.starts[i])


def fixed_pool(z, f, g, worlds, seed, exact, points):
    delta = f - g
    scale = np.mean(np.abs(delta))
    center = np.mean(delta * (f + g - 1))
    close([z["center"], z["mean_gap"]], [center, scale])
    oracle = ((f - worlds) ** 2 - (g - worlds) ** 2).mean(axis=1)
    close(z["true_gap"], oracle)
    reference_pool = exact(f, g) if points else None
    if points:
        close(
            [center, scale],
            [
                float(reference_pool.offset),
                float(reference_pool.total_gap / reference_pool.n_rows),
            ],
        )
    repetitions = 64
    signed = 1.0 - 2 * worlds.astype(float)
    ids = z["draw_indices"]
    if scale:
        rngs = np.random.SeedSequence(seed).spawn(repetitions)
        u = np.stack([np.random.default_rng(sub).random(128) for sub in rngs])
        active = np.flatnonzero(delta)
        cdf = np.cumsum(np.abs(delta[active]) / np.abs(delta[active]).sum())
        cdf /= cdf[-1]
        idx = np.stack(
            [(u * len(f)).astype(int), active[np.searchsorted(cdf, u, side="right")]]
        )
        np.testing.assert_array_equal(ids, idx)
    else:
        assert np.all(ids == -1)
    numerical_intervals = 0
    for j, k in enumerate(BUDGET):
        if scale:
            cut = ids[:, :, :k]
            paid = np.array([[len(set(draw)) for draw in method] for method in cut])
            obs = center + np.mean((signed * delta)[:, cut[0]], axis=-1)
            signs = (signed * delta)[:, cut[1]] > 0
            successes = signs.sum(axis=-1)
            hh = center + scale * (2 * successes / k - 1)
            low, high = binomial(k)
            lower = center + scale * (2 * low[successes] - 1)
            upper = center + scale * (2 * high[successes] - 1)
        else:
            paid = np.zeros((2, repetitions), int)
            obs = hh = lower = upper = np.zeros((len(worlds), repetitions))
        np.testing.assert_array_equal(z["n_unique"][:, :, j], paid)
        close(z["estimate"][0, :, :, j], obs)
        close(z["estimate"][1, :, :, j], hh)
        close(z["interval_lower"][:, :, j], lower)
        close(z["interval_upper"][:, :, j], upper)
        var = (signed * delta).var(axis=1) / k
        gapvar = var - np.abs(delta).var() / k
        close(z["exact_variance"][:, :, j], np.stack([var, gapvar]))
        if points:
            for world in (0, 4):
                # One replayed schedule on the existing independent pinned implementation.
                fx = reference_pool.sample(
                    lambda i, world=world: int(worlds[world, i]),
                    k,
                    ChosenRows(reference_pool.gaps, ids[1, 0, :k]),
                )
                close(
                    [float(fx.estimate), fx.lower, fx.upper, fx.labels_read],
                    [hh[world, 0], lower[world, 0], upper[world, 0], paid[1, 0]],
                )
                numerical_intervals += 1
    order = sorted(range(len(delta)), key=lambda i: -abs(delta[i]))
    for j, k in enumerate((0, *BUDGET)):
        seen = [i for i in order[:k] if delta[i] != 0.0]
        known = center + (signed[:, seen] * delta[seen]).sum(1) / len(f)
        missing = np.ones(len(f), bool)
        missing[seen] = False
        radius = np.abs(delta[missing]).sum() / len(f)
        close(z["det_lower"][:, j], known - radius)
        close(z["det_upper"][:, j], known + radius)
        assert z["det_n_unique"][j] == len(seen)
        if points:
            fx = reference_pool.attainable(lambda i: int(worlds[0, i]), k)
            close(
                [float(fx.lower), float(fx.upper), fx.labels_read],
                [known[0] - radius, known[0] + radius, len(seen)],
            )
    return numerical_intervals


@cache
def binomial(k):
    intervals = [
        binomtest(s, k).proportion_ci(0.95, method="exact") for s in range(k + 1)
    ]
    return np.array([v.low for v in intervals]), np.array([v.high for v in intervals])


def check_audit_rows(table, z, w, risks):
    f, g = risks
    truth = z["true_gap"][w]
    assert len(table) == 10
    for r in table.itertuples():
        k = r.budget
        close(
            [r.transport_test_brier, r.validation_test_brier, r.test_pair_oracle],
            [f, g, min(f, g)],
        )
        if r.proposal == "deterministic":
            j = (0, *BUDGET).index(k)
            a, b = z["det_lower"][w, j], z["det_upper"][w, j]
            decision = (
                "transport" if b < 0 else "validation" if a >= 0 else "unresolved"
            )
            certified = decision != "unresolved"
            chosen = (f if decision == "transport" else g) if certified else np.nan
            assert r.repetitions == 1 and r.det_decision == decision
            assert str(r.det_certified).lower() == str(certified).lower()
            assert (
                str(r.det_contains_pool).lower()
                == str(a - 2e-15 <= truth <= b + 2e-15).lower()
            )
            close(
                [
                    r.det_lower,
                    r.det_upper,
                    r.det_width,
                    r.mean_unique_labels,
                    r.det_certified_test_brier,
                    r.det_certified_test_regret,
                ],
                [a, b, b - a, z["det_n_unique"][j], chosen, chosen - min(f, g)],
            )
            continue
        i = ("uniform", "gap").index(r.proposal)
        j = BUDGET.index(k)
        assert r.repetitions == 64
        estimates = z["estimate"][i, w, :, j]
        paid = z["n_unique"][i, :, j]
        select = estimates < 0
        chosen = np.where(select, f, g)
        for name, val in {
            "true_pool_gap": truth,
            "mean_abs_gap": z["mean_gap"],
            "center": z["center"],
            "transport_test_brier": f,
            "validation_test_brier": g,
            "mean_unique_labels": paid.mean(),
            "sum_unique_labels": paid.sum(),
            "mean_estimate": estimates.mean(),
            "mse": np.mean((estimates - truth) ** 2),
            "rmse": np.sqrt(np.mean((estimates - truth) ** 2)),
            "true_estimator_variance": z["exact_variance"][i, w, j],
            "transport_rate": select.mean(),
            "pool_sign_error": np.mean(select != (truth < 0)),
            "independent_sign_error": np.mean(select != (f < g)),
            "test_brier": chosen.mean(),
            "test_regret": np.mean(chosen - min(f, g)),
        }.items():
            close(getattr(r, name), val)
        if r.proposal == "gap":
            lo, hi = z["interval_lower"][w, :, j], z["interval_upper"][w, :, j]
            close(
                [r.cp_excludes_zero, r.cp_contains_pool, r.cp_width],
                [
                    np.mean((lo > 0) | (hi < 0)),
                    np.mean((lo - 2e-15 <= truth) & (truth <= hi + 2e-15)),
                    np.mean(hi - lo),
                ],
            )


def predict(trees, frozen, x):
    return np.stack(
        [decoded(program, x) for program in trees] + [model(x) for model in frozen]
    )


def fixture(exact):
    """Equal input forecasts on two rows with different hidden labels, plus a zero-gap pair."""
    f = np.array([0.75, 0.75, 0.25, 0.25])
    g = np.array([0.25, 0.25, 0.25, 0.25])
    labels = np.array([0, 1, 0, 1])
    pair = exact(f, g)
    sample = pair.sample(
        lambda i: int(labels[i]), 4, ChosenRows(pair.gaps, [0, 0, 1, 1])
    )
    assert (
        sample.draws == 4
        and sample.labels_read == 2
        and sample.estimate == 0
        and sample.positives == 2
    )
    assert sample.lower < 0 < sample.upper
    assert pair.attainable(lambda i: int(labels[i]), 2).lower == 0
    noop = exact(f, f).sample(
        lambda i: (_ for _ in ()).throw(AssertionError(i)), 128, None
    )
    assert noop.draws == noop.labels_read == 0 and noop.lower == noop.upper == 0


def check(folder, okapi):
    import okapi.okapi as core

    pinned = "3f64a166f1882d256d068347308fc446e3ec9d58"
    assert Path(core.__file__).resolve().is_relative_to(okapi.resolve())
    if (okapi / ".git").exists():
        assert (
            subprocess.check_output(
                ["git", "-C", str(okapi), "rev-parse", "HEAD"], text=True
            ).strip()
            == pinned
        )
    impl = okapi / "experiments" / "paired_label_audit" / "design.py"
    assert (
        hashlib.sha256(impl.read_bytes()).hexdigest()
        == "44fe490a6883d9c2055abbd04098a1ba2e68d97249681d61f6cdac19a41376c6"
    )
    exact = run_path(str(impl))["PairedBrierAudit"]
    fixture(exact)
    cols = ["candidates", "source", "risks", "selections", "audit", "pools"]
    tables = {
        s: pd.read_csv(folder / f"{s}.csv.gz", float_precision="round_trip")
        for s in cols
    }
    with np.load(folder / "audit-draws.npz") as data:
        arrays = {k: data[k] for k in data.files}
    source_checked = risks_checked = selection_checked = pools_checked = (
        pinned_intervals
    ) = native = 0
    rounding_witness = {}
    for bi, bank in enumerate(range(7500, 7516)):
        cand, source, risks, selections, audit, pools = (
            tables[key][tables[key].data_seed.eq(bank)] for key in cols
        )
        cand = cand.sort_values("gp_seed")
        assert cand.gp_seed.tolist() == list(range(941, 953))
        programs = [json.loads(v) for v in cand.program]
        a, b, y = observation(bank, 0, 384)
        mask = np.random.default_rng([bank, 11]).random(384) < 0.1
        train = mixed(a, b, mask)
        check_pool(pools[pools.role.eq("fit")], train, y, mask)
        pools_checked += 1
        rank, prefix, nsel, frozen = refs(train, y)
        trained = predict(programs, frozen, train)
        check_source(
            source[source.role.eq("fit")].set_index("method"),
            train,
            y,
            mask,
            trained,
            MODELS,
            rank,
            prefix,
            nsel,
        )
        source_checked += len(MODELS)
        if bank in (7500, 7515):
            for number in (941, 952):
                bare_gp(number, train, y, cand[cand.gp_seed.eq(number)].iloc[0])
                native += 1
        a, b, ytest = observation(bank, 1, 16384)
        check_pool(pools[pools.role.eq("eval_clean")], a, ytest)
        check_pool(pools[pools.role.eq("eval_changed")], b, ytest)
        pools_checked += 2
        ta, tb = predict(programs, frozen, a), predict(programs, frozen, b)
        rc = np.mean((ta - ytest) ** 2, axis=1)
        rs = np.mean((tb - ytest) ** 2, axis=1)
        rr = np.mean((tb - (1 - ytest)) ** 2, axis=1)
        all_risks = {}
        for p in P:
            rmat = np.vstack(
                [(1 - p) * rc + p * ((1 - lam) * rs + lam * rr) for lam in L]
            )
            all_risks[p] = rmat
            for i, model in enumerate(MODELS):
                sub = risks[risks.method.eq(model) & risks.target_p.eq(p)].sort_values(
                    "lambda"
                )
                assert (
                    len(sub) == 5
                    and sub.clean_prediction_sha256.eq(sha(ta[i])).all()
                    and sub.changed_prediction_sha256.eq(sha(tb[i])).all()
                )
                for key, expected in (
                    ("brier", rmat[:, i]),
                    ("brier_clean", rc[i]),
                    ("brier_changed", (1 - L) * rs[i] + L * rr[i]),
                    ("brier_changed_original", rs[i]),
                    ("brier_changed_flipped", rr[i]),
                ):
                    close(sub[key], expected)
                risks_checked += 5
        for rep in range(8):
            a, b, vy = observation(bank, 100 + rep, 256)
            sc = np.random.default_rng([bank, 401, rep]).random(256) < 0.1
            sx = mixed(a, b, sc)
            check_pool(
                pools[pools.role.eq("validation") & pools["repeat"].eq(rep)], sx, vy, sc
            )
            pools_checked += 1
            sp = predict(programs, frozen, sx)
            source_losses = check_source(
                source[
                    source.role.eq("validation") & source["repeat"].eq(rep)
                ].set_index("method"),
                sx,
                vy,
                sc,
                sp,
                MODELS,
                rank,
                prefix,
                nsel,
            )
            source_checked += len(MODELS)
            feats = bits(sx)
            a_sum, b_sum = feats[~sc].mean(axis=0), feats[sc].mean(axis=0)
            feature = int(np.argmax(np.abs(b_sum - a_sum)))
            ma, mb, _ = observation(bank, 200 + rep, 512)
            um = np.random.default_rng([bank, 402, rep]).random(512)
            aa, ab, ay = observation(bank, 300 + rep, 512)
            ua = np.random.default_rng([bank, 403, rep]).random(512)
            flips = np.random.default_rng([bank, 404, rep]).random(512)
            for pi, p in enumerate(P):
                mc, ac = um < p, ua < p
                monitor = mixed(ma, mb, mc)
                audit_x = mixed(aa, ab, ac)
                mon = bits(monitor)[:, feature]
                group = pools[pools["repeat"].eq(rep) & pools.target_p.eq(p)]
                mrow = check_pool(group[group.role.eq("monitor")], monitor, ctx=mc)
                assert mrow.selected_bit_sha256 == sha(mon, "u1")
                arow = check_pool(group[group.role.eq("audit")], audit_x, ay, ac)
                assert arow.flip_uniform_sha256 == sha(flips)
                pools_checked += 2
                diff = b_sum[feature] - a_sum[feature]
                phat = (
                    np.clip((mon.mean() - a_sum[feature]) * diff / (diff * diff), 0, 1)
                    if diff**2 >= 1e-12
                    else 0.1
                )
                ordinary, lo, hi = source_losses
                iv = int(np.argmin(ordinary))
                it = int(np.argmin((1 - phat) * lo + phat * hi))
                ik = int(np.argmin((1 - p) * lo + p * hi))
                f = decoded(programs[it], audit_x)
                g = decoded(programs[iv], audit_x)
                if (bank, rep, p) == (7505, 6, 0.5):
                    scalar = np.clip((mon.mean() - a_sum[feature]) / diff, 0, 1)
                    alt = int(np.argmin((1 - scalar) * lo + scalar * hi))
                    rounding_witness = {
                        "bank": bank,
                        "repeat": rep,
                        "p": p,
                        "protocol_projection": float(phat),
                        "scalar_division": float(scalar),
                        "protocol_seed": 941 + it,
                        "scalar_seed": 941 + alt,
                        "max_change_on_audit_pool": float(
                            np.max(abs(f - decoded(programs[alt], audit_x)))
                        ),
                    }
                assert (
                    sha(f) == arow.transport_prediction_sha256
                    and sha(g) == arow.validation_prediction_sha256
                ), (bank, rep, p, it, iv)
                worlds = np.stack(
                    [np.where(ac & (flips < lam), 1 - ay, ay) for lam in L]
                )
                assert json.loads(arow.world_labels_sha256) == [
                    sha(w, "u1") for w in worlds
                ]
                z = {name: val[bi, rep, pi] for name, val in arrays.items()}
                for key, want in [
                    ("f", f),
                    ("g", g),
                    ("audit_context", ac),
                    ("monitor_bit", mon),
                    ("source_context", sc),
                    ("world_labels", worlds),
                ]:
                    np.testing.assert_array_equal(z[key], want)
                pinned_intervals += fixed_pool(
                    z,
                    f,
                    g,
                    worlds,
                    [bank, 405, rep, round(100 * p)],
                    exact,
                    points=rep == 0,
                )
                label_delta = (tb[it] - tb[iv]) * (2 * ytest - 1)
                slope = float(2 * p * label_delta.mean())
                risk = all_risks[p]
                d = risk[:, it] - risk[:, iv]
                selsegs = selections[
                    selections["repeat"].eq(rep) & selections.target_p.eq(p)
                ].sort_values("lambda")
                auditsegs = audit[audit["repeat"].eq(rep) & audit.target_p.eq(p)]
                for world, (_, row) in enumerate(selsegs.iterrows()):
                    assert (
                        row.validation_idx == iv
                        and row.transport_idx == it
                        and row.known_p_idx == ik
                        and row.feature_index == feature
                    )
                    assert (row.left, row.right) == PAIRS[feature]
                    assert (
                        row.validation_seed == 941 + iv
                        and row.transport_seed == 941 + it
                        and row.known_p_seed == 941 + ik
                    )
                    # Compute the oracle from the stored-precision interpolation, which also defines the tie rule.
                    pub = (1 - p) * rc + p * (rs + L[world] * (rr - rs))
                    i_best = int(np.argmin(pub[:12]))
                    assert row.oracle_seed == 941 + i_best
                    risk_world = risk[world]
                    for field, v in {
                        "estimated_p": phat,
                        "monitor_bit_mean": mon.mean(),
                        "clean_bit_mean": a_sum[feature],
                        "changed_bit_mean": b_sum[feature],
                        "d": d[world],
                        "d0": d[0],
                        "d1": d[-1],
                        "slope": slope,
                        "transport_brier": risk_world[it],
                        "validation_brier": risk_world[iv],
                        "known_p_brier": risk_world[ik],
                        "oracle_brier": risk_world[i_best],
                        "transport_brier_clean": rc[it],
                        "validation_brier_clean": rc[iv],
                        "transport_source_brier": ordinary[it],
                        "validation_source_brier": ordinary[iv],
                    }.items():
                        close(row[field], v)
                    close(row.d, row.d0 + L[world] * slope)
                    assert bool(row.strict_rank_reversal) == (row.d0 * row.d1 < 0)
                    check_audit_rows(
                        auditsegs[auditsegs["lambda"].eq(L[world])],
                        z,
                        world,
                        (row.transport_brier, row.validation_brier),
                    )
                    selection_checked += 1
    assert (
        native == 4
        and source_checked == 2160
        and risks_checked == 2400
        and selection_checked == 1280
        and pools_checked == 688
    )
    return {
        "inputs_sha256": {
            k: hashlib.sha256((folder / f"{k}.csv.gz").read_bytes()).hexdigest()
            for k in cols
        },
        "packed_sha256": hashlib.sha256(
            (folder / "audit-draws.npz").read_bytes()
        ).hexdigest(),
        "verified_native_replays": native,
        "verified_frozen_programs": len(tables["candidates"]),
        "verified_pool_records": pools_checked,
        "verified_source_rows": source_checked,
        "verified_risk_rows": risks_checked,
        "verified_world_selection_rows": selection_checked,
        "verified_audit_aggregate_rows": len(tables["audit"]),
        "verified_scheduled_intervals": 16 * 8 * 2 * 5 * 64 * 3,
        "verified_estimates": 16 * 8 * 2 * 5 * 64 * 3 * 2,
        "pinned_OKAPI_CP_cross_checks": pinned_intervals,
        "pinned_OKAPI_sharp_bounds_cross_checks": 16 * 2 * 4,
        "pinned_OKAPI_commit": pinned,
        "floating_tie_witness": rounding_witness,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--okapi", type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(1):
        result = check(args.folder, args.okapi)
    (args.folder / "independent-verification.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )

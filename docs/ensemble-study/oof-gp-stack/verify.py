"""Audit out-of-fold weights, held-out predictions and paired registry estimates."""

import csv
import gzip
import hashlib
import json
import runpy
from math import sqrt
from pathlib import Path
from statistics import fmean, stdev

import numpy as np
from scipy.optimize import minimize
from scipy.stats import t
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from fusionbench.eval.full_suite_point import SHIFTS
from fusionbench.scenarios.registry import STANDARD_V1_SCENARIOS as SCENARIOS

FOLDER = Path(__file__).resolve().parent
METHODS = ("hybrid", "logit", "prob_lr", "linear_blend", "gp", "mean", "top_n")
# A separate interpreter was compared with native OKAPI on 480 earlier trees.
interpret = runpy.run_path(str(FOLDER.parent / "prune-frozen-mix" / "verify.py"))[
    "probabilities"
]


def read_csv(name):
    opener = gzip.open if name.endswith(".gz") else open
    with opener(FOLDER / name, "rt", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def digest(array, dtype):
    return hashlib.sha256(np.asarray(array, dtype=dtype).tobytes()).hexdigest()


def draw(name, seed, stream):
    scenario = SCENARIOS[name](**(SHIFTS[name] if stream == 2 else {}))
    x, y, _ = scenario._generate(
        640 if stream == 0 else 4096, np.random.default_rng([seed, stream])
    )
    return x, np.asarray(y).ravel()


def near(x, y, atol=1e-12):
    np.testing.assert_allclose(x, y, atol=atol, rtol=0)


def normalized(p):
    clipped = np.maximum(p, 1e-12)
    return clipped / clipped.sum(axis=1, keepdims=True)


def logistic(train, y, test, logit):
    def features(x):
        x = x.reshape(len(x), -1)
        if logit:
            x = x.clip(1e-6, 1 - 1e-6)
            x = np.log(x / (1 - x))
        return x

    scaler = StandardScaler()
    train_z = scaler.fit_transform(features(train))
    lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000).fit(train_z, y)
    proba = lr.predict_proba(scaler.transform(features(test)))
    padded = np.zeros((len(test), test.shape[-1]))
    padded[:, lr.classes_] = proba
    return normalized(padded)


def top_prefix(x, y):
    target = np.eye(x.shape[-1])[y]

    def loss(p):
        return float(((normalized(p) - target) ** 2).sum(1).mean())

    rank = np.argsort(
        [loss(x[:, i, :].astype(float)) for i in range(x.shape[1])], kind="stable"
    )
    total = np.zeros_like(target)
    scores = []
    for n, i in enumerate(rank, 1):
        total += x[:, i, :]
        scores.append(loss(total / n))
    return rank, scores


def average_prefix(x, chosen):
    total = np.zeros((len(x), x.shape[-1]))
    for i in chosen:
        total += x[:, i, :]
    return normalized(total / len(chosen))


def check_optimum(oof, target, weights):
    # The binary/multiclass Brier quadratic is convex. Its directional
    # derivative from an optimum to every simplex vertex must be nonnegative.
    weights = np.asarray(weights)
    assert np.isfinite(weights).all() and weights.min() >= -1e-12
    near(weights.sum(), 1.0)
    n, k, c = oof.shape
    residual = (oof - target[:, None, :]).transpose(1, 0, 2).reshape(k, n * c)
    gram = residual @ residual.T / n
    level = weights @ gram @ weights
    assert min(gram @ weights - level) >= -1e-10
    independently = minimize(
        lambda w: w @ gram @ w,
        np.full(k, 1 / k),
        jac=lambda w: 2 * gram @ w,
        bounds=[(0.0, 1.0)] * k,
        constraints=[
            {"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones(k)}
        ],
        method="SLSQP",
        options={"ftol": 1e-14, "maxiter": 200},
    )
    assert independently.success, independently.message
    near(level, independently.fun, atol=2e-10)


def ci(x):
    m = fmean(x)
    half = stdev(x) * float(t.ppf(0.975, len(x) - 1)) / sqrt(len(x))
    return m, m - half, m + half


def derive_statistics(scores):
    """Treat the registry as fixed and each of its twelve replicate means as one observation."""
    losses = {}
    diffs = {}
    family = {}
    leave_out = {}
    for split in ("iid", "shift"):
        names = [s for s in SCENARIOS if split == "iid" or s in SHIFTS]
        for left in METHODS:
            per_rep = [
                fmean(scores[s, r, split, left] for s in names) for r in range(12)
            ]
            losses[split, left] = ci(per_rep)
            for right in METHODS:
                if left == right:
                    continue
                d = {
                    (s, r): scores[s, r, split, left] - scores[s, r, split, right]
                    for s in names
                    for r in range(12)
                }
                for s in names:
                    family[s, split, left, right] = ci([d[s, r] for r in range(12)])
                    leave_out[s, split, left, right] = fmean(
                        d[other, r] for other in names if other != s for r in range(12)
                    )
                diffs[split, left, right] = ci(
                    [fmean(d[s, r] for s in names) for r in range(12)]
                )
    return losses, diffs, family, leave_out


def check_folds(record, x, y, oof, rep, verify_search):
    seed = record["data_seed"]
    folds = list(
        StratifiedKFold(n_splits=3, shuffle=True, random_state=seed).split(x, y)
    )
    near(np.sort(np.concatenate([held for _, held in folds])), np.arange(640), atol=0)
    folds.append((np.arange(640), np.empty(0, dtype=int)))
    assert len(record["folds"]) == 4
    for fold, ((train, held), meta) in enumerate(
        zip(folds, record["folds"], strict=True)
    ):
        sub_seed = int(np.random.SeedSequence([seed, fold, 781]).generate_state(1)[0])
        order = np.random.default_rng(sub_seed).permutation(train)
        fit, select = np.split(order, [len(train) * 4 // 5])
        assert not set(held).intersection(train) and not set(fit).intersection(select)
        for indices, field, hashfield in (
            (train, "train", "train_sha256"),
            (fit, "fit_gp", "fit_sha256"),
            (select, "select_gp", "select_sha256"),
            (held, "holdout", "holdout_sha256"),
        ):
            assert len(indices) == meta[field]
            assert (digest(indices, "<f8") if len(indices) else None) == meta[hashfield]
        assert (
            meta["calls"] == 600
            and 0 < meta["unique"] <= 600
            and len(meta["gp_digest"]) == 64
        )
        if len(held) and rep == 0:
            for idx, mode in enumerate((True, False)):
                near(logistic(x[train], y[train], x[held], mode), oof[held, idx])
            if verify_search:
                from fusionbench.eval.native_switch import serialize
                from fusionbench.eval.seed_geometry_standard import fit_seed_geometry

                cp = fit_seed_geometry(
                    x[fit],
                    y[fit],
                    x[select],
                    y[select],
                    seed=sub_seed,
                    shape="mixed_branch",
                    budgets=(600,),
                )[600]
                assert (cp.calls, cp.n_trees, cp.path_digest) == (
                    600,
                    meta["unique"],
                    meta["gp_digest"],
                )
                probs = interpret(serialize(cp.val_tree.root), x[held])
                near(probs, oof[held, 2], atol=0)


def audit_tables(scores):
    losses, gaps, family, leave_out = derive_statistics(scores)
    names = [
        (sp, n)
        for sp in ("iid", "shift")
        for n in SCENARIOS
        if sp == "iid" or n in SHIFTS
    ]
    fs = read_csv("families.csv")
    assert len(fs) == len(names)
    assert {(r["split"], r["scenario"]) for r in fs} == set(names)
    for row in fs:
        for m in METHODS:
            near(
                fmean(scores[row["scenario"], r, row["split"], m] for r in range(12)),
                float(row[m]),
            )
    contrast = read_csv("contrasts.csv")
    expected = {
        (sp, ref) for sp in ("iid", "shift") for ref in METHODS if ref != "hybrid"
    }
    assert (
        len(contrast) == len(expected)
        and {(r["split"], r["reference"]) for r in contrast} == expected
    )
    for row in contrast:
        sp, ref = row["split"], row["reference"]
        k = sp, "hybrid", ref
        near([float(row[f]) for f in ("delta", "low", "high")], gaps[k])
        fc = {name: family[name, *k][0] for split, name in names if split == sp}
        loo = [leave_out[name, *k] for name in fc]
        near(
            [float(row[f]) for f in ("loo_min", "loo_max", "worst_delta")],
            [min(loo), max(loo), max(fc.values())],
        )
        assert max(fc, key=fc.get) == row["worst"] and int(row["families"]) == len(fc)
        assert sum(v < 0 for v in fc.values()) == int(row["wins"])
    ranked = sorted(METHODS, key=lambda m: losses["iid", m][0])
    print("IID ranking:", [(m, round(losses["iid", m][0], 8)) for m in ranked])
    best = min(("logit", "prob_lr"), key=lambda m: losses["iid", m][0])
    for left, right in [
        ("hybrid", "logit"),
        ("hybrid", "prob_lr"),
        ("hybrid", "linear_blend"),
        ("linear_blend", best),
    ]:
        print(f"{left} minus {right}:", gaps["iid", left, right])


def main():
    from loguru import logger

    logger.disable("okapi")
    with gzip.open(FOLDER / "banks.jsonl.gz", "rt", encoding="utf-8") as stream:
        banks = [json.loads(line) for line in stream]
    manifests = {(r["scenario"], r["replicate"]): r for r in banks}
    assert len(banks) == len(manifests) == 228
    rows = read_csv("scores.csv.gz")
    refs = {
        (r["scenario"], int(r["replicate"]), r["split"], r["method"]): r for r in rows
    }
    assert len(rows) == len(refs) == 2352
    replayed = 0
    expected = set()
    with (
        threadpool_limits(limits=1),
        np.load(FOLDER / "oof.npz", allow_pickle=False) as arrays,
    ):
        assert set(arrays.files) == {f"b{r['data_seed']}" for r in banks}
        for i, name in enumerate(SCENARIOS):
            for rep in range(12):
                record = manifests[name, rep]
                seed = 1320000 + 1000 * i + rep
                assert record["data_seed"] == seed
                x, y = draw(name, seed, 0)
                oof = arrays[f"b{seed}"]
                c = x.shape[-1]
                assert (
                    len(x) == 640
                    and oof.dtype == np.float64
                    and oof.shape == (len(y), 3, c)
                )
                assert (
                    np.isfinite(oof).all()
                    and oof.min() >= -1e-12
                    and oof.max() <= 1 + 1e-12
                )
                near(oof.sum(axis=2), np.ones((640, 3)))
                check_folds(
                    record,
                    x,
                    y,
                    oof,
                    rep,
                    verify_search=name
                    in ("compositional_switch", "multiclass_committee"),
                )
                for w, k in [(record["weights"], 3), (record["linear_weights"], 2)]:
                    check_optimum(oof[:, :k], np.eye(c)[y], w)
                ranked, prefix_losses = top_prefix(x, y)
                assert list(ranked) == record["ranking"]
                near(prefix_losses, record["prefix_brier"])
                n_selected = 1 + np.argmin(prefix_losses)
                assert n_selected == record["top_n"]
                chosen = ranked[:n_selected]
                for sp, stream in [("iid", 1)] + (
                    [("shift", 2)] if name in SHIFTS else []
                ):
                    for m in METHODS:
                        tag = name, rep, sp, m
                        expected.add(tag)
                        assert int(refs[tag]["data_seed"]) == seed and np.isfinite(
                            float(refs[tag]["brier"])
                        )
                        assert len(refs[tag]["prediction_sha256"]) == 64
                    if rep != 0:
                        continue
                    z, labels = draw(name, seed, stream)
                    a, b = (logistic(x, y, z, mode) for mode in (True, False))
                    g = interpret(record["tree"], np.asarray(z, dtype=float))
                    vectors = np.stack([a, b, g], 1)
                    predictions = {
                        "logit": a,
                        "prob_lr": b,
                        "gp": g,
                        "mean": z.mean(1),
                        "top_n": average_prefix(z, chosen),
                        "linear_blend": np.einsum(
                            "nmc,m->nc", vectors[:, :2], record["linear_weights"]
                        ),
                        "hybrid": np.einsum("nmc,m->nc", vectors, record["weights"]),
                    }
                    for m, p in predictions.items():
                        ref = refs[name, rep, sp, m]
                        near(
                            float(ref["brier"]),
                            ((p - np.eye(c)[labels]) ** 2).sum(1).mean(),
                        )
                        assert digest(p, "<f8") == ref["prediction_sha256"], (
                            name,
                            sp,
                            m,
                        )
                        replayed += 1
    assert set(refs) == expected
    audit_tables({key: float(r["brier"]) for key, r in refs.items()})
    print(
        f"{len(banks)} banks, {4 * len(banks)} fold partitions, {2 * len(banks)} optimal simplex fits, "
        f"{len(rows)} scores, {replayed} test prediction hashes, 6 native GP fold replays."
    )


if __name__ == "__main__":
    main()

"""Independently audit archived risks, paired bootstrap intervals and the fixed selection."""

import json
import subprocess
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np
import pandas as pd
from banks import banks
from threadpoolctl import threadpool_limits

from fusionbench.transfer import load_preset

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
REFS = ("regularized", "mean", "top_n", "prob_lr")
ELIGIBLE = (
    (1, "gp_top8"),
    (2, "simplex"),
    (3, "bootstrap_medoid"),
    (4, "cf_gain"),
    (4, "cf_margin"),
)
CAPS = {"standard": 0.005, "digits_449": 0.004, "strong": 0.010}
TOL = 2e-12


def table(r, name):
    return pd.read_csv(OUT / f"round{r}-{name}.csv", float_precision="round_trip")


def norm(p):
    p = np.maximum(np.asarray(p, dtype=float), 1e-12)
    return p / p.sum(1, keepdims=True)


def empirical(p, labels):
    targets = np.eye(p.shape[-1])[labels]
    return float(((p - targets) ** 2).sum(1).mean())


def audit_risk(b, p):
    if b.audit_y is not None:
        return empirical(p, b.audit_y)
    # Integrate both possible label outcomes directly, independent of the runner's
    # posterior excess + irreducible-noise decomposition.
    return float(2 * b.mass @ (b.q * (1 - p[:, 1]) ** 2 + (1 - b.q) * p[:, 1] ** 2))


def h(p):
    return sha256(np.ascontiguousarray(p, dtype=np.float64).tobytes()).hexdigest()


def mean_and_prefix(x, y, audit):
    # Top-N normalizes each provider for Brier ranking, then accumulates the raw
    # probability vectors in stable rank order and normalizes their prefix average.
    losses = [empirical(norm(x[:, j]), y) for j in range(x.shape[1])]
    order = np.argsort(losses, kind="stable")
    fit_sum = np.zeros((len(x), x.shape[-1]))
    audit_sum = np.zeros((len(audit), x.shape[-1]))
    best = np.inf
    for k, j in enumerate(order, 1):
        fit_sum += x[:, j]
        audit_sum += audit[:, j]
        p = norm(fit_sum / k)
        loss = empirical(p, y)
        if loss < best:
            best = loss
            best_fit, best_audit, n_top = p, norm(audit_sum / k), k
    return {
        "mean": (norm(x.mean(1)), norm(audit.mean(1)), None),
        "top_n": (best_fit, best_audit, n_top),
    }


def summarize(r, scores):
    idx = ["group", "method"]
    levels = table(r, "levels").set_index(idx)
    contrasts = table(r, "contrasts").set_index(idx + ["reference"])
    fams = table(r, "families").set_index(["family", "method"])
    assert levels.index.is_unique and contrasts.index.is_unique and fams.index.is_unique
    differences, summary_errors = {}, []
    n_l = n_c = n_f = 0
    groups = scores.family.where(scores.cohort != "standard", "standard")
    for g, subset in scores.groupby(groups):
        matrix = subset.pivot(
            index=["family", "seed"], columns="method", values="test_brier"
        ).sort_index()
        assert np.isfinite(matrix).all().all()
        ns, nf = matrix.index.levels[1].size, matrix.index.levels[0].size
        assert len(matrix) == nf * ns
        v = matrix.to_numpy().reshape(nf, ns, -1)
        if not g.startswith("digits"):
            sampled = np.random.default_rng(89000 + r).integers(
                ns, size=(10000, nf, ns)
            )
            # Draws are paired across methods. Frequencies per seed within each fixed family
            # re-create all resampled means without resampling the families themselves.
            counts = np.stack([(sampled == j).sum(2) for j in range(ns)], axis=-1)
        for j, m in enumerate(matrix.columns):
            n_l += 1
            lr = levels.loc[g, m]
            assert (lr.families, lr.seeds_per_family) == (nf, ns)
            summary_errors.append(abs(lr.brier - v[:, :, j].mean()))
            if g == "standard":
                fam_mean = matrix[m].groupby("family").mean()
                fam_reg = matrix.regularized.groupby("family").mean()
                saved = fams.xs(m, level="method").loc[fam_mean.index]
                n_f += len(saved)
                summary_errors.extend(np.abs(fam_mean - saved.brier))
                summary_errors.extend(np.abs((fam_mean - fam_reg) - saved.delta_reg))
            for ref in REFS:
                if ref == m:
                    continue
                n_c += 1
                d = v[:, :, j] - v[:, :, matrix.columns.get_loc(ref)]
                delta = float(d.mean())
                cr = contrasts.loc[g, m, ref]
                assert (int((d < -1e-12).sum()), int((d > 1e-12).sum()), d.size) == (
                    cr.better_banks,
                    cr.worse_banks,
                    cr.total,
                )
                summary_errors.append(abs(delta - cr.delta))
                lo = hi = None
                if g.startswith("digits"):
                    assert pd.isna(cr.lo) and pd.isna(cr.hi)
                else:
                    boot = np.einsum("bfs,fs->b", counts, d) / d.size
                    lo, hi = map(float, np.quantile(boot, (0.025, 0.975)))
                    summary_errors.extend((abs(lo - cr.lo), abs(hi - cr.hi)))
                differences[g, m, ref] = (delta, lo, hi)
    assert (n_l, n_c, n_f) == (len(levels), len(contrasts), len(fams))
    assert max(summary_errors) < TOL
    return differences, {
        "levels": n_l,
        "contrasts": n_c,
        "family_rows": n_f,
        "max_summary_error": float(max(summary_errors)),
    }


def choose(differences):
    records = []
    for r, name in ELIGIBLE:
        ds = {g: differences[r][g, name, "regularized"][0] for g in ("weak", *CAPS)}
        violation = max([0.0] + [(ds[g] - cap) / cap for g, cap in CAPS.items()])
        records.append({"round": r, "candidate": name, **ds, "violation": violation})
    return min(records, key=lambda i: (i["violation"], i["weak"])), records


def separation_probe(r, b, observed):
    import run

    original_variants, original_train = run.variants, run.train
    state = {"remaining": None, "reads": 0}

    def variants(r, x, y):
        assert x is b.x and y is b.y
        models, info = original_variants(r, x, y)
        state["remaining"] = len(set(REFS) - set(models))
        return models, info

    def train(model, x, y):
        fitted = original_train(model, x, y)
        if state["remaining"] is not None:
            state["remaining"] -= 1
        return fitted

    class BlindedBank:
        x, y = b.x, b.y

        def __getattr__(self, name):
            if name in ("audit_x", "audit_y", "mass", "q", "context"):
                assert state["remaining"] == 0, (
                    f"Evaluation access before fit finished: {name}"
                )
                state["reads"] += 1
            # Flip all audit posterior probabilities. Source probabilities and labels
            # remain exactly the same, so frozen predictor hashes must match the archive.
            if name == "q" and b.q is not None:
                return 1 - b.q
            if name == "audit_y" and b.audit_y is not None:
                return (b.audit_y + 1) % b.x.shape[-1]
            return getattr(b, name)

    with patch.object(run, "variants", variants), patch.object(run, "train", train):
        rows, _ = run.measure(r, BlindedBank())
    assert state["reads"] > 0
    archived = observed.set_index("method")
    assert set(archived.index) == {v["method"] for v in rows}
    for v in rows:
        assert (
            v["test_prediction_hash"]
            == archived.loc[v["method"], "test_prediction_hash"]
        )
    assert any(
        abs(v["test_brier"] - archived.loc[v["method"], "test_brier"]) > 1e-4
        for v in rows
    )
    return {
        "seed": b.context["seed"],
        "family": b.context["family"],
        "audit_change": "complement posterior"
        if b.q is not None
        else "advance held-out label modulo class count",
        "identical_forecast_hashes": len(rows),
        "early_audit_accesses": 0,
    }


def candidate_model(b, chosen):
    name, r = chosen["candidate"], chosen["round"]
    if r == 2:
        from fusionbench.methods.simplex_tree import SimplexTree

        models = SimplexTree(regularized=True)
        task = "binary" if b.x.shape[-1] == 2 else "multiclass"
        return models.fit(b.x, b.y, task=task, n_classes=b.x.shape[-1])
    if r == 4:
        from r4 import fit_variant

        return fit_variant(b.x, b.y, name)[0]
    from importlib import import_module

    return import_module(f"r{r}").fit_variants(b.x, b.y)[0][name]


def arm_records(r, b, block, info):
    """Check every recorded fit-time decision that has an independently stored witness."""
    if r == 1:
        for m, limit in (
            ("regularized", info["archive_size"]),
            ("gp_top1", 1),
            ("gp_top8", 8),
        ):
            logged = info["choices"][m]
            assert logged["eligible_gp"] == min(limit, info["archive_size"])
            assert 1 <= logged["gp_discovery_rank"] <= logged["eligible_gp"]
            assert 0 <= logged["gp_visit_index"] < info["archive_size"]
            assert logged["validation_brier"] == block.loc[m, "validation_brier"]
    if r == 2:
        n, k, _ = b.x.shape
        for m in ("simplex", "simplex_free"):
            w = np.array(info[m]["weights"])
            recipe = info[m]["info"]
            assert w.shape == (k,) and np.all(w >= 0) and abs(w.sum() - 1) < TOL
            penalty = (
                np.sqrt(k / n) * np.sum((w - 1 / k) ** 2) if m == "simplex" else 0.0
            )
            assert abs(recipe["penalty"] - penalty) < TOL
            # Score the saved coefficients by direct arithmetic independent of native tree code.
            x = np.array(b.audit_x, dtype=float)
            x /= x.sum(2, keepdims=True)
            estimate = np.einsum("nmc,m->nc", x, w)
            assert abs(audit_risk(b, estimate) - block.loc[m, "test_brier"]) < 1e-10
    if r == 3:
        selected = int(np.argmin(info["source_distances"]))
        assert info["selected_index"] == selected
        assert info["bootstrap_seeds"] == [None, 1061, 1062, 1063]
        assert info["optimizer_seeds"] == [0, 1, 2, 3]
        assert info["split_seed"] == 0
        if selected == 0:
            assert (
                block.loc["bootstrap_medoid", "test_prediction_hash"]
                == block.loc["regularized", "test_prediction_hash"]
            )
    if r == 4:
        assert info["fold_seed"] == 7304
        assert sum(fold["held"] for fold in info["fold_rows"]) == len(b.y)
        gain = info["oof_brier"]["simplex"] - info["oof_brier"]["regularized"]
        assert abs(gain - info["oof_gain"]) < TOL
        for method, margin in (("cf_gain", 0.0), ("cf_margin", info["oof_se"])):
            expected = "regularized" if info["oof_gain"] > margin else "simplex"
            assert expected == info["selected_family"][method]
            assert (
                block.loc[method, "test_prediction_hash"]
                == block.loc[expected, "test_prediction_hash"]
            )


def check_round(r, winner=None):
    scores = table(r, "scores")
    fits = [
        json.loads(line)
        for line in (OUT / f"round{r}-fits.jsonl").read_text().splitlines()
    ]
    metadata = {(row["cohort"], row["family"], row["seed"]): row for row in fits}
    groups = scores.groupby(["cohort", "family", "seed"])
    expect = 220 if r == 5 else 106
    assert len(groups) == len(metadata) == len(fits) == expect
    assert not scores.duplicated(["cohort", "family", "seed", "method"]).any()
    methods = set(scores.method.unique())
    arms = {
        1: {"gp_top1", "gp_top8"},
        2: {"simplex", "simplex_free"},
        3: {"bootstrap_medoid", "bootstrap_average"},
        4: {"simplex", "cf_gain", "cf_margin"},
    }
    assert methods == set(REFS) | (
        arms[r] if r < 5 else {"simplex", winner["candidate"]}
    )
    assert (
        scores.fit_brier.between(0, 2 + TOL).all()
        and scores.test_brier.between(0, 2 + TOL).all()
    )
    assert scores.test_prediction_hash.str.fullmatch("[0-9a-f]{64}").all()
    assert scores.excess.notna().equals(scores.cohort == "controlled")
    assert set(scores.loc[scores.cohort == "standard", "family"]) == set(
        load_preset("standard-v5")["scenarios"]
    )
    errors, replay, seen, probes = [], [], set(), []
    for b in banks(r):
        ctx = b.context
        key = ctx["cohort"], ctx["family"], ctx["seed"]
        assert key not in seen
        seen.add(key)
        block = groups.get_group(key).set_index("method")
        assert set(block.index) == methods
        arm_records(r, b, block, metadata[key]["variants"])
        for col, value in ctx.items():
            if col.endswith("_sha256") or col in (
                "n_fit",
                "n_audit",
                "n_classes",
                "n_providers",
            ):
                assert (block[col] == value).all() and metadata[key][col] == value
        if b.mass is not None:
            bayes = np.dot(b.mass, 2 * b.q * (1 - b.q))
            errors.extend(np.abs(block.test_brier - block.excess - bayes))
            if key[1] == "equal":
                assert abs(block.loc["mean", "excess"]) < TOL
        for m, (fp, ep, k) in mean_and_prefix(b.x, b.y, b.audit_x).items():
            assert h(ep) == block.loc[m, "test_prediction_hash"], (r, key, m)
            if k is not None:
                assert k == block.loc[m, "providers_used"]
            errors.extend(
                (
                    abs(empirical(fp, b.y) - block.loc[m, "fit_brier"]),
                    abs(audit_risk(b, ep) - block.loc[m, "test_brier"]),
                )
            )
        if key[1] == "weak" and key[2] == 270000 + 1000 * r:
            probes.append(separation_probe(r, b, block.reset_index()))
        if r == 5 and key == ("digits", "digits_449", 275200):
            probes.append(separation_probe(r, b, block.reset_index()))
        if r == 5 and (key[1], key[2]) in (
            ("weak", 275000),
            ("committee", 275100),
            ("multiclass_committee", 275100),
            ("digits_449", 275200),
        ):
            model = candidate_model(b, winner)
            p = model.predict(b.audit_x)
            m = winner["candidate"]
            assert h(p) == block.loc[m, "test_prediction_hash"]
            score = audit_risk(b, p)
            fit_score = empirical(model.predict(b.x), b.y)
            errors.extend(
                (
                    abs(score - block.loc[m, "test_brier"]),
                    abs(fit_score - block.loc[m, "fit_brier"]),
                )
            )
            replay.append(
                {
                    "family": key[1],
                    "seed": key[2],
                    "test_prediction_hash": h(p),
                    "audit_brier": score,
                    "fit_brier": fit_score,
                }
            )
    assert max(errors) < TOL and seen == set(metadata)
    assert len(probes) == (2 if r == 5 else 1)
    assert (r < 5 and not replay) or (r == 5 and len(replay) == 4)
    delta, stats = summarize(r, scores)
    return delta, {
        **stats,
        "banks": len(seen),
        "raw_score_rows": len(scores),
        "methods": sorted(methods),
        "equal_hash_mean_topn_replays": 2 * len(seen),
        "max_raw_risk_error": max(errors),
        "audit_probes": probes,
        "finalist_replays": replay,
    }


def main():
    start = perf_counter()
    # The frozen choice must be tracked in its own pre-confirmation commit.
    rel = "docs/ensemble-study/stability-five/selection.json"
    root = HERE.parents[2]
    commit = subprocess.check_output(
        ["git", "log", "-1", "--diff-filter=A", "--format=%H", "--", rel],
        cwd=root,
        text=True,
    ).strip()
    if not commit:
        raise RuntimeError(
            "The candidate-selection commit must exist before confirmation verification"
        )
    frozen_bytes = subprocess.check_output(["git", "show", f"{commit}:{rel}"], cwd=root)
    assert frozen_bytes == (HERE / "selection.json").read_bytes()
    assert not subprocess.check_output(
        [
            "git",
            "ls-tree",
            commit,
            "--",
            rel.replace("selection.json", "results/round5-scores.csv"),
        ],
        cwd=root,
    )
    frozen = json.loads(frozen_bytes)
    deltas, rounds = {}, {}
    with threadpool_limits(limits=1):
        for r in range(1, 5):
            deltas[r], rounds[r] = check_round(r)
        winner, screen = choose(deltas)
        assert (winner["round"], winner["candidate"]) == (
            frozen["round"],
            frozen["candidate"],
        )
        assert frozen["limits"] == CAPS and frozen["passed_screen"] == (
            winner["violation"] == 0
        )
        for our, saved in zip(screen, frozen["screen"], strict=True):
            assert (our["round"], our["candidate"]) == (
                saved["round"],
                saved["candidate"],
            )
            for g in ("weak", *CAPS):
                assert abs(our[g] - saved[g]) < TOL
            assert abs(our["violation"] - saved["largest_scaled_violation"]) < TOL
            assert (our["violation"] == 0) == saved["passes_screen"]
        for name, digest in frozen["screen_table_sha256"].items():
            assert sha256((OUT / name).read_bytes()).hexdigest() == digest
        deltas[5], rounds[5] = check_round(5, winner)
    m = winner["candidate"]
    d = {
        g: deltas[5][g, m, "regularized"]
        for g in ("weak", "strong", "standard", "digits_449", "digits_100")
    }
    checks = {
        "weak_upper_below_zero": d["weak"][2] < 0,
        "standard_upper_within_005": d["standard"][2] <= 0.005,
        "strong_upper_within_010": d["strong"][2] <= 0.010,
        "digits449_mean_within_004": d["digits_449"][0] <= 0.004,
    }
    claim = json.loads((OUT / "confirmation.json").read_text())
    assert claim["candidate"] == m and claim["checks"] == checks
    assert claim["primary_confirmed"] == checks["weak_upper_below_zero"]
    assert claim["broad_criteria"] == all(checks.values())
    final = table(5, "scores")
    families = final[final.cohort == "standard"].pivot_table(
        index="family", columns="method", values="test_brier"
    )
    gains = families[m] - families["regularized"]
    assert (
        int((gains < -1e-12).sum()),
        int((np.abs(gains) <= 1e-12).sum()),
        int((gains > 1e-12).sum()),
    ) == (
        claim["improved_families"],
        claim["tied_families"],
        claim["worsened_families"],
    )
    for key, endpoints in (
        ("largest_gains", gains.nsmallest(3)),
        ("largest_regressions", gains.nlargest(3)),
    ):
        assert set(claim[key]) == set(endpoints.index)
        np.testing.assert_allclose(
            [claim[key][k] for k in endpoints.index], endpoints, rtol=0, atol=TOL
        )
    output = {
        "rounds": rounds,
        "selection_commit": commit,
        "selection": winner,
        "screen": screen,
        "round5_candidate_minus_regularized_mean_lo_hi": d,
        "round5_candidate_minus_mean_weak": deltas[5]["weak", m, "mean"],
        "confirmation_checks": checks,
        "primary_pass": checks["weak_upper_below_zero"],
        "broad_pass": all(checks.values()),
        "verification_seconds": perf_counter() - start,
    }
    (OUT / "verification.json").write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()

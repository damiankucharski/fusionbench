"""Rank a frozen bank under paired label mechanisms and audit the selected pair."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from fusionbench.eval.native_switch import serialize
from fusionbench.eval.search_stability import _predict_trees
from fusionbench.eval.seed_selection import (
    disagreement_features,
    observations,
    search_champion,
)
from fusionbench.methods.averaging import TopN
from fusionbench.methods.okapi import _brier_objective, _task_postprocessing
from fusionbench.methods.stacking import StackingLogit


@dataclass(frozen=True)
class Setting:
    data_seeds: tuple = tuple(range(7500, 7516))
    gp_seeds: tuple = tuple(range(941, 953))
    generations: int = 10
    repeats: int = 8
    fit_n: int = 384
    val_n: int = 256
    monitor_n: int = 512
    audit_n: int = 512
    test_n: int = 16384
    train_p: float = 0.10
    target_p: tuple = (0.10, 0.50)
    lambdas: tuple = (0.0, 0.25, 0.5, 0.75, 1.0)
    budgets: tuple = (8, 32, 128)
    schedules: int = 64


def digest(x, dtype="<f8"):
    return hashlib.sha256(np.asarray(x, dtype=dtype).tobytes()).hexdigest()


def mix(clean, changed, mask):
    return np.where(mask[:, None, None], changed, clean)


def integrated_risks(pred_clean, pred_changed, y, p, lambdas):
    """Rows are models. Return each model's clean, changed-original and changed-flipped risk."""
    clean = np.mean((pred_clean - y) ** 2, axis=1)
    same = np.mean((pred_changed - y) ** 2, axis=1)
    reverse = np.mean((pred_changed - (1 - y)) ** 2, axis=1)
    changed = same[:, None] + np.asarray(lambdas)[None, :] * (reverse - same)[:, None]
    return clean, same, reverse, changed, (1 - p) * clean[:, None] + p * changed


def validation_losses(pred, y, ctx):
    loss = (pred - y) ** 2
    ordinary = loss.mean(axis=1)
    clean = (
        loss[:, ~ctx].mean(axis=1) if (~ctx).any() else np.full_like(ordinary, np.nan)
    )
    changed = loss[:, ctx].mean(axis=1) if ctx.any() else np.full_like(ordinary, np.nan)
    return ordinary, clean, changed


def select_frozen(source_x, ctx, val_losses, monitor, source_p, target_p):
    """Select both trees using only labeled source validation and unlabeled monitoring inputs."""
    from okapi.selection import estimate_context_prevalence

    ordinary, clean, changed = val_losses
    features = disagreement_features(source_x)[0]
    m_features = disagreement_features(monitor)[0]
    clean_features, changed_features = features[~ctx], features[ctx]
    both = bool(len(clean_features) and len(changed_features))
    j = (
        int(np.argmax(abs(changed_features.mean(0) - clean_features.mean(0))))
        if both
        else 0
    )
    m_bit = m_features[:, j]
    means = clean_features[:, j : j + 1], changed_features[:, j : j + 1]
    estimated = float(estimate_context_prevalence([m_bit.mean()], *means, source_p))
    source = int(np.argmin(ordinary))

    def pick(p):
        return int(np.argmin((1 - p) * clean + p * changed)) if both else source

    return {
        "validation_idx": source,
        "transport_idx": pick(estimated),
        "known_p_idx": pick(target_p),
        "estimated_p": estimated,
        "feature_index": j,
        "monitor_bit_mean": float(m_bit.mean()),
        "clean_bit_mean": float(means[0].mean()) if len(means[0]) else None,
        "changed_bit_mean": float(means[1].mean()) if len(means[1]) else None,
        "missing_context": not both,
    }, m_bit


def _bank(args):
    from loguru import logger
    from threadpoolctl import threadpool_limits

    from fusionbench.eval.frozen_pair_sampling import audit_fixed_pair

    bank, cfg = args
    logger.disable("okapi")
    tables = {
        k: [] for k in ("candidates", "source", "risks", "selections", "audit", "pools")
    }
    arrays = {}
    rows = lambda name: tables[name]

    def pool_record(role, rep, p, x, y=None, ctx=None, **extra):
        rows("pools").append(
            {
                "data_seed": bank,
                "repeat": rep,
                "target_p": p,
                "role": role,
                "n": len(x),
                "inputs_sha256": digest(x),
                "labels_sha256": None if y is None else digest(y, "u1"),
                "context_sha256": None if ctx is None else digest(ctx, "u1"),
                **extra,
            }
        )

    a, b, y = observations(bank, 0, cfg.fit_n)
    fit_ctx = np.random.default_rng([bank, 11]).random(cfg.fit_n) < cfg.train_p
    train = mix(a, b, fit_ctx)
    pool_record("fit", -1, cfg.train_p, train, y, fit_ctx)
    trees, unique_index, lookup = [], [], {}
    models = list(cfg.gp_seeds)
    names = [f"gp_{s}" for s in models] + ["mean", "top_n", "stack_logit"]
    with _task_postprocessing("binary"), threadpool_limits(limits=1):
        for seed in models:
            calls = 0

            def counted(pred, gt):
                nonlocal calls
                calls += 1
                return _brier_objective(pred, gt)

            start = perf_counter()
            tree = search_champion(train, y, seed, cfg.generations, objective=counted)
            seconds = perf_counter() - start
            key = tree.structural_key()
            if key not in lookup:
                lookup[key] = len(trees)
                trees.append(tree)
            unique_index.append(lookup[key])
            rows("candidates").append(
                {
                    "data_seed": bank,
                    "gp_seed": seed,
                    "method": f"gp_{seed}",
                    "genotype": lookup[key],
                    "program": json.dumps(serialize(tree.root), separators=(",", ":")),
                    "nodes": tree.nodes_count,
                    "fit_calls": calls,
                    "generations": cfg.generations,
                    "search_seconds": seconds,
                    "genotype_sha256": hashlib.sha256(repr(key).encode()).hexdigest(),
                }
            )
        unique_index = np.array(unique_index)
        top = TopN("brier").fit(train, y, task="binary", n_classes=2)
        stack = StackingLogit().fit(train, y, task="binary", n_classes=2)

        def predict(x):
            gp = _predict_trees(trees, x)[unique_index]
            return np.vstack(
                [gp, x[:, :, 1].mean(1), top.predict(x)[:, 1], stack.predict(x)[:, 1]]
            )

        def source_scores(x, labels, ctx, pred, rep, role):
            overall, clean, changed = validation_losses(pred, labels, ctx)
            for i, name in enumerate(names):
                extra = {}
                if name == "top_n":
                    extra = {
                        "n_selected": top.n_selected_,
                        "ranking": json.dumps(top.ranking_.tolist()),
                        "prefix_scores": json.dumps(top.prefix_scores_.tolist()),
                    }
                rows("source").append(
                    {
                        "data_seed": bank,
                        "repeat": rep,
                        "role": role,
                        "method": name,
                        "n": len(x),
                        "n_changed": int(ctx.sum()),
                        "brier": overall[i],
                        "brier_clean": clean[i],
                        "brier_changed": changed[i],
                        "prediction_sha256": digest(pred[i]),
                        **extra,
                    }
                )
            return overall[: len(models)], clean[: len(models)], changed[: len(models)]

        source_scores(train, y, fit_ctx, predict(train), -1, "fit")
        test_a, test_b, test_y = observations(bank, 1, cfg.test_n)
        pa, pb = predict(test_a), predict(test_b)
        pool_record("eval_clean", -1, None, test_a, test_y)
        pool_record("eval_changed", -1, None, test_b, test_y)
        risk_tables = {}
        for p in cfg.target_p:
            clean, old, reverse, new, overall = integrated_risks(
                pa, pb, test_y, p, cfg.lambdas
            )
            risk_tables[p] = (clean, new, overall)
            for i, name in enumerate(names):
                for j, lam in enumerate(cfg.lambdas):
                    rows("risks").append(
                        {
                            "data_seed": bank,
                            "target_p": p,
                            "lambda": lam,
                            "method": name,
                            "brier": overall[i, j],
                            "brier_clean": clean[i],
                            "brier_changed": new[i, j],
                            "brier_changed_original": old[i],
                            "brier_changed_flipped": reverse[i],
                            "clean_prediction_sha256": digest(pa[i]),
                            "changed_prediction_sha256": digest(pb[i]),
                        }
                    )
        for rep in range(cfg.repeats):
            a, b, sy = observations(bank, 100 + rep, cfg.val_n)
            sc = np.random.default_rng([bank, 401, rep]).random(cfg.val_n) < cfg.train_p
            sx = mix(a, b, sc)
            source = source_scores(sx, sy, sc, predict(sx), rep, "validation")
            pool_record("validation", rep, cfg.train_p, sx, sy, sc)
            m0, m1, _ = observations(bank, 200 + rep, cfg.monitor_n)
            m_uniform = np.random.default_rng([bank, 402, rep]).random(cfg.monitor_n)
            a0, a1, ay = observations(bank, 300 + rep, cfg.audit_n)
            a_uniform = np.random.default_rng([bank, 403, rep]).random(cfg.audit_n)
            flip = np.random.default_rng([bank, 404, rep]).random(cfg.audit_n)
            for pi, p in enumerate(cfg.target_p):
                mc = m_uniform < p
                monitor = mix(m0, m1, mc)
                ac = a_uniform < p
                ax = mix(a0, a1, ac)
                worlds = np.where(
                    ac[None, :] & (flip[None, :] < np.asarray(cfg.lambdas)[:, None]),
                    1 - ay,
                    ay,
                )
                choose, mb = select_frozen(sx, sc, source, monitor, cfg.train_p, p)
                ia, ib, ik = [
                    choose[k]
                    for k in ("transport_idx", "validation_idx", "known_p_idx")
                ]
                tg, val = unique_index[ia], unique_index[ib]
                predicted = _predict_trees(
                    [trees[i] for i in dict.fromkeys([tg, val])], ax
                )
                f, g = predicted[0], predicted[-1]
                pool_record(
                    "monitor",
                    rep,
                    p,
                    monitor,
                    None,
                    mc,
                    selected_bit_sha256=digest(mb, "u1"),
                )
                pool_record(
                    "audit",
                    rep,
                    p,
                    ax,
                    ay,
                    ac,
                    flip_uniform_sha256=digest(flip),
                    world_labels_sha256=json.dumps([digest(z, "u1") for z in worlds]),
                    transport_prediction_sha256=digest(f),
                    validation_prediction_sha256=digest(g),
                )
                seed = [bank, 405, rep, round(100 * p)]
                simulated = audit_fixed_pair(
                    f,
                    g,
                    worlds,
                    seed=seed,
                    draw_budgets=cfg.budgets,
                    repetitions=cfg.schedules,
                )
                slot = f"r{rep}_p{pi}"
                arrays[slot] = _audit_arrays(simulated, f, g, worlds, ac, mb, sc)
                clean, changed, overall = risk_tables[p]
                pair_slope = (
                    2 * p * float(np.mean((pb[ia] - pb[ib]) * (2 * test_y - 1)))
                )
                d0 = float(overall[ia, 0] - overall[ib, 0])
                d1 = float(overall[ia, -1] - overall[ib, -1])
                if not np.isclose(d1 - d0, pair_slope, rtol=0, atol=2e-15):
                    raise AssertionError(
                        "Direct paired Brier and analytic slope disagree"
                    )
                left, right = np.triu_indices(sx.shape[1], 1)
                for j, lam in enumerate(cfg.lambdas):
                    test = overall[: len(models), j]
                    best = int(np.argmin(test))
                    scores = {
                        "transport": test[ia],
                        "validation": test[ib],
                        "known_p": test[ik],
                        "oracle": test[best],
                    }
                    metadata = {
                        "data_seed": bank,
                        "repeat": rep,
                        "target_p": p,
                        "lambda": lam,
                    }
                    dt = float(test[ia] - test[ib])
                    rows("selections").append(
                        {
                            **metadata,
                            **choose,
                            "target_n": cfg.monitor_n,
                            "validation_n": cfg.val_n,
                            "source_changed": int(sc.sum()),
                            "left": int(left[choose["feature_index"]]),
                            "right": int(right[choose["feature_index"]]),
                            "transport_seed": models[ia],
                            "validation_seed": models[ib],
                            "known_p_seed": models[ik],
                            "oracle_seed": models[best],
                            "different_seed": ia != ib,
                            "different_program": tg != val,
                            "transport_source_brier": source[0][ia],
                            "validation_source_brier": source[0][ib],
                            "transport_brier_clean": clean[ia],
                            "validation_brier_clean": clean[ib],
                            "transport_brier_changed": changed[ia, j],
                            "validation_brier_changed": changed[ib, j],
                            **{name + "_brier": val for name, val in scores.items()},
                            "transport_regret": test[ia] - test[best],
                            "validation_regret": test[ib] - test[best],
                            "d": dt,
                            "d0": d0,
                            "d1": d1,
                            "slope": pair_slope,
                            "strict_rank_reversal": d0 * d1 < 0,
                        }
                    )
                    rows("audit").extend(
                        audit_summary(
                            simulated, j, metadata, float(test[ia]), float(test[ib])
                        )
                    )
    return bank, tables, arrays


def _audit_arrays(result, f, g, worlds, context, monitor_bit, source_context):
    """Pack one realized pair in fixed shape. -1 marks a zero-label draw slot."""
    r = {k: np.asarray(v) for k, v in result.items()}
    ids = r["draw_indices"]
    if ids.shape[-1] == 0:
        ids = np.full((2, ids.shape[1], int(max(r["draw_budgets"]))), -1)
    r["draw_indices"] = ids.astype(np.int32)
    r["n_unique"] = r["n_unique"].astype(np.int32)
    r.update(
        f=f,
        g=g,
        world_labels=worlds.astype(np.uint8),
        audit_context=context.astype(np.uint8),
        monitor_bit=monitor_bit.astype(np.uint8),
        source_context=source_context.astype(np.uint8),
    )
    return r


def audit_summary(result, world, metadata, test_f, test_g):
    """Keep every fixed pair, including ties. Strictly negative gaps select transport."""
    rows = []
    truth = float(result["true_gap"][world])
    common = {
        **metadata,
        "true_pool_gap": truth,
        "mean_abs_gap": result["mean_gap"],
        "center": result["center"],
        "transport_test_brier": test_f,
        "validation_test_brier": test_g,
        "test_pair_oracle": min(test_f, test_g),
    }
    for i, proposal in enumerate(("uniform", "gap")):
        for j, k in enumerate(result["draw_budgets"]):
            est = result["estimate"][i, world, :, j]
            paid = result["n_unique"][i, :, j]
            transport = est < 0
            chosen = np.where(transport, test_f, test_g)
            row = {
                **common,
                "proposal": proposal,
                "budget": int(k),
                "repetitions": len(est),
                "mean_unique_labels": paid.mean(),
                "sum_unique_labels": int(paid.sum()),
                "mean_estimate": est.mean(),
                "mse": np.mean((est - truth) ** 2),
                "rmse": np.sqrt(np.mean((est - truth) ** 2)),
                "true_estimator_variance": result["exact_variance"][i, world, j],
                "transport_rate": transport.mean(),
                "pool_sign_error": np.mean(transport != (truth < 0)),
                "independent_sign_error": np.mean(transport != (test_f < test_g)),
                "test_brier": chosen.mean(),
                "test_regret": np.mean(chosen - min(test_f, test_g)),
            }
            if proposal == "gap":
                low, high = (
                    result["interval_lower"][world, :, j],
                    result["interval_upper"][world, :, j],
                )
                row.update(
                    cp_excludes_zero=np.mean((low > 0) | (high < 0)),
                    cp_contains_pool=np.mean(
                        (low - 2e-15 <= truth) & (truth <= high + 2e-15)
                    ),
                    cp_width=np.mean(high - low),
                )
            rows.append(row)
    for j, k in enumerate(result["det_budgets"]):
        low, high = result["det_lower"][world, j], result["det_upper"][world, j]
        certificate = (
            "transport" if high < 0 else "validation" if low >= 0 else "unresolved"
        )
        certified = certificate != "unresolved"
        chosen = test_f if certificate == "transport" else test_g
        rows.append(
            {
                **common,
                "proposal": "deterministic",
                "budget": int(k),
                "repetitions": 1,
                "mean_unique_labels": int(result["det_n_unique"][j]),
                "det_lower": low,
                "det_upper": high,
                "det_width": high - low,
                "det_certified": certified,
                "det_decision": certificate,
                "det_contains_pool": bool(low - 2e-15 <= truth <= high + 2e-15),
                "det_certified_test_brier": chosen if certified else None,
                "det_certified_test_regret": chosen - min(test_f, test_g)
                if certified
                else None,
            }
        )
    return rows


def run(out, workers=2, setting=None):
    cfg = Setting() if setting is None else setting
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    table_names = ("candidates", "source", "risks", "selections", "audit", "pools")
    tables = {k: [] for k in table_names}
    packed = {}
    started = perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for bank, bank_tables, arrays in pool.map(
            _bank, [(bank, cfg) for bank in cfg.data_seeds]
        ):
            for name, items in bank_tables.items():
                tables[name].extend(items)
            # Outer axes in the numeric file always follow config order.
            for key in arrays["r0_p0"]:
                means = [
                    np.stack(
                        [arrays[f"r{r}_p{p}"][key] for p in range(len(cfg.target_p))]
                    )
                    for r in range(cfg.repeats)
                ]
                packed.setdefault(key, []).append(np.stack(means))
    for name, items in tables.items():
        tmp = out / f"{name}.partial.csv.gz"
        pd.DataFrame(items).to_csv(tmp, index=False)
        tmp.replace(out / f"{name}.csv.gz")
    arr_file = out / "audit-draws.partial.npz"
    np.savez_compressed(arr_file, **{k: np.stack(v) for k, v in packed.items()})
    arr_file.replace(out / "audit-draws.npz")
    cfg_json = {
        **asdict(cfg),
        "elapsed_seconds": perf_counter() - started,
        "gp_search_seconds": sum(row["search_seconds"] for row in tables["candidates"]),
        "fit_objective_calls": sum(row["fit_calls"] for row in tables["candidates"]),
        "npz_outer_axes": ["data_seed", "repeat", "target_p"],
        "npz_proposals": ["uniform", "gap"],
        "npz_inner_axes": {
            "estimate": ["proposal", "lambda", "schedule", "budget"],
            "interval_lower": ["lambda", "schedule", "budget"],
            "interval_upper": ["lambda", "schedule", "budget"],
            "n_unique": ["proposal", "schedule", "budget"],
            "draw_indices": ["proposal", "schedule", "position"],
            "world_labels": ["lambda", "audit_row"],
            "true_gap": ["lambda"],
            "exact_variance": ["proposal", "lambda", "budget"],
            "det_lower": ["lambda", "det_budget"],
            "det_upper": ["lambda", "det_budget"],
            "det_n_unique": ["det_budget"],
            "draw_budgets": ["budget"],
            "det_budgets": ["det_budget"],
            "f": ["audit_row"],
            "g": ["audit_row"],
            "audit_context": ["audit_row"],
            "source_context": ["validation_row"],
            "monitor_bit": ["monitor_row"],
            "center": [],
            "mean_gap": [],
        },
        "draw_indices_missing": -1,
        "table_rows": {k: len(v) for k, v in tables.items()},
    }
    (out / "config.json").write_text(json.dumps(cfg_json, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    run(args.out, args.workers)

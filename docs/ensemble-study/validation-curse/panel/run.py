"""Two-stage standard-v3 panel for selection optimism and independent confirmation."""

import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from fusionbench import get_scenario
from fusionbench.eval.suite_point_gp import predict_tree
from fusionbench.eval.transfer_grid import iter_input_stresses
from fusionbench.methods.averaging import SimpleAverage, TopN
from fusionbench.methods.native_pattern import RegularizedNativePatternOKAPI
from fusionbench.methods.search_guard import SearchGuardedNativeOKAPI
from fusionbench.methods.stacking import StackingLogit, StackingLR
from fusionbench.methods.validation_guard import GuardedNativeOKAPI
from fusionbench.scenarios.blocks import draw_block
from fusionbench.scenarios.registry import STANDARD_SCENARIOS

HERE = Path(__file__).resolve().parent
SUITE = tuple(STANDARD_SCENARIOS)
assert len(SUITE) == 24


def digest(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def score(p, y):
    p = np.asarray(p, dtype=float)
    if not np.isfinite(p).all() or not np.allclose(p.sum(1), 1, atol=1e-6, rtol=0):
        raise ValueError("Invalid predicted probability vectors")
    loss = np.square(p - np.eye(p.shape[1])[y]).sum(1)
    return {"brier": float(loss.mean()), "accuracy": float((p.argmax(1) == y).mean())}


def controls(x, y, task, c):
    methods = {
        "mean": SimpleAverage(),
        "top_n": TopN("brier"),
        "prob_lr": StackingLR(c_value=1),
        "logit_lr": StackingLogit(c_value=1),
    }
    for m in methods.values():
        m.fit(x, y, task=task, n_classes=c)
    return methods


def run_bank(job):
    round_number, scenario_id, rep = job
    index = SUITE.index(scenario_id)
    data_seed = 4_600_000 + 10_000 * round_number + 1_000 * index + rep
    gp_seed = 6_600_000 + 10_000 * round_number + 1_000 * index + rep
    common = {
        "round": round_number,
        "scenario": scenario_id,
        "rep": rep,
        "data_seed": data_seed,
        "search_seed": gp_seed,
    }
    with threadpool_limits(limits=1):
        s = get_scenario(scenario_id)
        x, y, _ = draw_block(s, data_seed, 0, 640, split="fit")
        a, ay, _ = draw_block(s, data_seed, 3, 1024, split="fit")
        z, zy, _ = draw_block(s, data_seed, 1, 1024, split="test")
        _, _, swap = next(
            iter_input_stresses(
                z, data_seed, severities=(0.5,), modes=("provider_swap",)
            )
        )
        domains = {
            "source_audit": (a, ay),
            "suite_test": (z, zy),
            "swapped_test": (swap, zy),
        }
        tick = perf_counter()
        incumbent = RegularizedNativePatternOKAPI(
            seed=gp_seed, split_seed=data_seed
        ).fit(x, y, task=s.task, n_classes=s.n_classes)
        native_s = perf_counter() - tick
        models = controls(x, y, s.task, s.n_classes)
        val = incumbent.validation_indices_
        meta = {
            **common,
            "n_fit": len(y),
            "n_validation": len(val),
            "winner": incumbent.chosen_,
            "n_gp_programs": incumbent.counts_["gp_unique_programs"],
            "gp_path_sha256": incumbent.counts_["gp_path_digest"],
            "native_calls": incumbent.counts_["total_score_calls"],
            "incumbent_seconds": native_s,
            "split_hashes": {
                name: digest(arr)
                for name, arr in {
                    "fit_x": x,
                    "fit_y": y,
                    "audit_x": a,
                    "audit_y": ay,
                    "test_x": z,
                    "test_y": zy,
                    "val_idx": val,
                }.items()
            },
        }
        rows = []

        def add(domain, stage, name, p, labels):
            rows.append(
                {
                    **common,
                    "domain": domain,
                    "stage": stage,
                    "method": name,
                    "n": len(labels),
                    **score(p, labels),
                }
            )

        for domain, (inputs, labels) in domains.items():
            for name, m in models.items():
                add(domain, "deployment", name, m.predict(inputs), labels)
            add(
                domain,
                "deployment",
                "incumbent_full",
                incumbent.predict(inputs),
                labels,
            )

        def frozen(arm, fitted):
            for domain, (inputs, labels) in {
                "internal_validation": (x[val], y[val]),
                "source_audit": (a, ay),
            }.items():
                add(domain, arm, "mean", models["mean"].predict(inputs), labels)
                for name, t in fitted.discovery_trees_.items():
                    p = predict_tree(t, inputs)
                    add(domain, arm, name, p, labels)
                    if name == fitted.chosen_:
                        add(domain, arm, "internal_winner", p, labels)

        frozen("frozen", incumbent)
        if round_number == 2:
            tick = perf_counter()
            challenger = SearchGuardedNativeOKAPI(
                seed=gp_seed, split_seed=data_seed, z=1
            ).fit(x, y, task=s.task, n_classes=s.n_classes)
            np.testing.assert_array_equal(challenger.validation_indices_, val)
            meta.update(
                search_guard_seconds=perf_counter() - tick,
                search_guard_winner=challenger.chosen_,
                search_guard_gp_path_sha256=challenger.counts_["gp_path_digest"],
                search_guard_calls=challenger.counts_["total_score_calls"],
                search_guard_unique_programs=challenger.counts_["gp_unique_programs"],
            )
            frozen("guarded_frozen", challenger)
            tick = perf_counter()
            guard = GuardedNativeOKAPI(seed=gp_seed, split_seed=data_seed).fit(
                x, y, task=s.task, n_classes=s.n_classes
            )
            meta.update(
                outer_seconds=perf_counter() - tick,
                inner_winner=guard.estimator_.chosen_,
                outer_winner=guard.chosen_,
                raw_winner=guard.raw_choice_,
                inner_n=len(guard.inner_indices_),
                outer_n=len(guard.outer_indices_),
                outer_sha256=digest(guard.outer_indices_),
                threshold=guard.guard_decision_.threshold,
                gains=guard.guard_decision_.gains,
                outer_reason=guard.guard_decision_.reason,
            )
            for domain, (inputs, labels) in domains.items():
                add(
                    domain,
                    "deployment",
                    "search_guard_full",
                    challenger.predict(inputs),
                    labels,
                )
                pred = guard.candidate_predictions(inputs)
                add(
                    domain,
                    "deployment",
                    "incumbent_inner",
                    pred[guard.estimator_.chosen_],
                    labels,
                )
                add(domain, "deployment", "outer_raw", pred[guard.raw_choice_], labels)
                add(domain, "deployment", "outer_guard", pred[guard.chosen_], labels)
                for name, pp in pred.items():
                    add(domain, "outer_candidates", name, pp, labels)
            outer = guard.outer_indices_
            for name, pp in guard.candidate_predictions(x[outer]).items():
                add("outer_validation", "outer_candidates", name, pp, y[outer])

        return rows, meta


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--round", type=int, required=True, choices=(1, 2))
    p.add_argument("--workers", type=int, choices=(1, 2, 3), default=3)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    tick = perf_counter()
    jobs = [(args.round, sc, r) for sc in SUITE for r in range(6)]
    scores, banks = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for rows, meta in pool.map(run_bank, jobs):
            scores.extend(rows)
            banks.append(meta)
    frame = pd.DataFrame(scores)
    frame.to_csv(args.out / "scores.csv", index=False)
    (args.out / "banks.json").write_text(json.dumps(banks, indent=2) + "\n")
    (args.out / "run.json").write_text(
        json.dumps(
            {
                "seconds": perf_counter() - tick,
                "workers": args.workers,
                "pid": os.getpid(),
                "n_banks": len(banks),
                "n_rows": len(frame),
                "round": args.round,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

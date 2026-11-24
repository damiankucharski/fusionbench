"""Evaluate evolved-subtree class heads on the two fixed follow-up cohorts.

The published comparison uses native OKAPI revision 7a2a790.
"""

import hashlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from run import load_bank, sha
from threadpoolctl import threadpool_limits

from fusionbench.eval.suite_point_gp import predict_tree
from fusionbench.eval.transfer_grid import brier
from fusionbench.methods.native_pattern import NativePatternOKAPI

STUDY = Path(__file__).resolve().parent
COHORTS = {"digits_hetero_a": 8, "shared_38": 6}
MODES = ("diagonal", "full")


def fit_bank(job):
    cohort, replicate, archived = job
    rows = []
    with threadpool_limits(limits=1):
        data, seed = load_bank(cohort, replicate)
        fingerprint = hashlib.sha256("".join(sha(a) for a in (data.fit_preds, data.fit_y, data.test_preds, data.test_y)).encode()).hexdigest()
        if not all(r["seed"] == seed and r["bank_sha256"] == fingerprint for r in archived.values()):
            raise ValueError(f"Reference bank mismatch for {cohort}, replicate {replicate}")
        for mode in MODES:
            model = NativePatternOKAPI(seed=seed, class_head=mode, class_head_max_evaluations=4096, l2_strength=0.25)
            start = perf_counter()
            model.fit(data.fit_preds, data.fit_y, task=data.task, n_classes=data.n_classes)
            elapsed = perf_counter() - start
            prediction = model.predict(data.test_preds)
            replay = predict_tree(model.trees_[model.chosen_], data.test_preds)
            np.testing.assert_array_equal(prediction, replay)
            loss = brier(prediction, data.test_y)
            pure = archived[f"okapi_class_{mode}"]["brier"]
            flat = brier(predict_tree(model.trees_["flat"], data.test_preds), data.test_y)
            np.testing.assert_allclose(flat, pure, rtol=0, atol=1e-12)
            original = archived["native"]["brier"]
            rows.append({
                "cohort": cohort, "replicate": replicate, "seed": seed, "bank_sha256": fingerprint,
                "mode": mode, "choice": model.chosen_, "brier": loss, "original_native_brier": original,
                "pure_head_brier": pure, "difference_vs_original": loss - original,
                "difference_vs_pure": loss - pure, "augmented_basis_count": len(model.counts_["screen"]["selected"]),
                "total_score_calls": model.counts_["total_score_calls"], "fit_seconds": elapsed,
            })
    return rows


def main():
    reference = pd.read_csv(STUDY / "results" / "scores.csv").set_index(["cohort", "replicate", "method"], verify_integrity=True)
    jobs = [
        (cohort, r, {key: reference.loc[cohort, r, key].to_dict() for key in ("native", "okapi_class_diagonal", "okapi_class_full")})
        for cohort, repeats in COHORTS.items() for r in range(repeats)
    ]
    with ProcessPoolExecutor(max_workers=3) as pool:
        scores = pd.DataFrame([row for both in pool.map(fit_bank, jobs) for row in both])
    scores["exact_chosen_replay"] = True  # Every returned row passed assert_array_equal.
    summaries = []
    for (cohort, mode), group in scores.groupby(["cohort", "mode"], sort=False):
        if len(group) != COHORTS[cohort] or group.seed.nunique() != len(group):
            raise ValueError("Incomplete paired cohort")
        summaries.append({
            "cohort": cohort, "mode": mode, "banks": len(group),
            **group[["brier", "original_native_brier", "pure_head_brier", "difference_vs_original", "difference_vs_pure",
                     "total_score_calls", "fit_seconds"]].mean().add_prefix("mean_").to_dict(),
            **{f"chosen_{name}": int((group.choice == name).sum()) for name in ("flat", "augmented", "gp")},
        })
    out = STUDY / "integrated"
    out.mkdir(exist_ok=True)
    scores.to_csv(out / "scores.csv", index=False)
    pd.DataFrame(summaries).to_csv(out / "summary.csv", index=False)


if __name__ == "__main__":
    main()

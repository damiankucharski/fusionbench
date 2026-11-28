"""Replay the normal fit/predict API on every R5 bank against the frozen diagnostic implementation."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from banks import banks
from threadpoolctl import threadpool_limits

from fusionbench.methods.native_pattern import RegularizedNativePatternOKAPI

OUT = Path(__file__).resolve().parent / "results"
KEY = ["cohort", "family", "seed"]


def replay():
    rows = []
    start = perf_counter()
    with threadpool_limits(limits=1):
        for b in banks(5):
            c = b.x.shape[2]
            model = RegularizedNativePatternOKAPI(
                seed=0, split_seed=0, gp_validation_top_k=8
            )
            model.fit(b.x, b.y, task="binary" if c == 2 else "multiclass", n_classes=c)
            prob = model.predict(b.audit_x)
            gp = model.trees_["gp"].structural_key()
            rows.append(
                {
                    **{k: b.context[k] for k in KEY},
                    "chosen": model.chosen_,
                    "gp_hash": hashlib.sha256(repr(gp).encode()).hexdigest(),
                    "audit_sha256": hashlib.sha256(
                        np.ascontiguousarray(prob).tobytes()
                    ).hexdigest(),
                    "search_path": model.counts_["gp_path_digest"],
                    "archive": model.counts_["gp_unique_programs"],
                    "eligible": model.counts_["gp_final_eligible"],
                    "validation_calls": model.counts_["score_calls"]["gp_validation"],
                }
            )
    assert len(rows) == 220
    pd.DataFrame(rows).to_csv(OUT / "round5-shortlist-api.csv", index=False)
    print(f"Replayed {len(rows)} bank fits in {perf_counter() - start:.1f}s")


def compare():
    api = pd.read_csv(OUT / "round5-shortlist-api.csv").set_index(
        KEY, verify_integrity=True
    )
    scores = (
        pd.read_csv(OUT / "round5-scores.csv")
        .query("method=='gp_top8'")
        .set_index(KEY, verify_integrity=True)
    )
    records = [
        json.loads(line)
        for line in (OUT / "round5-fits.jsonl").read_text().splitlines()
    ]
    assert len(api) == len(scores) == len(records) == 220
    for r in records:
        key = tuple(r[k] for k in KEY)
        native = api.loc[key]
        diagnostics = r["variants"]
        choice = diagnostics["choices"]["gp_top8"]
        assert native.chosen == choice["family"] and native.gp_hash == choice["gp_hash"]
        assert native.audit_sha256 == scores.at[key, "test_prediction_hash"]
        assert native.search_path == diagnostics["native_path"]
        assert (
            native.archive == diagnostics["archive_size"]
            and native.eligible == choice["eligible_gp"]
        )
        assert (
            native.eligible
            <= native.validation_calls
            <= min(2 * native.eligible, native.archive)
        )
    output = {
        "round": 5,
        "banks": len(api),
        "forecast_hash_matches": len(api),
        "gp_champion_matches": len(api),
        "validation_calls_with_full_archive": int(api.archive.sum()),
        "validation_calls_with_shortlist": int(api.validation_calls.sum()),
        "min_validation_calls": int(api.validation_calls.min()),
        "max_validation_calls": int(api.validation_calls.max()),
    }
    (OUT / "round5-shortlist-api-verified.json").write_text(
        json.dumps(output, indent=2) + "\n"
    )
    print(json.dumps(output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare-only", action="store_true")
    args = parser.parse_args()
    if not args.compare_only:
        replay()
    if args.compare_only or (OUT / "round5-fits.jsonl").exists():
        compare()

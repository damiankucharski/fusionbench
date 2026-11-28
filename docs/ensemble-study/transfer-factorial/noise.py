"""Fit each model on source rows once, then enumerate its exact risk under target shifts."""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from fusionbench.methods.averaging import SimpleAverage, TopN
from fusionbench.methods.native_pattern import RegularizedNativePatternOKAPI
from fusionbench.methods.stacking import StackingLR
from fusionbench.scenarios.rare_shortcut_noise import RareShortcutNoise

HERE = Path(__file__).resolve().parent
SEEDS = range(253100, 253116)
STATES = {"clean": (0.0, 0.0), "provider": (0.20, 0.0), "label": (0.0, 0.20)}
SHARES = (1.0, 0.25, 0.0)
REGIONS = {"all": None, "common": 0, "rare": 1}
BOOTSTRAP_SEED = 97943
TOL = 1e-12


def law(train="clean", test="clean", share=0.0):
    core_fit, label_fit = STATES[train]
    core_test, label_test = STATES[test]
    return RareShortcutNoise(
        region_fit=0.05, region_test=0.5, shortcut_fit=1.0, shortcut_test=share,
        provider_noise_fit=core_fit, provider_noise_test=core_test,
        label_noise_fit=label_fit, label_noise_test=label_test,
    )


def sha256(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def models():
    return {"mean": SimpleAverage(), "top_n": TopN(metric="brier"),
            "prob_lr": StackingLR(c_value=1.0),
            "regularized": RegularizedNativePatternOKAPI(seed=0, split_seed=0)}


def risks(p, mass, q):
    keep = mass > 0
    p, q, w = p[keep], q[keep], mass[keep]
    assert np.isclose(w.sum(), 1.0) and np.isfinite(q).all()
    brier = 2 * np.dot(w, q * (1 - p) ** 2 + (1 - q) * p ** 2)
    bayes = 2 * np.dot(w, q * (1 - q))
    excess = 2 * np.dot(w, (p - q) ** 2)
    assert abs(brier - bayes - excess) < TOL
    return {"brier": brier, "bayes": bayes, "excess": excess}


def population_rows(train):
    # Each tuple contains both observed-label and pre-flip outcome posteriors.
    cells = [("source", train, 1.0)] + [
        ("target", state, share) for share in SHARES for state in STATES
    ]
    for domain, state, share in cells:
        split = "fit" if domain == "source" else "test"
        clean_state = "clean" if state == "label" else state
        scenario = law(train, state, share)
        clean = law(clean_state, clean_state, share)
        for name, region in REGIONS.items():
            x, mass, q = scenario.support(split, region=region)
            clean_x, clean_mass, clean_q = clean.support(split, region=region)
            np.testing.assert_array_equal(x, clean_x)
            np.testing.assert_array_equal(mass, clean_mass)
            flip = STATES[state][1]
            keep = mass > 0
            discrepancy = np.max(np.abs(q[keep] - (flip + (1 - 2 * flip) * clean_q[keep])))
            assert discrepancy < TOL
            yield domain, state, share, name, x, mass, q, clean_q, discrepancy


def evaluate(seed):
    raw, forecasts, fits = [], [], []
    for train in STATES:
        source = law(train)
        x, y, latent = source.sample_split(512, np.random.default_rng(seed), "fit")
        clean_train = "clean" if train == "label" else train
        clean_x, clean_y, _ = law(clean_train).sample_split(512, np.random.default_rng(seed), "fit")
        np.testing.assert_array_equal(x, clean_x)
        regions = np.asarray(latent["region"], dtype=bool)
        canonical_x, _, _ = source.support("fit")
        cells = list(population_rows(train))
        for method, model in models().items():
            start = perf_counter()
            model.fit(x, y, task="binary", n_classes=2)
            seconds = perf_counter() - start
            fit_p = model.predict(x)[:, 1]
            p = model.predict(canonical_x)[:, 1]
            assert len(p) == 16 and np.all(np.isfinite(p)) and np.all((p >= 0) & (p <= 1))
            h = sha256(p)
            meta = {"data_seed": seed, "train_noise": train, "method": method,
                    "prediction_sha256": h}
            for i, (state_x, probability) in enumerate(zip(canonical_x, p, strict=True)):
                bits = "".join(str(int(bit)) for bit in (state_x[:, 1] > 0.5))
                forecasts.append({**meta, "state_index": i, "bits": bits, "p1": probability})
            fit_info = {
                **meta, "fit_seconds": seconds, "fit_inputs_sha256": sha256(x),
                "fit_labels_sha256": sha256(y), "fit_clean_labels_sha256": sha256(clean_y),
                "family": getattr(model, "chosen_", method),
                "objective_evaluations": getattr(model, "counts_", {}).get("total_score_calls"),
                "unique_providers": (len(model.trees_[model.chosen_].unique_value_node_ids)
                                     if method == "regularized" else getattr(model, "n_selected_", 4)),
            }
            for name, region in REGIONS.items():
                m = np.ones(len(y), dtype=bool) if region is None else regions == region
                fits.append({**fit_info, "region": name, "n": int(m.sum()),
                             "fit_brier": 2 * np.mean((fit_p[m] - y[m]) ** 2),
                             "fit_clean_brier": 2 * np.mean((fit_p[m] - clean_y[m]) ** 2)})
            for domain, state, share, name, support_x, mass, q, clean_q, q_error in cells:
                np.testing.assert_array_equal(support_x, canonical_x)
                observed, truth = risks(p, mass, q), risks(p, mass, clean_q)
                flip = STATES[state][1]
                identity = ((1 - 2 * flip) * truth["brier"]
                            + 2 * flip * np.dot(mass, p ** 2 + (1 - p) ** 2))
                err = abs(observed["brier"] - identity)
                assert err < TOL
                raw.append({**meta, "domain": domain, "eval_noise": state, "shortcut_share": share,
                            "region": name, **observed, **{"clean_" + k: v for k, v in truth.items()},
                            "brier_identity_error": err, "posterior_identity_error": q_error})
    return raw, forecasts, fits


def contrasts(raw):
    rows = []
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(16, size=(10_000, 16))
    definitions = [
        ("train_label_minus_clean", ("label", "clean"), ("clean", "clean")),
        ("train_provider_minus_clean", ("provider", "clean"), ("clean", "clean")),
        ("training_label_minus_provider", ("label", "clean"), ("provider", "clean")),
        ("test_label_only_minus_matched", ("clean", "label"), ("label", "label")),
        ("test_provider_only_minus_matched", ("clean", "provider"), ("provider", "provider")),
        ("test_label_only_minus_clean", ("clean", "label"), ("clean", "clean")),
        ("test_provider_only_minus_clean", ("clean", "provider"), ("clean", "clean")),
        ("test_label_minus_provider", ("clean", "label"), ("clean", "provider")),
    ]
    groups = raw[raw.domain == "target"].groupby(["method", "shortcut_share", "region"])
    for (method, share, region), cell in groups:
        for metric in ("brier", "excess", "clean_excess"):
            table = cell.pivot(index="data_seed", columns=["train_noise", "eval_noise"],
                               values=metric).reindex(SEEDS)
            assert len(table) == 16 and not table.isna().any().any()
            for name, left, right in definitions:
                d = (table[left] - table[right]).to_numpy()
                low, high = np.quantile(d[draws].mean(axis=1), [0.025, 0.975])
                rows.append({"method": method, "shortcut_share": share, "region": region,
                             "metric": metric, "contrast": name, "mean": d.mean(),
                             "ci_low": low, "ci_high": high, "n_positive": int((d > TOL).sum()),
                             "n_negative": int((d < -TOL).sum()), "n_banks": len(d)})
    return pd.DataFrame(rows)


def ranking(raw):
    records = []
    subset = raw[(raw.domain == "target") & (raw.region == "all")]
    for (train, share), group in subset.groupby(["train_noise", "shortcut_share"]):
        score = group.pivot(index="data_seed", columns=["method", "eval_noise"], values="brier")
        for a, b in combinations(models(), 2):
            clean = (score[a, "clean"] - score[b, "clean"]).to_numpy()
            noisy = (score[a, "label"] - score[b, "label"]).to_numpy()
            flipped = ((clean < -TOL) & (noisy > TOL)) | ((clean > TOL) & (noisy < -TOL))
            records.append({"train_noise": train, "shortcut_share": share, "method_a": a,
                            "method_b": b, "n_banks": len(clean), "n_reversals": flipped.sum(),
                            "clean_mean_a_minus_b": clean.mean(),
                            "label_mean_a_minus_b": noisy.mean()})
    return pd.DataFrame(records)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, default=HERE / "results")
    out = cli.parse_args().output
    out.mkdir(parents=True, exist_ok=True)
    import torch
    torch.set_num_threads(1)
    start = perf_counter()
    rows, predictions, fits = [], [], []
    with threadpool_limits(limits=1):
        for seed in SEEDS:
            r, p, f = evaluate(seed)
            rows.extend(r)
            predictions.extend(p)
            fits.extend(f)
            print(f"Scored {seed}", flush=True)
    scores = pd.DataFrame(rows)
    assert len(scores) == 16 * 3 * 4 * 10 * 3
    for name, df in (("scores", scores), ("predictions", pd.DataFrame(predictions)),
                     ("fits", pd.DataFrame(fits)), ("contrasts", contrasts(scores)),
                     ("rankings", ranking(scores))):
        df.to_csv(out / f"noise-{name}.csv", index=False, float_format="%.17g")
    metadata = {
        "data_seeds": [SEEDS.start, SEEDS.stop - 1], "native_seed": 0, "split_seed": 0,
        "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_draws": 10_000, "n_fit": 512,
        "fits": len(fits) // len(REGIONS), "source_population_rows": int((scores.domain == "source").sum()),
        "target_population_rows": int((scores.domain == "target").sum()),
        "max_brier_identity_error": scores.brier_identity_error.max(),
        "max_posterior_identity_error": scores.posterior_identity_error.max(),
        "seconds": perf_counter() - start,
    }
    (out / "noise-manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()

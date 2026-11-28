"""Post hoc removal of audit label noise on the exact 64 registered round-five scene fits."""

from pathlib import Path

import numpy as np
import pandas as pd
from banks_full import _clustered
from engine import fit_full, fit_model
from interventions import fit_candidate
from run_full import digest, forecast
from threadpoolctl import threadpool_limits

from fusionbench.methods.averaging import SimpleAverage
from fusionbench.scenarios.checkpoint_bank import generate_scene_pool

OUT = Path(__file__).resolve().parent / "results"


def replay():
    archived = pd.read_csv(OUT / "round5-scores.csv", float_precision="round_trip")
    archived = archived.query(
        "cohort == 'clustered' and method in ['full','pull_mean','mean']"
    )
    archived = archived.set_index(["seed", "family", "method"], verify_integrity=True)
    seeds = sorted(archived.index.get_level_values("seed").unique())
    assert seeds == list(range(385300, 385332)) and len(archived) == 192
    rows = []
    for seed in seeds:
        # Keep the oracle separate from source-only replay. Check the target realization
        # against the recorded experiment before integrating over its unseen Bernoulli outcomes.
        oracle = generate_scene_pool(
            seed,
            law="equal",
            n_fit_scenes=128,
            n_audit_scenes=256,
            scene_error_share=0.85,
            event_offset=2.0,
        )["I"]
        for bank in _clustered(5, [seed]):
            family = bank.context["family"]
            models = {
                "full": fit_full(bank.x, bank.y),
                "pull_mean": fit_candidate(
                    "pull_mean", bank.x, bank.y, seed=0, groups=bank.groups
                ),
                "mean": fit_model(SimpleAverage(), bank.x, bank.y),
            }
            np.testing.assert_array_equal(oracle.P[:, :, 0], bank.audit_x[:, :, 1])
            np.testing.assert_array_equal(oracle.y, bank.audit_y)
            np.testing.assert_array_equal(oracle.scene, bank.audit_groups)
            for name, model in models.items():
                record = archived.loc[(seed, family, name)]
                pred = forecast(model, bank.audit_x)
                pred_hash = digest(pred)
                assert record.test_prediction_hash == pred_hash
                assert record.audit_x_sha256 == digest(bank.audit_x)
                assert record.audit_y_sha256 == digest(bank.audit_y)
                loss_zero = (pred[:, 0] - 1) ** 2 + pred[:, 1] ** 2
                loss_one = pred[:, 0] ** 2 + (pred[:, 1] - 1) ** 2
                observed = np.where(oracle.y, loss_one, loss_zero).mean()
                expected = (
                    (1 - oracle.posterior) * loss_zero + oracle.posterior * loss_one
                ).mean()
                np.testing.assert_allclose(observed, record.brier, rtol=0, atol=2e-14)
                rows.append(
                    {
                        "seed": seed,
                        "family": family,
                        "method": name,
                        "observed": observed,
                        "expected": expected,
                        "prediction_sha256": pred_hash,
                    }
                )
    return pd.DataFrame(rows)


def summary(rows):
    out = []
    for family, group in rows.groupby("family"):
        draws = np.random.default_rng(91705).integers(32, size=(10000, 32))
        for m, ref in [("pull_mean", "full"), ("pull_mean", "mean"), ("full", "mean")]:
            deltas = {}
            for metric in ["observed", "expected"]:
                piv = group.pivot(
                    index="seed", columns="method", values=metric
                ).sort_index()
                d = (piv[m] - piv[ref]).to_numpy()
                deltas[metric] = d
            deltas["observed_minus_expected"] = deltas["observed"] - deltas["expected"]
            for quantity, delta in deltas.items():
                lo, hi = np.quantile(delta[draws].mean(axis=1), [0.025, 0.975])
                out.append(
                    {
                        "family": family,
                        "method": m,
                        "reference": ref,
                        "quantity": quantity,
                        "delta": delta.mean(),
                        "lo": lo,
                        "hi": hi,
                        "paired_draws": len(delta),
                    }
                )
    return pd.DataFrame(out)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        scores = replay()
    scores.to_csv(OUT / "round5-scene-posterior-scores.csv", index=False)
    contrasts = summary(scores)
    contrasts.to_csv(OUT / "round5-scene-posterior-contrasts.csv", index=False)
    print(contrasts.to_string(index=False))

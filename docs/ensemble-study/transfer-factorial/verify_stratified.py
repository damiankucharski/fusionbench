"""Check every archived population risk and reconstruct two complete training draws."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from stratified import bank

OUT = Path(__file__).resolve().parent / "results"


def main():
    scores = pd.read_csv(OUT / "stratified-scores.csv")
    summary = pd.read_csv(OUT / "stratified-decomposition.csv")
    stored = np.load(OUT / "stratified-predictions.npz")["p"]
    assert len(scores) == len(stored) == 32 * 3 * 3 * 6
    # Reconstruct the marginal probabilities from the binary coding of each state.
    signs = (np.arange(256)[:, None] >> np.arange(7, -1, -1)) % 2
    evidence = np.concatenate([0.5 + a * (2 * signs - 1) for a in (0.15, 0.35)])
    for (d, n, name), frame in scores.groupby(["specialization", "size", "method"]):
        q = (1 - d) * evidence.mean(axis=1) + d * evidence[:, 0]
        p = stored[frame.index]
        risk = 2 * np.mean(q * (1 - p) ** 2 + (1 - q) * p**2, axis=1)
        np.testing.assert_allclose(risk, frame.population_brier, atol=1e-14, rtol=0)
        mean_p = p.mean(axis=0)
        part = summary[
            (summary.specialization == d)
            & (summary["size"] == n)
            & (summary.method == name)
        ].iloc[0]
        np.testing.assert_allclose(
            [2 * np.mean((p - mean_p) ** 2), 2 * np.mean((mean_p - q) ** 2)],
            [part.variance, part.bias_squared],
            atol=1e-14,
            rtol=0,
        )
        if name == "mean":
            np.testing.assert_allclose(
                p, np.broadcast_to(evidence.mean(axis=1), p.shape), rtol=0, atol=1e-14
            )

    for (_, _, _), group in scores.groupby(["seed", "specialization", "size"]):
        ix = group.set_index("method", drop=False)
        for name in ("guard", "plain"):
            rec = ix.loc[name]
            choose = rec.gate_gain > (rec.gate_se if name == "guard" else 0)
            assert choose == rec[f"{name}_uses_okapi"]
            compare = ix.loc["regularized" if choose else "mean"].population_brier
            np.testing.assert_allclose(
                rec.population_brier, compare, atol=1e-14, rtol=0
            )
    for seed in (255100, 255131):
        rows, p = bank(seed)
        frame = scores[scores.seed == seed]
        np.testing.assert_allclose(p, stored[frame.index], rtol=0, atol=1e-14)
        np.testing.assert_allclose(
            [r["fit_brier"] for r in rows], frame.fit_brier, rtol=0, atol=1e-14
        )

    primary = scores[(scores.specialization == 0.25) & (scores["size"] == 128)]
    empirical = primary.pivot(index="seed", columns="method", values="fit_brier")
    risk = primary.pivot(index="seed", columns="method", values="population_brier")
    diff = (risk.regularized - risk["mean"]).to_numpy()
    draw = np.random.default_rng(97944).integers(32, size=(10000, 32))
    low, high = np.quantile(diff[draw].mean(axis=1), [0.025, 0.975])
    archived = (
        pd.read_csv(OUT / "stratified-contrasts.csv")
        .query(
            "specialization==0.25 and size==128 and method=='regularized' and ref=='mean'"
        )
        .iloc[0]
    )
    np.testing.assert_allclose(
        [diff.mean(), low, high],
        [archived.difference, archived.lo, archived.hi],
        atol=1e-14,
        rtol=0,
    )
    report = {
        "verified_risk_rows": len(scores),
        "verified_prediction_values": stored.size,
        "full_bank_replays": [255100, 255131],
        "primary_reg_minus_mean": float(diff.mean()),
        "primary_interval": [float(low), float(high)],
        "regularized_better_on_fit": int(
            (empirical.regularized < empirical["mean"] - 1e-12).sum()
        ),
        "regularized_worse_on_test": int((diff > 1e-12).sum()),
    }
    (OUT / "verification-stratified.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()

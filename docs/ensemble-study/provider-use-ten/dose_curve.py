"""Descriptive R4 audit of the best attenuation of each frozen mixed-source predictor.

The prediction at s is F+s(M-F). Any full-class Brier expectation over a fixed
target law is therefore quadratic in s. No provider, model or target is sampled.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "results"
DOSES = {"native": 0, "quarter": 0.25, "half": 0.5, "three_quarter": 0.75, "full": 1}
WORLDS = ("s0", "s50", "s100", "specialist80")


def _quadratic(y0, yh, y1):
    a = 2 * (y0 + y1 - 2 * yh)
    return np.array([a, y1 - y0 - a, y0])


def _minimax(left, right):
    """An upper envelope can attain its minimum at a boundary, vertex or crossing."""
    candidates = [0.0, 1.0]
    for a, b, _ in (left, right):
        if a > 0:
            candidates.append(-b / (2 * a))
    for root in np.roots(left - right):
        if abs(root.imag) <= 1e-12:
            candidates.append(float(root.real))
    candidates = sorted({float(s) for s in candidates if 0 <= s <= 1})
    return min(candidates, key=lambda s: max(np.polyval(left, s), np.polyval(right, s)))


def _curve(world, rows):
    target = (
        rows[rows.world == world]
        .set_index("method", verify_integrity=True)
        .loc[list(DOSES)]
    )
    for key in ("target_hash", "test_x_hash", "oracle_brier"):
        if target[key].nunique() != 1:
            raise ValueError(f"Nonidentical {key} within {world}")
    y = target.brier
    curve = _quadratic(y.native, y.half, y.full)
    if curve[0] < -2e-12:
        raise ValueError("The fitted Brier curve must be convex")
    check_doses = ("quarter", "three_quarter")
    error = max(abs(np.polyval(curve, DOSES[name]) - y[name]) for name in check_doses)
    np.testing.assert_allclose(error, 0, rtol=0, atol=2e-12)
    return curve, float(target.oracle_brier.iloc[0]), error


def analyze(raw, oracle):
    keep = (raw.cohort == "focus") & (raw.family == "mixed") & raw.method.isin(DOSES)
    lower = float(oracle["minimax_regret"]["max_excess"])
    records = []
    for seed, source in raw.loc[keep].groupby("seed", sort=True):
        if not (
            source["source_x_hash"].nunique() == source["source_y_hash"].nunique() == 1
        ):
            raise ValueError("Each source seed must identify one fit bank")
        if set(source.world) != set(WORLDS):
            raise ValueError("The dose audit needs the four registered target worlds")
        flags = source.loc[source.method != "native", "drop"].fillna(-1).unique()
        if len(flags) != 1:
            raise ValueError("The four attenuations must share their flagged provider")
        flagged = int(flags[0]) >= 0
        if not flagged:
            if not source.groupby("world").test_hash.nunique().eq(1).all():
                raise ValueError(
                    "An unflagged predictor changed across attenuation levels"
                )
        else:
            for name, s in DOSES.items():
                if not source.loc[source.method == name, "strength"].eq(s).all():
                    raise ValueError("Unexpected attenuation strength")
        curves, baselines, errors = {}, {}, []
        for world in WORLDS:
            curves[world], baselines[world], error = _curve(world, source)
            errors.append(error)
        regrets = {w: c - np.array([0.0, 0.0, baselines[w]]) for w, c in curves.items()}
        for world in ("s0", "s100"):
            np.testing.assert_allclose(
                baselines[world],
                oracle["endpoint_bayes_brier"][world],
                rtol=0,
                atol=2e-12,
            )
        left, right = regrets["s0"], regrets["s100"]
        optimum = _minimax(left, right) if flagged else 0.0
        base_risk = max(left[2], right[2])
        best = max(np.polyval(left, optimum), np.polyval(right, optimum))
        if not lower - 2e-12 <= best <= base_risk + 2e-12:
            raise ValueError(
                "Mixture optimum is outside the paired oracle and native bounds"
            )
        bound = "s100" if right[2] > left[2] else "s0" if left[2] > right[2] else "tie"
        row = {
            "seed": int(seed),
            "flagged_provider": int(flags[0]) if flagged else None,
            "optimal_strength": optimum,
            "native_max_regret": base_risk,
            "optimal_max_regret": best,
            "gain_over_native": base_risk - best,
            "population_minimax_regret": lower,
            "excess_over_population_minimax": best - lower,
            "binding_endpoint_at_zero": bound,
            "max_quarter_check_error": max(errors),
        }
        for world in WORLDS:
            a, b, c = curves[world]
            row.update(
                {
                    f"{world}_brier_quadratic": a,
                    f"{world}_brier_linear": b,
                    f"{world}_brier_intercept": c,
                    f"{world}_bayes_brier": baselines[world],
                    f"{world}_regret_at_optimum": np.polyval(regrets[world], optimum),
                }
            )
        records.append(row)
    output = pd.DataFrame(records)
    if len(output) != 6:
        raise ValueError("R4 specified six distinct mixed-source fits")
    active = output[output.binding_endpoint_at_zero == "s100"]
    summary = {
        "round": 4,
        "cohort": "mixed",
        "sources": len(output),
        "flagged_sources": int(output.flagged_provider.notna().sum()),
        "target_worlds": list(WORLDS),
        "reconstruction_strengths": [0, 0.5, 1],
        "held_back_strengths": [0.25, 0.75],
        "checks_of_measured_quarter_losses": len(output) * 2 * len(WORLDS),
        "largest_absolute_check_error": float(output.max_quarter_check_error.max()),
        "strength_zero_minimizers": int(output.optimal_strength.eq(0).sum()),
        "interior_minimizers": int(
            output.optimal_strength.between(0, 1, inclusive="neither").sum()
        ),
        "strength_one_minimizers": int(output.optimal_strength.eq(1).sum()),
        "mean_native_max_regret": float(output.native_max_regret.mean()),
        "mean_optimal_max_regret": float(output.optimal_max_regret.mean()),
        "mean_gain_over_native": float(output.gain_over_native.mean()),
        "population_minimax_regret": lower,
        "mean_excess_over_population_minimax": float(
            output.excess_over_population_minimax.mean()
        ),
        "s100_binding_at_zero": len(active),
        "s100_binding_with_positive_slope_at_zero": int(
            (active.s100_brier_linear > 0).sum()
        ),
        "s100_slope_range_at_zero": [
            float(output.s100_brier_linear.min()),
            float(output.s100_brier_linear.max()),
        ],
        "s0_slope_range_at_zero": [
            float(output.s0_brier_linear.min()),
            float(output.s0_brier_linear.max()),
        ],
    }
    if summary["s100_binding_with_positive_slope_at_zero"] == len(output):
        summary["interpretation"] = (
            "In every fitted source the active-shortcut endpoint sets the larger Bayes-relative loss at zero attenuation. "
            "The loss at that endpoint increases immediately as attenuation increases. Its convexity then makes "
            "zero the optimum for the whole interval. The population minimax predictor can change each report "
            "pattern independently; the attainable dose curve is restricted to one line from the fitted native predictor."
        )
    return output, summary


if __name__ == "__main__":
    table, result = analyze(
        pd.read_csv(OUT / "r04.csv", float_precision="round_trip"),
        json.loads((OUT / "endpoint-regret-oracle.json").read_text()),
    )
    table.to_csv(OUT / "r04-dose-curves.csv", index=False)
    (OUT / "r04-dose-curves.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(result, indent=2, allow_nan=False))

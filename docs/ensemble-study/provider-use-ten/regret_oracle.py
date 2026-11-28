"""Population risk and regret frontier for the two exact shortcut endpoint laws."""

import json
from pathlib import Path

import numpy as np

from fusionbench.scenarios.rare_shortcut_noise import RareShortcutNoise

OUT = Path(__file__).resolve().parent / "results" / "endpoint-regret-oracle.json"


def support(activity):
    return RareShortcutNoise(
        region_test=0.5,
        shortcut_test=activity,
        provider_noise_test=0.0,
        label_noise_test=0.0,
    ).support("test")


def calculate():
    x0, m0, q0 = support(0)
    x1, m1, q1 = support(1)
    np.testing.assert_array_equal(x0, x1)
    if not np.all((m0 > 0) & (m1 > 0)):
        raise ValueError("Expected common positive endpoint support")
    masses = np.stack((m0, m1))
    posteriors = np.stack((q0, q1))
    baseline = 2 * np.sum(masses * posteriors * (1 - posteriors), axis=1)

    def posterior(t):
        return ((1 - t) * m0 * q0 + t * m1 * q1) / ((1 - t) * m0 + t * m1)

    def excess(p):
        return 2 * np.sum(masses * (p - posteriors) ** 2, axis=1)

    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        e0, e1 = excess(posterior(mid))
        if e0 < e1:
            lo = mid
        else:
            hi = mid
    t_star = (lo + hi) / 2
    optimum = posterior(t_star)
    np.testing.assert_allclose(*excess(optimum), atol=2e-15, rtol=0)

    def record(t):
        p = posterior(t)
        e = excess(p)
        r = baseline + e
        # Independent score from the two possible labels at each observed pattern.
        direct = 2 * np.sum(
            masses * ((1 - posteriors) * p**2 + posteriors * (1 - p) ** 2), axis=1
        )
        np.testing.assert_allclose(r, direct, atol=2e-15, rtol=0)
        return {
            "active_prior": float(t),
            "risk_s0": float(r[0]),
            "risk_s100": float(r[1]),
            "excess_s0": float(e[0]),
            "excess_s100": float(e[1]),
            "max_risk": float(r.max()),
            "max_excess": float(e.max()),
        }

    # Directly enumerate the intermediate generator at t*. This checks the
    # full joint mixture, including the change in report-pattern frequencies.
    _, m_t, q_t = support(t_star)
    np.testing.assert_allclose(m_t, (1 - t_star) * m0 + t_star * m1, atol=2e-15, rtol=0)
    np.testing.assert_allclose(q_t, optimum, atol=2e-15, rtol=0)
    # Independent lower bound obtained by completing the square per pattern.
    dual = 2 * t_star * (1 - t_star) * np.sum(m0 * m1 / m_t * (q1 - q0) ** 2)
    minimax = record(t_star)
    np.testing.assert_allclose(dual, minimax["max_excess"], atol=2e-15, rtol=0)
    absolute = record(0.0)
    np.testing.assert_allclose(absolute["max_risk"], baseline[0], atol=2e-15, rtol=0)
    return {
        "metric": "full binary Brier",
        "region_prevalence": 0.5,
        "endpoint_bayes_brier": {"s0": float(baseline[0]), "s100": float(baseline[1])},
        "minimax_regret": minimax,
        "minimax_absolute": absolute,
        "dual_regret_lower_bound": float(dual),
        "shortcut_accuracy_at_least_favorable_prior": 0.5 + 0.4 * t_star,
        "frontier": [record(t) for t in np.linspace(0, 1, 21)],
        "report_patterns": [
            {
                "votes": votes.astype(int).tolist(),
                "m0": float(m0[i]),
                "m1": float(m1[i]),
                "q0": float(q0[i]),
                "q1": float(q1[i]),
                "minimax_regret_forecast": float(optimum[i]),
            }
            for i, votes in enumerate(x0[:, :, 1] > 0.5)
        ],
    }


if __name__ == "__main__":
    result = calculate()
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("endpoint_bayes_brier", "minimax_regret", "minimax_absolute")
            }
        )
    )

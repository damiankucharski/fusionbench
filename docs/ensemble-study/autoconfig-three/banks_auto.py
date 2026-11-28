"""Synthetic source banks, historical controls and a paired indistinguishability witness."""

from dataclasses import replace
from numbers import Integral
from pathlib import Path
from runpy import run_path

import numpy as np

from fusionbench.scenarios.grouped_rare_event import GroupedRareEvent

# Reuse the pinned mechanism study's public scenario integration and scoring rules.
PREVIOUS = run_path(
    str(Path(__file__).resolve().parents[1] / "mechanism-four" / "banks_study.py")
)
Bank = PREVIOUS["Bank"]
score = PREVIOUS["score"]
_digest = PREVIOUS["_digest"]
LEGACY = PREVIOUS["LEGACY"]


def _factorial(round_no, seeds):
    for old in PREVIOUS["_factorial"](round_no, seeds):
        ctx = old.context
        seed = ctx["seed"]
        knobs = {
            k: ctx[k]
            for k in (
                "heterogeneity",
                "common_error_share",
                "specialist_strength",
                "specialist_rate",
                "scene_error_share",
                "event_offset",
                "scene_size",
            )
        }
        # Use a separate stream for structural ordering and both data draws.
        order_seed = int(np.random.SeedSequence([seed, 317]).generate_state(1)[0])
        law = GroupedRareEvent(
            **knobs, provider_signal_slope=1.8, provider_order_seed=order_seed
        )
        x, y, fit = law.sample(len(old.y), np.random.default_rng([seed, 0]))
        ax, ay, audit = law.sample(len(old.audit_y), np.random.default_rng([seed, 1]))
        np.testing.assert_array_equal(y, old.y)
        np.testing.assert_array_equal(ay, old.audit_y)
        np.testing.assert_array_equal(audit["posterior"], old.q)
        np.testing.assert_array_equal(audit["specialist_region"], old.audit_regime)
        np.testing.assert_array_equal(
            PREVIOUS["_group_ids"](fit["scene_id"], 0), old.groups
        )
        np.testing.assert_array_equal(
            PREVIOUS["_group_ids"](audit["scene_id"], 1), old.audit_groups
        )
        # Provider role identities are part of the simulator audit only.
        context = {
            **ctx,
            "provider_signal_slope": 1.8,
            "provider_order_seed": order_seed,
            "fit_x_sha256": _digest(x),
            "audit_x_sha256": _digest(ax),
        }
        yield replace(old, x=x, audit_x=ax, context=context)


def _witness(round_no, seeds):
    names = [f"provider_{i}" for i in range(3)]
    for seed in seeds:
        draw_fit = np.random.default_rng([seed, 0])
        draw_test = np.random.default_rng([seed, 1])
        fit = draw_fit.uniform(0.05, 0.95, size=(128, 3))
        audit = draw_test.uniform(0.05, 0.95, size=(2048, 3))
        x, ax = np.stack((1 - fit, fit), axis=2), np.stack((1 - audit, audit), axis=2)
        y = (draw_fit.random(len(fit)) < fit[:, 0]).astype(np.int64)
        shared_uniform = draw_test.random(len(audit))
        qa, qb = audit[:, 0], audit.mean(1)
        bound = float(0.5 * np.mean((qa - qb) ** 2))
        for name, q in (("best_world", qa), ("mean_world", qb)):
            ay = (shared_uniform < q).astype(np.int64)
            b = LEGACY["_bank"](
                round_no, "nfl", name, seed, names, x, y, ax, audit_y=ay
            )
            b.q = q
            b.context.update(
                q_sha256=_digest(q),
                audit_uniform_sha256=_digest(shared_uniform),
                bayes_brier=float(2 * np.mean(q * (1 - q))),
                observed_bayes_brier=float(2 * np.mean((ay - q) ** 2)),
                common_rule_excess_bound=bound,
            )
            yield Bank(**vars(b))


def banks(round_no):
    """Iterate 216, 186 or 404 bank conditions in the three disjoint data blocks."""
    if (
        isinstance(round_no, bool)
        or not isinstance(round_no, Integral)
        or not 1 <= round_no <= 3
    ):
        raise ValueError("round_no must be an integer from 1 to 3")
    round_no = int(round_no)
    n = 16 if round_no == 3 else 8
    n_standard = {1: 4, 2: 3, 3: 6}[round_no]
    base = 610000 + 1000 * round_no
    yield from _factorial(round_no, range(base, base + n))
    yield from PREVIOUS["_clustered"](round_no, range(base + 100, base + 100 + n))
    for b in LEGACY["_controlled"](round_no, range(base + 200, base + 200 + n)):
        if b.context["family"] in {"weak", "strong"}:
            yield Bank(**vars(b))
    yield from PREVIOUS["_standard"](
        round_no, range(base + 400, base + 400 + n_standard)
    )
    if round_no == 3:
        yield from _witness(round_no, range(base + 800, base + 800 + n))

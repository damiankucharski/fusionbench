"""Synthetic-only banks for four rounds of paired mechanism experiments."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from numbers import Integral
from pathlib import Path
from runpy import run_path

import numpy as np

from fusionbench import generate_dataset, get_scenario
from fusionbench.scenarios.checkpoint_bank import generate_scene_pool
from fusionbench.scenarios.grouped_rare_event import GroupedRareEvent
from fusionbench.transfer import load_preset

PREVIOUS = run_path(
    str(Path(__file__).resolve().parents[1] / "full-data-five" / "banks_full.py")
)
LEGACY = PREVIOUS["LEGACY"]
_digest = PREVIOUS["_digest"]
GROUPED_CASES = {"grouped_rare_event", "shared_error_bloc", "rare_scene_specialist"}


@dataclass
class Bank(PREVIOUS["Bank"]):
    """The audit regime is a score-only field, like the conditional audit risk."""

    audit_regime: np.ndarray | None = None


def _group_ids(raw, split):
    """Give scenes disjoint integer identifiers across the two independent streams."""
    _, index = np.unique(raw, return_inverse=True)
    return 2 * index + split


def _factorial(round_no, seeds):
    for seed in seeds:
        anchor = None
        for h, c, s in product(range(2), repeat=3):
            levels = {
                "heterogeneity": (0.0, 0.65)[h],
                "common_error_share": (0.0, 0.65)[c],
                "specialist_strength": (0.0, 0.85)[s],
                "specialist_rate": 0.2,
                "scene_error_share": 0.85,
                "event_offset": 2.0,
                "scene_size": 8,
            }
            law = GroupedRareEvent(**levels)
            x, y, f = law.sample(128, np.random.default_rng([seed, 0]))
            ax, ay, a = law.sample(2048, np.random.default_rng([seed, 1]))
            groups = _group_ids(f["scene_id"], 0)
            audit_groups = _group_ids(a["scene_id"], 1)
            # Only predictions are allowed to change between cells of one draw.
            paired = (
                y,
                ay,
                groups,
                audit_groups,
                f["signal"],
                a["signal"],
                f["posterior"],
                a["posterior"],
                f["specialist_region"],
                a["specialist_region"],
            )
            if anchor is None:
                anchor = paired
            else:
                for original, current in zip(anchor, paired, strict=True):
                    np.testing.assert_array_equal(original, current)

            bank = LEGACY["_bank"](
                round_no,
                "factorial",
                f"h{h}c{c}s{s}",
                seed,
                law.model_ids(),
                x,
                y,
                ax,
                audit_y=ay,
            )
            q = np.asarray(a["posterior"], dtype=float)
            region = np.asarray(a["specialist_region"], dtype=bool)
            bank.q = q
            bank.context.update(
                levels,
                fit_groups_sha256=_digest(groups),
                audit_groups_sha256=_digest(audit_groups),
                q_sha256=_digest(q),
                audit_regime_sha256=_digest(region),
                n_fit_scenes=len(np.unique(groups)),
                n_audit_scenes=len(np.unique(audit_groups)),
                fit_regime_fraction=float(np.mean(f["specialist_region"])),
                audit_regime_fraction=float(np.mean(region)),
            )
            yield Bank(
                **vars(bank),
                groups=groups,
                audit_groups=audit_groups,
                audit_regime=region,
            )


def _clustered(round_no, seeds):
    for seed in seeds:
        audit = generate_scene_pool(
            seed,
            law="equal",
            n_fit_scenes=128,
            n_audit_scenes=256,
            scene_error_share=0.85,
            event_offset=2.0,
        )["I"]
        for bank in PREVIOUS["_clustered"](round_no, (seed,)):
            np.testing.assert_array_equal(bank.audit_y, audit.y)
            np.testing.assert_array_equal(bank.audit_x[:, :, 1], audit.P[:, :, 0])
            np.testing.assert_array_equal(bank.audit_groups, audit.scene)
            bank.q = audit.posterior
            bank.context["q_sha256"] = _digest(bank.q)
            yield Bank(**vars(bank))


def _standard(round_no, seeds):
    families = load_preset("standard-v6")["scenarios"]
    if len(families) != 30 or not GROUPED_CASES.issubset(families):
        raise ValueError("Expected all 30 registered standard-v6 scenarios")
    for family in families:
        for seed in seeds:
            ds = generate_dataset(family, seed=seed, n_per_split=512)
            bank = LEGACY["_bank"](
                round_no,
                "standard",
                family,
                seed,
                ds.model_ids,
                ds.fit_preds,
                ds.fit_y,
                ds.test_preds,
                audit_y=ds.test_y,
            )
            if family in GROUPED_CASES:
                evaluation = ds.meta["evaluation_groups"]
                groups = _group_ids(evaluation["fit"]["scene_id"], 0)
                audit_groups = _group_ids(evaluation["test"]["scene_id"], 1)
                regime = np.asarray(evaluation["test"]["specialist_region"], dtype=bool)
                # The latent event probability is not the observable Bayes oracle
                # p(y | provider reports); regenerate the public audit to retrieve it.
                rng = np.random.default_rng(seed)
                law = get_scenario(family, **ds.meta["parameters"])
                law.sample(512, rng)
                replay_x, replay_y, latent = law.sample(512, rng)
                np.testing.assert_array_equal(replay_x, ds.test_preds)
                np.testing.assert_array_equal(replay_y, ds.test_y)
                bank.q = latent["posterior"]
                bank.context.update(
                    q_sha256=_digest(bank.q),
                    audit_regime_sha256=_digest(regime),
                    fit_groups_sha256=_digest(groups),
                    audit_groups_sha256=_digest(audit_groups),
                )
                yield Bank(
                    **vars(bank),
                    groups=groups,
                    audit_groups=audit_groups,
                    audit_regime=regime,
                )
            else:
                yield Bank(**vars(bank))


def banks(round_no):
    if (
        isinstance(round_no, bool)
        or not isinstance(round_no, Integral)
        or not 1 <= round_no <= 4
    ):
        raise ValueError("round_no must be an integer from 1 to 4")
    n = 24 if round_no == 4 else 8
    n_suite = 4 if round_no == 4 else 2
    base = 491000 + 1000 * int(round_no)
    yield from _factorial(round_no, range(base, base + n))
    yield from _clustered(round_no, range(base + 100, base + 100 + n))
    for bank in LEGACY["_controlled"](round_no, range(base + 200, base + 200 + n)):
        if bank.context["family"] in {"weak", "strong"}:
            yield Bank(**vars(bank))
    yield from _standard(round_no, range(base + 400, base + 400 + n_suite))


def score(bank, p):
    """Full-class Brier, its label expectation where known, macro-F1 and binary AP."""
    scores = PREVIOUS["audit_scores"](bank, p)
    conditional = np.nan
    if bank.audit_y is None:
        # On finite populations audit_scores integrates both label outcomes and all input masses.
        conditional = scores["test_brier"]
    elif bank.q is not None:
        p = np.asarray(p, dtype=float)
        # Both class channels enter this sum, including any floating-point normalization error.
        loss_zero = np.square(p[:, 0] - 1) + np.square(p[:, 1])
        loss_one = np.square(p[:, 0]) + np.square(p[:, 1] - 1)
        conditional = float(np.mean((1 - bank.q) * loss_zero + bank.q * loss_one))
    return {
        "brier": scores["test_brier"],
        "expected_brier": conditional,
        "macro_f1": scores["test_macro_f1"],
        "ap": scores["test_ap"],
    }

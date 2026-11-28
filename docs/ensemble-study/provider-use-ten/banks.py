"""Fresh source banks, paired shortcut targets and a broken-routing control."""

from numbers import Integral
from pathlib import Path
from runpy import run_path
from types import SimpleNamespace

import numpy as np

from fusionbench import generate_dataset
from fusionbench.scenarios.source_environments import source_environment_bank
from fusionbench.transfer import load_preset

GROUPED = {"grouped_rare_event", "shared_error_bloc", "rare_scene_specialist"}
SOURCE_PROFILES = run_path(
    str(Path(__file__).resolve().parents[1] / "environment-ten" / "banks_env.py")
)["SOURCE_PROFILES"]
EXTRAS = {
    2: ("varied",),
    6: ("narrow",),
    7: ("small128",),
    8: ("homogeneous",),
    9: ("reversed",),
    10: ("varied", "narrow", "small128", "reversed", "homogeneous"),
}
TARGETS = {
    "s0": (0.5, 0),
    "s50": (0.5, 0.5),
    "s100": (0.5, 1),
    "specialist80": (0.8, 0.5),
}


def _observed_source(
    cohort, family, seed, x, y, groups, idx, test_x, test_y, selector=None
):
    # The same observable source proxy and scene subsampling used by environment-ten.
    p = x[idx]
    avg = p.mean(1)
    dis = np.square(p - avg[:, None, :]).sum(axis=2).mean(axis=1)
    conf = avg.max(axis=1)
    return SimpleNamespace(
        cohort=cohort,
        family=family,
        seed=seed,
        x=x,
        y=y,
        groups=groups,
        idx=idx,
        env=(dis > np.median(dis)).astype(int),
        ctx=(conf > np.median(conf)).astype(int),
        shortcut=None,
        specialist=None,
        selector=selector,
        worlds=[("test", test_x, test_y, None, None)],
    )


def _standard(seeds):
    names = load_preset("standard-v6")["scenarios"]
    if len(set(names)) != 30:
        raise ValueError(
            "The registered study expects 30 distinct standard-v6 scenarios"
        )
    for family in names:
        for seed in seeds:
            data = generate_dataset(family, seed=seed, n_per_split=512)
            idx = groups = np.arange(len(data.fit_y))
            if family in GROUPED:
                groups = np.asarray(data.meta["evaluation_groups"]["fit"]["scene_id"])
                rng = np.random.default_rng([seed, 217])
                idx = np.sort(
                    [rng.choice(np.flatnonzero(groups == g)) for g in np.unique(groups)]
                )
            yield _observed_source(
                "standard",
                family,
                seed,
                data.fit_preds,
                data.fit_y,
                groups,
                idx,
                data.test_preds,
                data.test_y,
            )


def _focus(r, seeds):
    for family in ("mixed", "case_mix", *EXTRAS.get(r, ())):
        region, shortcut = SOURCE_PROFILES[family]
        for seed in seeds:
            bank = source_environment_bank(
                region=region,
                shortcut=shortcut,
                seed=seed,
                n_rows=128 if family == "small128" else 512,
            )
            idx = np.arange(len(bank.y))
            worlds = []
            for world, knobs in TARGETS.items():
                target = bank.target(*knobs)
                worlds.append((world, target.x, None, target.mass, target.q))
            yield SimpleNamespace(
                cohort="focus",
                family=family,
                seed=seed,
                x=bank.x,
                y=bank.y,
                idx=idx,
                groups=idx,
                env=bank.env,
                ctx=bank.context,
                shortcut=bank.shortcut_index,
                specialist=bank.specialist_index,
                selector=None,
                worlds=worlds,
            )


def _permuted_selector(x, column, seed, split):
    # Shuffle the vector of class probabilities as a unit. Draw independent
    # permutations for source and target without using any outcomes or regions.
    perm = np.random.default_rng([seed, 691, split]).permutation(len(x))
    shifted = x.copy()
    shifted[:, column] = x[perm, column]
    return shifted


def _gate(seeds):
    for seed in seeds:
        data = generate_dataset("gated_experts", seed=seed, n_per_split=512)
        idx = np.arange(len(data.fit_y))
        col = data.model_ids.index("selector")
        for family, fit, test in (
            ("gated_original", data.fit_preds, data.test_preds),
            (
                "gated_shuffled",
                _permuted_selector(data.fit_preds, col, seed, 0),
                _permuted_selector(data.test_preds, col, seed, 1),
            ),
        ):
            yield _observed_source(
                "gate_control",
                family,
                seed,
                fit,
                data.fit_y,
                idx,
                idx,
                test,
                data.test_y,
                selector=col,
            )


def sources(r):
    """Yield a round's banks with full-source groups and idx-aligned env/ctx.

    Each world holds (name, reports, labels, population mass, posterior). Sampled
    targets have labels. Analytic targets provide the mass and posterior for
    the sixteen possible anonymous report patterns. All worlds share one fit.
    """
    if isinstance(r, bool) or not isinstance(r, Integral) or not 1 <= r <= 10:
        raise ValueError("r must be an integer from 1 to 10")
    base = 940000 + 1000 * int(r)
    standard_n = {9: 4, 10: 8}.get(r, 2)
    focus_n = {9: 12, 10: 24}.get(r, 6)
    yield from _standard(range(base + 400, base + 400 + standard_n))
    yield from _focus(r, range(base + 100, base + 100 + focus_n))
    if r in (3, 10):
        yield from _gate(range(base + 700, base + 700 + focus_n))

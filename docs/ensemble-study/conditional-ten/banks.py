"""Paired source environments and a scene-level screening sample on fresh seeds."""

from numbers import Integral
from pathlib import Path
from runpy import run_path
from types import SimpleNamespace

import numpy as np

from fusionbench.scenarios.source_environments import source_environment_bank

PREVIOUS = run_path(
    str(Path(__file__).resolve().parents[1] / "case-mix-ten" / "banks.py")
)
PROFILES = PREVIOUS["PROFILES"]
TARGETS = PREVIOUS["PREVIOUS"]["TARGETS"]
GROUPED = PREVIOUS["PREVIOUS"]["GROUPED"]
# The old generator already implements every requested profile except narrow.
# Map its profile schedule to this study and supply new independent data seeds.
SOURCE_ROUND = {1: 1, 2: 2, 3: 3, 4: 6, 5: 5, 6: 1, 7: 7, 8: 8, 9: 9, 10: 10}


def _focus(r, seeds):
    yield from PREVIOUS["_focus"](SOURCE_ROUND[r], seeds)
    if r in (6, 10):
        region, shortcut = PROFILES["narrow"]
        for seed in seeds:
            bank = source_environment_bank(
                region=region, shortcut=shortcut, seed=seed, n_rows=512
            )
            idx = np.arange(len(bank.y))
            targets = {
                name: bank.target(*settings) for name, settings in TARGETS.items()
            }
            yield SimpleNamespace(
                cohort="focus",
                family="narrow",
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
                worlds=[(name, t.x, None, t.mass, t.q) for name, t in targets.items()],
            )


def sources(r):
    """Yield the full fit data, fixed diagnostic indices and all matching test laws.

    Environment, context, labels and group vectors all refer to the full source.
    Use screen_idx on all diagnostic inputs together. In grouped standard
    scenarios it selects exactly one example per scene. Other banks use all rows.
    """
    if isinstance(r, bool) or not isinstance(r, Integral) or not 1 <= r <= 10:
        raise ValueError("r must be an integer from 1 to 10")
    base = 1140000 + 1000 * int(r)
    n = {9: 4, 10: 8}.get(r, 2)
    for bank in PREVIOUS["_standard"](range(base + 400, base + 400 + n)):
        bank.screen_idx = bank.idx
        if bank.family in GROUPED:
            rng = np.random.default_rng([bank.seed, 217])
            bank.screen_idx = np.sort(
                [
                    rng.choice(np.flatnonzero(bank.groups == g))
                    for g in np.unique(bank.groups)
                ]
            )
        yield bank
    n = {9: 12, 10: 24}.get(r, 6)
    for bank in _focus(r, range(base + 100, base + 100 + n)):
        bank.screen_idx = bank.idx
        yield bank

"""Fresh all-row standard banks and matched case-mix/shortcut source environments."""

from numbers import Integral
from pathlib import Path
from runpy import run_path
from types import SimpleNamespace

import numpy as np

from fusionbench import generate_dataset
from fusionbench.scenarios.source_environments import source_environment_bank
from fusionbench.transfer import load_preset

PREVIOUS = run_path(
    str(Path(__file__).resolve().parents[1] / "provider-use-ten" / "banks.py")
)
PROFILES = {**PREVIOUS["SOURCE_PROFILES"], "extreme": ((0.035, 0.965), (0.0, 1.0))}
EXTRAS = {
    2: ("varied",),
    3: ("reversed",),
    5: ("extreme",),
    6: ("small128",),
    7: ("noisy_context",),
    8: ("homogeneous",),
    9: ("reversed",),
    10: ("varied", "reversed", "extreme", "small128", "noisy_context", "homogeneous"),
}


def _standard(seeds):
    names = load_preset("standard-v6")["scenarios"]
    if len(names) != 30 or len(set(names)) != 30:
        raise ValueError("Expected all 30 standard-v6 scenarios")
    for family in names:
        for seed in seeds:
            data = generate_dataset(family, seed=seed, n_per_split=512)
            idx = np.arange(len(data.fit_y))
            groups = (
                np.asarray(data.meta["evaluation_groups"]["fit"]["scene_id"])
                if family in PREVIOUS["GROUPED"]
                else idx
            )
            # Always compute proxy medians on the complete provider bank.
            yield PREVIOUS["_observed_source"](
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
        region, shortcut = PROFILES[family]
        for seed in seeds:
            bank = source_environment_bank(
                region=region,
                shortcut=shortcut,
                seed=seed,
                n_rows=128 if family == "small128" else 512,
                context_noise=0.25 if family == "noisy_context" else 0,
            )
            idx = np.arange(len(bank.y))
            worlds = []
            for name, settings in PREVIOUS["TARGETS"].items():
                target = bank.target(*settings)
                worlds.append((name, target.x, None, target.mass, target.q))
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


def sources(r):
    """Yield one source per fit and one tuple per target distribution for scoring."""
    if isinstance(r, bool) or not isinstance(r, Integral) or not 1 <= r <= 10:
        raise ValueError("r must be an integer from 1 to 10")
    base = 1040000 + 1000 * int(r)
    n_standard = {9: 3, 10: 6}.get(r, 2)
    n_focus = {9: 10, 10: 20}.get(r, 6)
    yield from _standard(range(base + 400, base + 400 + n_standard))
    yield from _focus(r, range(base + 100, base + 100 + n_focus))

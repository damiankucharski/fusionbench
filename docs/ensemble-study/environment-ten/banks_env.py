"""Registered source profiles and fresh, paired exact targets for ten rounds."""

from numbers import Integral

from fusionbench.scenarios.source_environments import (
    SOURCE_ENVIRONMENT_PRESETS,
    source_environment_bank,
)

SOURCE_PROFILES = {
    **SOURCE_ENVIRONMENT_PRESETS,
    "four_env": ((0.15, 0.15, 0.85, 0.85), (0, 0, 1, 1)),
    "narrow": ((0.15, 0.85), (0.25, 0.75)),
    "small128": SOURCE_ENVIRONMENT_PRESETS["mixed"],
    "original_scale": SOURCE_ENVIRONMENT_PRESETS["mixed"],
    "noisy_context": SOURCE_ENVIRONMENT_PRESETS["mixed"],
    "reversed": ((0.85, 0.15), (0, 1)),
}
EXTRA_PROFILES = {
    5: ("four_env",),
    6: ("narrow",),
    7: ("small128",),
    8: ("original_scale", "noisy_context"),
    10: ("reversed", "narrow", "small128"),
}


def focus_banks(r):
    """Yield (profile, seed, bank, {target name: exact law}) for one round."""
    if isinstance(r, bool) or not isinstance(r, Integral) or not 1 <= r <= 10:
        raise ValueError("r must be an integer from 1 to 10")
    base = 840000 + 1000 * int(r) + 100
    repeats = 24 if r == 10 else 12 if r == 9 else 8
    worlds = {"s0": (0.5, 0), "s50": (0.5, 0.5), "s100": (0.5, 1)}
    if r in (3, 4, 9, 10):
        worlds["specialist80"] = (0.8, 0.5)
    if r == 10:
        worlds.update(s25=(0.5, 0.25), s75=(0.5, 0.75))
    for name in (*SOURCE_ENVIRONMENT_PRESETS, *EXTRA_PROFILES.get(r, ())):
        region, shortcut = SOURCE_PROFILES[name]
        for seed in range(base, base + repeats):
            bank = source_environment_bank(
                region=region,
                shortcut=shortcut,
                n_rows=128 if name == "small128" else 512,
                seed=seed,
                original_scale=name == "original_scale",
                context_noise=0.25 if name == "noisy_context" else 0.0,
            )
            yield (
                name,
                seed,
                bank,
                {target: bank.target(*settings) for target, settings in worlds.items()},
            )

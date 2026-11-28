"""Seeded scenario draws and the canonical target-domain parameter changes."""

import numpy as np

SHIFTS = {
    "two_pathways": {"accuracy": 0.78},
    "parity": {"accuracy": 0.78},
    "weakest_link": {"accuracy": 0.76},
    "correlated": {"corr": 0.1},
    "calibrated_suppressor": {"corr": 0.0},
    "polarity_fault": {"correlation": 0.0},
    "native_switch": {"mixing": 1.0},
    "native_close_switch": {"mixing": 1.0},
    "compositional_switch": {"corner_probability": 0.0},
}


def default_target_shift(name):
    """Return overrides for the canonical endpoint; a split-aware law can use its own test settings."""
    return {} if name in ("environment_shortcut", "cloned_shortcut") else SHIFTS.get(name)


def draw_block(scenario, seed, index, n, *, split=None):
    """Draw predictions, labels and generator metadata from a fixed RNG stream."""
    rng = np.random.default_rng([seed, index])
    return scenario.sample(n, rng) if split is None else scenario.sample_split(n, rng, split)

"""Fixed standard panel, continuity controls, and single-factor shortcut intervention."""

from pathlib import Path
from runpy import run_path
from types import SimpleNamespace

import numpy as np
from fusionbench.scenarios.rare_shortcut_noise import RareShortcutNoise

HERE = Path(__file__).resolve().parent
PREV = run_path(str(HERE.parent / "mechanism-four" / "banks_study.py"))
score = PREV["score"]


def _draws(r):
    base = 720000 + 1000 * r
    size = 10 if r == 10 else 5 if r == 9 else 3
    controls = 16 if r == 10 else 8
    yield from PREV["_standard"](r, range(base + 400, base + 400 + size))
    yield from PREV["_clustered"](r, range(base + 100, base + 100 + controls))
    for b in PREV["LEGACY"]["_controlled"](r, range(base + 200, base + 200 + controls)):
        if b.context["family"] in {"weak", "strong"}:
            yield b
    if r in (5, 10):
        for seed in range(base + 300, base + 300 + controls):
            for share in (1.0, 0.5, 0.0):
                law = RareShortcutNoise(
                    region_test=0.05, shortcut_test=share, provider_noise_test=0.0
                )
                x, y, _ = law.sample(512, np.random.default_rng(seed))
                audit, mass, q = law.support("test")
                name = f"shortcut_{round(100 * share):03d}"
                yield PREV["LEGACY"]["_bank"](
                    r,
                    "control",
                    name,
                    seed,
                    law.model_ids(),
                    x,
                    y,
                    audit,
                    mass=mass,
                    q=q,
                )


def source_subset(x, groups, seed):
    rng = np.random.default_rng([seed, 419])
    if groups is None:
        return np.sort(rng.permutation(len(x))[:128])
    unique, counts = np.unique(groups, return_counts=True)
    order = rng.permutation(len(unique))
    keep = unique[order[np.cumsum(counts[order]) <= 128]]
    idx = np.flatnonzero(np.isin(groups, keep))
    if len(idx) != 128:
        raise ValueError("The registered grouped scenarios require 128 complete rows")
    return idx


def banks(r):
    for bank in _draws(r):
        b = SimpleNamespace(**vars(bank))
        b.groups = getattr(b, "groups", None)
        b.cohort = "standard" if b.context["cohort"] == "standard" else "control"
        b.family, b.seed = b.context["family"], b.context["seed"]
        b.variant, b.optimizer_seed = "primary", int(r == 8)
        if b.cohort == "standard" and r == 7:
            large = SimpleNamespace(**vars(b))
            large.variant = "large"
            idx = source_subset(b.x, b.groups, b.seed)
            b.x, b.y = b.x[idx], b.y[idx]
            if b.groups is not None:
                b.groups = b.groups[idx]
            yield large
        if b.cohort == "standard" and r == 8:
            alternate = SimpleNamespace(**vars(b))
            alternate.variant, alternate.optimizer_seed = "seed0", 0
            yield alternate
        yield b

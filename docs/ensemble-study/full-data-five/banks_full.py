"""Full-data study banks. Only x, y and optional training-scene IDs enter fit."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from runpy import run_path

import numpy as np
from sklearn.metrics import average_precision_score, f1_score

from fusionbench.scenarios.checkpoint_bank import (
    generate_scene_pool,
    scene_sample_indices,
)

# Use the published generators, class ordering and split manifests from the previous five rounds.
LEGACY = run_path(
    str(Path(__file__).resolve().parents[1] / "stability-five" / "banks.py")
)
BaseBank = LEGACY["Bank"]


@dataclass
class Bank(BaseBank):
    groups: np.ndarray | None = None
    audit_groups: np.ndarray | None = None


def _digest(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _clustered(round_no, seeds):
    for seed in seeds:
        blocks = generate_scene_pool(
            seed,
            law="equal",
            n_fit_scenes=128,
            n_audit_scenes=256,
            scene_error_share=0.85,
            event_offset=2.0,
        )
        fit = blocks["F"]
        audit = blocks["I"]
        # Freeze public checkpoint zero for all providers without using any A labels.
        fit_p, audit_p = fit.P[:, :, 0], audit.P[:, :, 0]
        x = np.stack((1 - fit_p, fit_p), axis=-1)
        audit_x = np.stack((1 - audit_p, audit_p), axis=-1)
        if np.intersect1d(fit.scene, audit.scene).size:
            raise RuntimeError("Training and test scenes overlap")
        for n_scenes, rows_per_scene in ((16, 8), (128, 1)):
            idx = scene_sample_indices(n_scenes, rows_per_scene)
            groups = fit.scene[idx]
            count = np.unique(groups, return_counts=True)[1]
            assert len(count) == n_scenes and np.all(count == rows_per_scene)
            bank = LEGACY["_bank"](
                round_no,
                "clustered",
                f"scene{n_scenes}",
                seed,
                [f"provider_{j}" for j in range(x.shape[1])],
                x[idx],
                fit.y[idx],
                audit_x,
                audit_y=audit.y,
            )
            bank.context.update(
                n_fit_scenes=n_scenes,
                fit_rows_per_scene=rows_per_scene,
                n_audit_scenes=256,
                audit_rows_per_scene=8,
                scene_error_share=0.85,
                event_offset=2.0,
                fixed_checkpoint=0,
                fit_groups_sha256=_digest(groups),
                audit_groups_sha256=_digest(audit.scene),
            )
            yield Bank(**vars(bank), groups=groups, audit_groups=audit.scene)


def banks(round_no):
    """Yield 118 banks in screening or 284 in confirmation, using new preassigned draws."""
    if (
        isinstance(round_no, bool)
        or not isinstance(round_no, Integral)
        or not 1 <= round_no <= 5
    ):
        raise ValueError("round_no must be an integer from 1 to 5")
    base = 380000 + 1000 * int(round_no)
    n = 32 if round_no == 5 else 12
    for b in LEGACY["_controlled"](round_no, range(base, base + n)):
        yield Bank(**vars(b))
    yield from _clustered(round_no, range(base + 300, base + 300 + n))
    standard_n, digits_n = (4, 8) if round_no == 5 else (2, 4)
    for b in LEGACY["_standard"](round_no, range(base + 100, base + 100 + standard_n)):
        yield Bank(**vars(b))
    for b in LEGACY["_digits"](round_no, range(base + 200, base + 200 + digits_n)):
        yield Bank(**vars(b))


def audit_scores(bank, pred):
    """Return full Brier and macro-F1, plus positive-class average precision for binary banks."""
    pred = np.asarray(pred, dtype=float)
    n, classes = len(bank.audit_x), bank.x.shape[2]
    if pred.shape != (n, classes) or not np.all(np.isfinite(pred)):
        raise ValueError("Invalid prediction shape or nonfinite probabilities")
    if not np.allclose(pred.sum(1), 1, rtol=0, atol=1e-6) or np.any(
        (pred < -1e-8) | (pred > 1.0 + 1e-6)
    ):
        raise ValueError("Expected one probability distribution per observation")
    # np.argmax gives the negative class at the binary 0.5 tie.
    decision = pred.argmax(axis=1)
    target = bank.audit_y
    weights = None
    if target is None:
        pos = pred[:, 1]
        q = bank.q
        brier = float(2 * (bank.mass @ (q * (1 - pos) ** 2 + (1 - q) * pos**2)))
        # Expand both possible labels at each finite-support point. Threshold and ranking
        # metrics integrate joint masses here, including ties, without generating labels.
        target = np.tile([0, 1], n)
        weights = np.column_stack([bank.mass * (1 - q), bank.mass * q]).ravel()
        decision = np.repeat(decision, 2)
        positive = np.repeat(pos, 2)
    else:
        brier = float(np.mean(np.sum((pred - np.eye(classes)[target]) ** 2, axis=1)))
        positive = pred[:, 1]
    macro_f1 = f1_score(
        target,
        decision,
        labels=np.arange(classes),
        average="macro",
        zero_division=0,
        sample_weight=weights,
    )
    ap = (
        float(average_precision_score(target, positive, sample_weight=weights))
        if classes == 2
        else np.nan
    )
    return {"test_brier": brier, "test_macro_f1": float(macro_f1), "test_ap": ap}

"""Matched scene-level banks of anonymous, correlated binary checkpoints.

Scene signal S_g and row signal E_gr are independent standard Gaussians.
Set z_gr = 0.9 S_g + 0.8 E_gr - 1.1 and q_gr = sigmoid(1.8 z_gr). Draw
Y_gr as Bernoulli(q_gr). Under the equal law, provider m at latent checkpoint
time t reports the sigmoid of

    1.3 z_gr + 0.5 U_gm + 0.5 V_grm + 0.85 H_gmt + 0.5 J_grmt.

U,V are independent standard Gaussians. H and J are independent stationary
unit-variance AR(1) paths with lag-one correlation 0.65. Each path starts
from a standard Gaussian; later terms are 0.65 times the preceding value plus
sqrt(1 - 0.65**2) times an independent standard Gaussian innovation. All
noise families and label uniforms are independent of each other and of S,E.
Sharing S, U, and H across the eight rows of a scene creates within-scene
dependence. Distinct scenes are independent conditional on the bank's order.

Each provider gets one uniformly random permutation of eight latent times.
It is held fixed across all rows and blocks in the bank. Thus released
checkpoint IDs are exchangeable over the order randomization while the latent
chronology has stronger dependence for nearer checkpoints. Conditional on any
fixed order, every individual provider/checkpoint still has exactly the same
population joint distribution with Y. Since providers have independent noise
conditional on the signal, any fixed choice of one slot per provider has the
same joint population law on fresh scenes. The bank therefore isolates
selection among equally competent checkpoints in the equal law.

In the paired signal law, one anonymous provider reports q_gr at all eight
checkpoints; the other seven providers and all label draws are unchanged.
The provider is chosen by an independent stream (block=0, channel=1), once
per bank. All blocks use the same model and data parameters.

RNGs are NumPy Generator(PCG64(SeedSequence([seed, block_id, channel]))).
Block IDs 1, 2, 3, 4 denote A, B, I, F. Channels 0 through 6 draw S, E, Y's
uniform, U, V, H's innovations, and J's innovations respectively, in
row-major order. The independent (block=0, channel=0) stream draws one
permutation per provider in order. Each entry is the positive-class probability; the
last array axis indexes public checkpoint IDs. Nothing outside the
block passed to select_checkpoints contributes labels or predictions to that
choice.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from numpy.typing import NDArray

PROVIDERS = 8
CHECKPOINTS = 8
ROWS_PER_SCENE = 8
RHO = 0.65


@dataclass(frozen=True)
class Block:
    """P has shape [row, provider, checkpoint]; y and scene have one entry per row.

    Scene IDs are disjoint across blocks within each bank. Identify a scene
    across multiple banks by (bank seed, scene ID). posterior is the latent
    probability P(Y=1 | z) for auditing; in the signal law it is observed
    through one provider. In the equal law the providers are noisy views of z.
    """

    P: NDArray[np.float64]
    y: NDArray[np.int64]
    scene: NDArray[np.int64]
    posterior: NDArray[np.float64] | None = None


def _rng(seed, block, channel):
    return np.random.Generator(
        np.random.PCG64(np.random.SeedSequence([seed, block, channel]))
    )


def _time_noise(rng, shape):
    path = rng.standard_normal((*shape, CHECKPOINTS))
    scale = np.sqrt(1 - RHO**2)
    for t in range(1, CHECKPOINTS):
        path[..., t] = RHO * path[..., t - 1] + scale * path[..., t]
    return path


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _block(seed, block_id, n_scenes, offset, stride, order, signal_index):
    rows = ROWS_PER_SCENE
    s = _rng(seed, block_id, 0).standard_normal((n_scenes, 1))
    e = _rng(seed, block_id, 1).standard_normal((n_scenes, rows))
    z = 0.9 * s + 0.8 * e - 1.1
    posterior = _sigmoid(1.8 * z)
    y = _rng(seed, block_id, 2).random(z.shape) < posterior
    u = _rng(seed, block_id, 3).standard_normal((n_scenes, 1, PROVIDERS, 1))
    v = _rng(seed, block_id, 4).standard_normal((n_scenes, rows, PROVIDERS, 1))
    h = _time_noise(_rng(seed, block_id, 5), (n_scenes, 1, PROVIDERS))
    j = _time_noise(_rng(seed, block_id, 6), (n_scenes, rows, PROVIDERS))
    latent_logits = 1.3 * z[:, :, None, None] + 0.5 * u + 0.5 * v + 0.85 * h + 0.5 * j
    p = _sigmoid(np.take_along_axis(latent_logits, order[None, None, :, :], axis=-1))
    if signal_index is not None:
        p[:, :, signal_index, :] = posterior[:, :, None]
    scene = np.repeat(offset + stride * np.arange(n_scenes, dtype=np.int64), rows)
    return Block(
        p.reshape(-1, PROVIDERS, CHECKPOINTS),
        y.ravel().astype(np.int64),
        scene,
        posterior.ravel(),
    )


def _checked_seed(seed: int) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    return int(seed)


def signal_provider_index(seed: int) -> int:
    """Return the planted provider ID for scoring or auditing a signal bank."""
    return int(_rng(_checked_seed(seed), 0, 1).integers(PROVIDERS))


def _generate_blocks(seed, law, specs):
    seed = _checked_seed(seed)
    if law not in ("equal", "signal"):
        raise ValueError("law must be 'equal' or 'signal'")
    signal_index = signal_provider_index(seed) if law == "signal" else None
    rng = _rng(seed, 0, 0)
    order = np.stack([rng.permutation(CHECKPOINTS) for _ in range(PROVIDERS)])
    return {
        name: _block(seed, block_id, n_scenes, offset, stride, order, signal_index)
        for name, block_id, n_scenes, offset, stride in specs
    }


def generate_checkpoint_bank(seed: int, *, law: str = "equal") -> dict[str, Block]:
    """Generate 24 A, 24 B, and 256 independent I scenes, each with eight rows.

    The default equal law retains the historical draws. The signal law exposes
    the latent class probability q in every slot of one anonymous provider.
    Evaluate a binary fused prediction p using 2*(p-q)**2 for full-class Brier
    excess relative to the latent oracle, conditional on the generated inputs.
    Pass only frozen provider scores and labels into fusion fitting.
    """
    return _generate_blocks(
        seed, law, (("A", 1, 24, 0, 1), ("B", 2, 24, 24, 1), ("I", 3, 256, 48, 1))
    )


def _scene_count(value, name, maximum=None):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Integral)
        or value < 1
        or (maximum is not None and value > maximum)
    ):
        limit = f" at most {maximum}" if maximum is not None else ""
        raise ValueError(f"{name} must be a positive integer{limit}")
    return int(value)


def generate_scene_pool(
    seed: int,
    *,
    law: str = "equal",
    n_fit_scenes: int = 1536,
    n_audit_scenes: int = 512,
) -> dict[str, Block]:
    """Generate checkpoint-selection (A), fit (F) and audit (I) blocks.

    All blocks have eight ordered rows per independent scene. A contains 24
    scenes with IDs 0 through 23. F and I each contain a positive number of
    scenes. Scene g, counting from zero, has ID 24+2*g in F and 25+2*g in I.
    The RNG streams and scene IDs of each block stay fixed when either size changes.
    Increasing a block extends its existing prefix. A and I use the same
    checkpoint and label streams as generate_checkpoint_bank for the same
    seed; I has its own ID range in this layout.
    """
    n_fit = _scene_count(n_fit_scenes, "n_fit_scenes")
    n_audit = _scene_count(n_audit_scenes, "n_audit_scenes")
    return _generate_blocks(
        seed,
        law,
        (("A", 1, 24, 0, 1), ("F", 4, n_fit, 24, 2), ("I", 3, n_audit, 25, 2)),
    )


def scene_sample_indices(n_scenes: int, rows_per_scene: int) -> NDArray[np.int64]:
    """Select the first rows in each of the first n_scenes of an eight-row pool.

    For a shortened F pool, request at most the number of scenes it contains.
    Increasing either count preserves the previously selected set of rows.
    """
    n_scenes = _scene_count(n_scenes, "n_scenes")
    rows = _scene_count(rows_per_scene, "rows_per_scene", ROWS_PER_SCENE)
    starts = ROWS_PER_SCENE * np.arange(n_scenes, dtype=np.int64)
    return (starts[:, None] + np.arange(rows, dtype=np.int64)).ravel()


def select_checkpoints(block: Block, k: int) -> NDArray[np.int64]:
    """Return one slot per provider from the first k, selected by pooled F1.

    A positive prediction uses a strict score > 0.5. Pool confusion counts
    over all rows before taking the ratio, giving zero F1 for a zero
    denominator. Tied scores, including all-zero scores, keep the earliest
    public slot. Apply the returned indices unchanged on other blocks with
    other.P[:, np.arange(PROVIDERS), indices].
    """
    if (
        isinstance(k, (bool, np.bool_))
        or not isinstance(k, Integral)
        or not 1 <= k <= CHECKPOINTS
    ):
        raise ValueError(f"k must be an integer between 1 and {CHECKPOINTS}")
    p, y = block.P, block.y
    if (
        p.ndim != 3
        or p.shape[1:] != (PROVIDERS, CHECKPOINTS)
        or y.shape != (len(p),)
        or not len(y)
    ):
        raise ValueError(
            "Expected aligned, nonempty rows with eight providers and eight checkpoints"
        )
    p = p[:, :, :k]
    if not np.all((y == 0) | (y == 1)) or not np.all((p >= 0) & (p <= 1)):
        raise ValueError("Expected binary labels and finite probabilities in [0,1]")
    positive = p > 0.5
    true_positives = np.count_nonzero(positive & (y[:, None, None] == 1), axis=0)
    denominator = positive.sum(axis=0) + np.count_nonzero(y)
    f1 = np.divide(
        2 * true_positives,
        denominator,
        out=np.zeros(denominator.shape),
        where=denominator != 0,
    )
    return f1.argmax(axis=1).astype(np.int64, copy=False)

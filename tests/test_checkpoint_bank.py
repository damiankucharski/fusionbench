import numpy as np
import pytest

from fusionbench.scenarios.checkpoint_bank import (
    Block,
    generate_checkpoint_bank,
    select_checkpoints,
)


def test_shapes_disjoint_groups_and_determinism():
    blocks = generate_checkpoint_bank(7)
    repeat = generate_checkpoint_bank(np.int64(7))
    different = generate_checkpoint_bank(8)
    seen = set()
    for key, scenes in (("A", 24), ("B", 24), ("I", 256)):
        block = blocks[key]
        assert block.P.shape == (scenes * 8, 8, 8)
        assert block.y.shape == block.scene.shape == (scenes * 8,)
        assert block.P.dtype == np.float64
        assert set(np.unique(block.y)) <= {0, 1}
        assert ((block.P > 0) & (block.P < 1)).all()
        groups, counts = np.unique(block.scene, return_counts=True)
        assert len(groups) == scenes
        np.testing.assert_array_equal(counts, np.full(scenes, 8))
        assert seen.isdisjoint(groups)
        seen.update(groups)
        np.testing.assert_array_equal(block.P, repeat[key].P)
        np.testing.assert_array_equal(block.y, repeat[key].y)
        assert not np.array_equal(block.P, different[key].P)
    assert not np.array_equal(blocks["A"].P, blocks["B"].P)


def test_group_and_checkpoint_noise_matches_scalar_construction():
    # Independent reconstruction of a complete path for one row and provider.
    # Reveals nonstationary starts, missing scene effects and per-split reshuffling.
    seed, g, r, m = 12, 1, 3, 5
    order_rng = np.random.default_rng([seed, 0, 0])
    orders = [order_rng.permutation(8) for _ in range(8)]
    blocks = generate_checkpoint_bank(seed)
    for bid, name, ng in ((1, "A", 24), (2, "B", 24), (3, "I", 256)):

        def stream(channel, block_id=bid):
            return np.random.default_rng([seed, block_id, channel])

        def path(innov):
            result = [innov[0]]
            for value in innov[1:]:
                result.append(0.65 * result[-1] + np.sqrt(1 - 0.65**2) * value)
            return np.array(result)

        s = stream(0).standard_normal(ng)[g]
        e = stream(1).standard_normal((ng, 8))[g, r]
        z = 0.9 * s + 0.8 * e - 1.1
        u = stream(3).standard_normal((ng, 8))[g, m]
        v = stream(4).standard_normal((ng, 8, 8))[g, r, m]
        h = path(stream(5).standard_normal((ng, 8, 8))[g, m])
        j = path(stream(6).standard_normal((ng, 8, 8, 8))[g, r, m])
        logits = 1.3 * z + 0.5 * u + 0.5 * v + 0.85 * h + 0.5 * j
        row = 8 * g + r
        np.testing.assert_allclose(
            blocks[name].P[row, m], 1 / (1 + np.exp(-logits[orders[m]])), rtol=2e-15
        )
        u_label = stream(2).random((ng, 8))[g, r]
        assert blocks[name].y[row] == int(u_label < 1 / (1 + np.exp(-1.8 * z)))


def test_strict_threshold_f1_nested_prefix_ties_and_zero_denominators():
    p = np.zeros((4, 8, 8))
    y = np.array([1, 1, 0, 0])
    p[:, 0, 0] = [0.5, 0.5, 0, 0]
    p[:, 0, 1] = p[:, 0, 2] = [0.6, 0.2, 0.9, 0.4]
    p[:, 0, 3] = [0.8, 0.7, 0.4, 0.3]
    block = Block(p, y, np.arange(4))
    for k, best in [(1, 0), (2, 1), (3, 1), (4, 3), (8, 3)]:
        np.testing.assert_array_equal(select_checkpoints(block, k), [best] + [0] * 7)
    np.testing.assert_array_equal(
        select_checkpoints(Block(np.zeros_like(p), np.zeros_like(y), block.scene), 8),
        np.zeros(8),
    )
    # Columns outside a locked prefix never enter score comparison.
    p[:, :, 2:] = np.nan
    assert select_checkpoints(block, 2)[0] == 1
    with pytest.raises(ValueError, match="probabilities"):
        select_checkpoints(block, 3)


def test_select_pools_rows_before_computing_f1():
    y = np.array([1, 0, 0, 0, 1, 1, 1, 0])
    decisions = np.array([[1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1, 0, 1]])
    p = np.zeros((8, 8, 8))
    p[:, :, 0] = decisions[0, :, None]
    p[:, :, 1] = decisions[1, :, None]
    block = Block(p, y, np.repeat([0, 1], 4))
    # Pooled F1 is 2/5 vs 4/7. Per-scene F1 means are 1/2 vs 1/3.
    np.testing.assert_array_equal(select_checkpoints(block, 2), np.ones(8, dtype=int))


@pytest.mark.parametrize("bad", [-1, True, 2.5])
def test_bad_seed(bad):
    with pytest.raises(ValueError, match="seed"):
        generate_checkpoint_bank(bad)


@pytest.mark.parametrize("k", [0, 9, 1.5, True])
def test_bad_candidate_count(k):
    with pytest.raises(ValueError, match="integer"):
        select_checkpoints(
            Block(np.zeros((1, 8, 8)), np.zeros(1, dtype=int), np.zeros(1, dtype=int)),
            k,
        )

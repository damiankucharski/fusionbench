import numpy as np
import pytest

from fusionbench.eval.replacement_search import run_replacement_search
from fusionbench.eval.suite_point_gp import predict_tree

pytest.importorskip("okapi.replacement")


def fixture(seed, rows, providers=4):
    rng = np.random.default_rng(seed)
    positive = rng.uniform(0.05, 0.95, (rows, providers))
    x = np.stack((1.0 - positive, positive), axis=2)
    y = (rng.uniform(size=rows) < positive[:, 0]).astype(int)
    return x, y


def test_open_breeders_ignore_fresh_gate_values_and_account_for_each_slot():
    x, y = fixture(0, 32)
    blocks = [fixture(i + 20, 11) for i in range(6)]
    flipped = [(x, 1 - y) for x, y in blocks]
    variants = [
        ("ordinary", None),
        ("open_raw", blocks),
        ("open_kl", blocks),
        ("open_raw", flipped),
    ]
    results = [
        run_replacement_search(x, y, gates, seed=7, policy=policy, incumbent="mean")
        for policy, gates in variants
    ]
    paths = {r.path_digest for r in results}
    assert len(paths) == 1
    for r in results:
        assert len(r.initial_survivors) == 5
        assert r.initial_incumbent_id in r.initial_survivors
        assert r.incumbent_scan == 0
        assert r.discovery_evaluations == 5 + 72
        assert r.proposal_count == 72
        assert len({c.genotype for c in r.records}) < len(r.records)
        assert r.unique_providers == len(set(r.records[r.champion_id].provider_ids))
        assert all(
            len(g.proposals) == 12 and 0 < len(g.survivors) <= 8 for g in r.generations
        )
    assert sum(g.gate_comparisons for g in results[0].generations) == 0
    for r in results[1:]:
        assert sum(g.gate_comparisons for g in r.generations) == 72
        assert all(g.gate_rows == 11 and len(g.gains) == 12 for g in r.generations)
        last_incumbent = r.initial_incumbent_id
        for g in r.generations:
            assert g.incumbent_before == last_incumbent
            if g.promoted is not None:
                assert g.promoted in g.eligible
            last_incumbent = g.incumbent_after
        assert r.champion_id == last_incumbent


def test_strict_admission_excludes_uncertified_children_and_tracks_real_parent_ids():
    # On identical forecasts the paired gain and its row-conditional range are
    # zero for all offspring regardless of genotype. No gate may promote.
    x = np.full((16, 4, 2), 0.5)
    y = np.arange(16) % 2
    blocks = [(x.copy(), np.roll(y, i)) for i in range(6)]
    strict = run_replacement_search(
        x, y, blocks, seed=19, policy="strict_kl", incumbent="mean"
    )
    opened = run_replacement_search(
        x, y, blocks, seed=19, policy="open_kl", incumbent="mean"
    )
    for r in (strict, opened):
        assert r.champion_id == r.initial_incumbent_id
        assert not r.displaced
        prior = r.initial_survivors
        for g in r.generations:
            assert g.promoted is None and not g.eligible
            for i in g.proposals:
                c = r.records[i]
                assert len(c.parent_ids) == 2 and set(c.parent_ids).issubset(prior)
                assert c.confirmed is False
                assert c.has_unconfirmed_ancestor == any(
                    r.records[p].confirmed is False
                    or r.records[p].has_unconfirmed_ancestor
                    for p in c.parent_ids
                )
            prior = g.survivors
    assert all(
        set(g.survivors) == set(strict.initial_survivors) for g in strict.generations
    )
    assert not any(c.has_unconfirmed_ancestor for c in strict.records)
    assert any(c.has_unconfirmed_ancestor for c in opened.records)
    assert strict.path_digest != opened.path_digest


def test_multiclass_provider_scan_includes_singletons_outside_breeding_capacity():
    from okapi import Okapi

    # Read the native seeded subset first, then place the only exact singleton
    # at a held-out provider column to test separation of deployment and breeding.
    rng = np.random.default_rng(31)
    x = rng.dirichlet(np.ones(3), size=(24, 12))
    y = np.arange(24) % 3
    assert hasattr(Okapi, "propose_offspring")
    probe = run_replacement_search(x, y, seed=4, policy="ordinary")
    included = {probe.records[i].provider_ids[0] for i in probe.initial_survivors}
    target = next(i for i in range(12) if i not in included)
    x[:, target, :] = 0.05 + 0.85 * np.eye(3)[y]
    r = run_replacement_search(x, y, [(x, y)] * 6, seed=4, policy="strict_kl")
    assert r.initial_incumbent_id not in r.initial_survivors
    assert r.incumbent_scan == 12
    assert r.discovery_evaluations == 80
    assert len(r.records) == 81
    assert r.records[r.initial_incumbent_id].provider_ids == (target,)
    assert r.initial_survivors == probe.initial_survivors
    np.testing.assert_allclose(predict_tree(r.tree, x), x[:, target, :])


def test_replacement_requires_six_blocks():
    x, y = fixture(28, 12)
    with pytest.raises(ValueError, match="Six fresh"):
        run_replacement_search(x, y, seed=3, policy="open_raw")
    with pytest.raises(ValueError, match="complete blocks"):
        run_replacement_search(x, y, [(x, y)] * 5 + [None], seed=3, policy="open_kl")
    with pytest.raises(ValueError, match="Expected 6"):
        run_replacement_search(x, y, [(x, y)], seed=3, policy="strict_kl")

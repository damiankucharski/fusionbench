import numpy as np
import pytest

from fusionbench.eval import seed_geometry_focal as experiment
from fusionbench.eval.lattice_signature import lattice_signature, provider_signatures
from fusionbench.eval.seed_geometry import SeedPlan, geometry_trees


def test_prescribed_geometry_templates_match_only_their_corresponding_law():
    pytest.importorskip("okapi")
    from okapi.globals import use_postprocessing_function

    k = 4
    x, masks, shift = experiment.support(k)
    names = tuple(f"p{i}.npy" for i in range(k))
    role = np.arange(k)
    providers = {name: x[:, j : j + 1] for j, name in enumerate(names)}
    plan = SeedPlan(names, "min")
    branch, chain = (
        geometry_trees(providers, [plan], shape)[0] for shape in ("branch", "chain")
    )
    with use_postprocessing_function(np.asarray):
        for tree, law in ((branch, "branch"), (chain, "chain")):
            expected = experiment.law_probability(x, role, law)
            assert tree.nodes_count == 7
            np.testing.assert_array_equal(
                tree.predict(clear_cache=True).ravel(), expected
            )
            sig = lattice_signature(tree, provider_signatures(names))
            assert sig == experiment.target_signature(k, role, law)
            np.testing.assert_array_equal(
                experiment.signature_values(sig, k, masks), expected
            )
    assert (
        experiment.law_probability(x, role, "branch")
        != experiment.law_probability(x, role, "chain")
    ).any()
    assert experiment.target_signature(k, role, "additive") is None
    assert abs(shift.sum() - 1) < 1e-12


def test_laws_share_raw_features_but_unknown_roles_can_ignore_decoys():
    roles, splits = experiment.data_for(8, 17, "chain")
    again, additive = experiment.data_for(8, 17, "additive")
    np.testing.assert_array_equal(roles, again)
    for split in ("fit", "validation", "iid", "shift"):
        np.testing.assert_array_equal(splits[split][0], additive[split][0])
    assert len(set(roles)) == 4
    p = splits["fit"][0].copy()
    changed = p.copy()
    for j in set(range(8)) - set(roles):
        changed[:, j] = 0.99
    np.testing.assert_array_equal(
        experiment.law_probability(p, roles, "chain"),
        experiment.law_probability(changed, roles, "chain"),
    )


def test_validation_labels_affect_selection_but_never_native_trajectory(tmp_path):
    pytest.importorskip("okapi")
    _, ds = experiment.data_for(4, 3, "chain")
    xf, yf = ds["fit"]
    xv, yv = ds["validation"]
    paths = [tmp_path / f"p{i}.npy" for i in range(4)]
    gt = tmp_path / "y.npy"
    np.save(gt, yf[:, None])
    for i, path in enumerate(paths):
        np.save(path, xf[:, i : i + 1])
    a, archive, first, initial = experiment.search(
        paths, gt, xf, yf, xv, yv, 12, "branch", budgets=(50, 90)
    )
    b, _, _, _ = experiment.search(
        paths, gt, xf, yf, xv, ~yv, 12, "branch", budgets=(50, 90)
    )
    assert set(a) == {34, 50, 90}
    assert len(first) == len(initial) == 34
    for n in a:
        assert a[n]["trajectory_sha256"] == b[n]["trajectory_sha256"]
        assert a[n]["fit"].signature == b[n]["fit"].signature
        candidates = [c for c in archive.values() if c.first_call <= n]
        assert a[n]["validation"] == min(
            candidates, key=lambda c: (c.val, c.fit, c.first_call)
        )
        assert a[n]["fit"] == min(candidates, key=lambda c: (c.fit, c.first_call))

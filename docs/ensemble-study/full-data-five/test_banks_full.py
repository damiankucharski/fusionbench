"""Check data separation, unit of replication and population scoring."""

from types import SimpleNamespace

import numpy as np
import pytest
from banks_full import _clustered, audit_scores


def test_scene_pair_has_matched_rows_and_disjoint_groups():
    few, many = list(_clustered(1, [102]))
    assert len(few.x) == len(many.x) == 128
    np.testing.assert_array_equal(
        np.unique(few.groups, return_counts=True)[1], [8] * 16
    )
    np.testing.assert_array_equal(
        np.unique(many.groups, return_counts=True)[1], [1] * 128
    )
    for bank in (few, many):
        assert not np.intersect1d(bank.groups, bank.audit_groups).size
    np.testing.assert_array_equal(few.x[::8], many.x[:16])
    np.testing.assert_array_equal(few.y[::8], many.y[:16])
    np.testing.assert_array_equal(few.audit_x, many.audit_x)
    np.testing.assert_array_equal(few.audit_y, many.audit_y)
    np.testing.assert_array_equal(few.audit_groups, many.audit_groups)


def test_exact_binary_scores_match_replication_with_ties():
    pred = np.array([[0.65, 0.35], [0.5, 0.5], [0.25, 0.75], [0.25, 0.75]])
    exact = SimpleNamespace(
        x=pred[:, None, :],
        audit_x=pred[:, None, :],
        audit_y=None,
        q=np.array([0.25, 0.75, 0.5, 1.0]),
        mass=np.full(4, 0.25),
    )
    y = np.concatenate([np.arange(4) < pos for pos in [1, 3, 2, 4]]).astype(int)
    p = np.repeat(pred, 4, axis=0)
    replicated = SimpleNamespace(x=p[:, None, :], audit_x=p[:, None, :], audit_y=y)
    assert audit_scores(exact, pred) == pytest.approx(
        audit_scores(replicated, p), abs=1e-14
    )

import numpy as np
import pytest

from fusionbench.eval.validation_curse import audit_menus


def test_menus_are_nested_and_future_outcomes_cannot_change_choice():
    val = [[0.2, 0.3, 0.1], [0.2, 0.01, 0.9]]
    audit = [[0.23, 0.35, 0.32]]
    iid = [[0.45, 0.37, 0.31]]
    shift = [[0.2, 0.19, 0.18]]
    rows = audit_menus(
        val,
        audit,
        iid,
        shift,
        menu_sizes=[1, 2, 3],
        validation_sizes=[1, 2],
        population_risk=[0.3, 0.3, 0.6],
    )
    assert [r["winner"] for r in rows] == [0, 0, 2, 0, 1, 1]
    assert len({r["audit_anchor"] for r in rows}) == 1
    r = rows[2]
    assert r["source_selection_regret"] == pytest.approx(0.3)
    assert r["optimism"] == pytest.approx(0.22)
    assert r["extra_optimism"] == pytest.approx(0.19)
    assert r["source_menu_penalty"] == 0 and not r["source_best_hit"]
    assert rows[-1]["source_best_hit"]
    reverse = audit_menus(
        val, shift, iid, audit, menu_sizes=[1, 2, 3], validation_sizes=[1, 2]
    )
    assert [r["winner"] for r in rows] == [r["winner"] for r in reverse]
    assert all("source_selection_regret" not in r for r in reverse)


def test_tie_resolution_and_separate_menu_opportunity():
    data = np.array([[0.3, 0.3, 0.3 - 1e-14]])
    rows = audit_menus(
        data,
        data,
        data,
        data,
        menu_sizes=(1, 3),
        validation_sizes=(1,),
        tie_atol=1e-12,
        population_risk=[0.6, 0.55, 0.45],
    )
    assert [r["winner"] for r in rows] == [0, 0]
    assert rows[0]["source_selection_regret"] == 0 and rows[0][
        "source_menu_penalty"
    ] == pytest.approx(0.15)
    assert (
        rows[1]["source_selection_regret"] == pytest.approx(0.15)
        and rows[1]["source_menu_penalty"] == 0
    )
    with pytest.raises(ValueError, match="menu sizes"):
        audit_menus(data, data, data, data, menu_sizes=(4,), validation_sizes=(1,))
    with pytest.raises(ValueError, match="finite"):
        audit_menus(
            np.full((2, 3), np.nan),
            data,
            data,
            data,
            menu_sizes=(2,),
            validation_sizes=(1,),
        )
    with pytest.raises(ValueError, match="same candidate order"):
        audit_menus(
            data, data[:, :2], data, data, menu_sizes=(2,), validation_sizes=(1,)
        )

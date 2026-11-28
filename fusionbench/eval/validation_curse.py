"""Measure selection optimism and finite-menu regret on disjoint prediction banks."""

from numbers import Integral

import numpy as np


def _prefixes(values, bound, name):
    values = tuple(values)
    if not values or any(
        isinstance(x, (bool, np.bool_))
        or not isinstance(x, Integral)
        or not 1 <= x <= bound
        for x in values
    ):
        raise ValueError(
            f"{name} must be nonempty integer prefixes between 1 and {bound}"
        )
    if len(set(values)) != len(values):
        raise ValueError(f"Duplicate {name}")
    return values


def _loss_table(rows, width=None):
    a = np.asarray(rows, dtype=np.float64)
    if a.ndim != 2 or min(a.shape) < 1 or (width is not None and a.shape[1] != width):
        raise ValueError(
            "Each split needs a nonempty row-by-candidate matrix in the same candidate order"
        )
    if not np.isfinite(a).all():
        raise ValueError("Row losses must be finite")
    return a


def audit_menus(
    validation,
    audit,
    iid,
    shifted,
    *,
    menu_sizes,
    validation_sizes,
    population_risk=None,
    shifted_risk=None,
    tie_atol=0.0,
):
    """Select on validation only, then evaluate each frozen winner on three other banks.

    Inputs are [row, candidate] matrices of losses (smaller is better) in the
    same candidate order. Candidate zero is the fixed reference in every menu.
    Supply row-disjoint samples with a fresh clean audit and IID test; the shifted
    sample may follow a different law. The function cannot determine split
    independence from losses. A known population-risk vector makes finite-menu
    regret exact. Without it, all estimated optimism measures still use only
    fresh scores, and no empirical test winner is substituted for an oracle.

    Prefixes select the first rows or candidates. A nonnegative tie_atol,
    measured in *mean loss* units, resolves near ties in original candidate order.
    """
    val = _loss_table(validation)
    menus = _prefixes(menu_sizes, val.shape[1], "menu sizes")
    sizes = _prefixes(validation_sizes, len(val), "validation sizes")
    if not np.isfinite(tie_atol) or tie_atol < 0:
        raise ValueError("tie_atol must be finite and nonnegative")
    splits = {
        key: _loss_table(arr, val.shape[1]).mean(0)
        for key, arr in (("audit", audit), ("iid", iid), ("shift", shifted))
    }
    risks = {}
    for key, arr in (("source", population_risk), ("shift", shifted_risk)):
        if arr is not None:
            arr = np.asarray(arr, dtype=float)
            if arr.shape != (val.shape[1],) or not np.isfinite(arr).all():
                raise ValueError(
                    "Each supplied population risk needs one finite value per candidate"
                )
            risks[key] = arr
    records = []
    for n in sizes:
        validation_means = val[:n].mean(axis=0)
        for k in menus:
            losses = validation_means[:k]
            index = int(np.flatnonzero(losses <= losses.min() + tie_atol)[0])
            row = {
                "validation_rows": int(n),
                "menu_size": int(k),
                "winner": index,
                "validation": float(losses[index]),
                "validation_anchor": float(losses[0]),
            }
            for split, values in splits.items():
                row[split] = float(values[index])
                row[split + "_anchor"] = float(values[0])
            row["optimism"] = row["audit"] - row["validation"]
            row["extra_optimism"] = row["optimism"] - (
                row["audit_anchor"] - row["validation_anchor"]
            )
            for key, values in risks.items():
                best = values[:k].min()
                row[key + "_risk"] = float(values[index])
                row[key + "_selection_regret"] = float(values[index] - best)
                row[key + "_menu_penalty"] = float(best - values.min())
                row[key + "_best_hit"] = bool(values[index] == best)
            if "source" in risks:
                row["exact_optimism"] = row["source_risk"] - row["validation"]
                row["exact_extra_optimism"] = row["exact_optimism"] - (
                    risks["source"][0] - row["validation_anchor"]
                )
            records.append(row)
    return records

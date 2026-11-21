"""Post-hoc identification of empirical lattice minimizers on the frozen 72 focal cells."""

from __future__ import annotations

import argparse
import itertools
from collections import deque
from fractions import Fraction
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from fusionbench.eval.lattice_brier_oracle import fit_lattice_brier
from fusionbench.eval.precision_lattice import exact_family
from fusionbench.eval.seed_geometry_focal import (
    LEVELS,
    TICKS,
    data_for,
    integer_brier,
    law_probability,
    signature_values,
    support,
    target_signature,
    threshold_masks,
)


def reachable(residual, start, reverse):
    reached = {start}
    queue = deque([start])
    while queue:
        current = queue.popleft()
        for other in (
            residual.predecessors(current) if reverse else residual.successors(current)
        ):
            u, v = (other, current) if reverse else (current, other)
            arc = residual[u][v]
            if arc["flow"] < arc["capacity"] and other not in reached:
                reached.add(other)
                queue.append(other)
    return reached


def integer_extremes(x, y):
    # Squared loss telescopes at the three level boundaries. Selecting the
    # Boolean upper set pays its signed boundary increments on the observed rows.
    k = x.shape[1]
    size = 1 << k
    ticks = np.rint(x * 20).astype(int)
    target = y.astype(int) * 20
    unary = np.zeros(size, dtype=int)
    for lo, hi in itertools.pairwise(TICKS):
        active = ((ticks >= hi) * (1 << np.arange(k))).sum(axis=1)
        np.add.at(unary, active, (hi - target) ** 2 - (lo - target) ** 2)
    graph = nx.DiGraph()
    source, sink = size, size + 1
    infinity = int(np.abs(unary).sum()) + 1

    def connect(u, v, capacity):
        graph.add_edge(
            u, v, capacity=capacity + graph.get_edge_data(u, v, {}).get("capacity", 0)
        )

    for mask, cost in enumerate(unary):
        if cost:
            u, v = (mask, sink) if cost > 0 else (source, mask)
            connect(u, v, abs(int(cost)))
        for i in range(k):
            if not mask & 1 << i:
                connect(mask, mask | 1 << i, infinity)
    connect(source, size - 1, infinity)
    connect(0, sink, infinity)
    flow = nx.algorithms.flow.preflow_push(graph, source, sink)
    must_include = reachable(flow, source, False)
    must_exclude = reachable(flow, sink, True)
    lo = np.array([i in must_include for i in range(size)])
    hi = np.array([i not in must_exclude for i in range(size)])
    best = int(
        np.sum((TICKS[0] - target) ** 2)
        + np.minimum(0, unary).sum()
        + flow.graph["flow_value"]
    )
    return lo, hi, best


def to_signature(bits):
    return sum(1 << i for i in np.flatnonzero(bits).tolist())


def _monotone(bits, k):
    masks = np.arange(1 << k)
    return all(np.all(bits[masks] <= bits[masks | (1 << j)]) for j in range(k))


def float_objective(p, y):
    ratios = [float(v).as_integer_ratio() for v in LEVELS]
    scale = max(b for a, b in ratios)
    weighted = 0
    for level, (a, b) in zip(LEVELS, ratios, strict=True):
        rows = p == level
        value = a * (scale // b)
        weighted += (
            int(np.sum(rows & y)) * (value - scale) ** 2
            + int(np.sum(rows & ~y)) * value**2
        )
    assert np.isin(p, LEVELS).all()
    return Fraction(weighted, len(y) * scale**2)


def verify_four_extremes(x, y, endpoints, raw):
    rules, _ = exact_family()
    digits = np.searchsorted(LEVELS, x)
    columns = digits @ (4 ** np.arange(3, -1, -1))
    corners = (((np.arange(16)[:, None] >> np.arange(4)) & 1) * 3) @ (
        4 ** np.arange(3, -1, -1)
    )
    tables = rules[:, corners] > 0.5
    predictions = rules[:, columns]
    for key, fn in (
        ("integer_levels", integer_brier),
        ("binary64_inputs", float_objective),
    ):
        scores = [fn(row, y) for row in predictions]
        minimum = min(scores)
        winners = tables[[v == minimum for v in scores]]
        lo, hi = endpoints[key]
        assert np.array_equal(winners.all(axis=0), lo) and np.array_equal(
            winners.any(axis=0), hi
        )
        if key == "binary64_inputs":
            assert minimum == raw.minimum_loss


def inspect_cell(k, seed, law, reference):
    roles, splits = data_for(k, seed, law)
    xf, y = splits["fit"]
    support_x, masks, shift = support(k)
    q = law_probability(support_x, roles, law)
    qfit = law_probability(xf, roles, law)
    target = target_signature(k, roles, law)
    fit_masks = threshold_masks(xf)
    ilo, ihi, best = integer_extremes(xf, y)
    base = fit_lattice_brier(xf, y)
    assert abs(float(base.minimum_loss) - best / (400 * len(y))) < 1e-13
    assert best == reference.cut_optimal_units
    assert hex(to_signature(base.lower)) == reference.cut_rule_hex
    assert np.all(ilo <= base.lower) and np.all(base.upper <= ihi)
    if k == 4:
        verify_four_extremes(
            xf,
            y,
            {"integer_levels": (ilo, ihi), "binary64_inputs": (base.lower, base.upper)},
            base,
        )
    target_is_b64_erm = bool(
        target is not None and float_objective(qfit, y) == base.minimum_loss
    )
    target_units80 = int(
        np.sum((np.rint(80 * qfit).astype(int) - 80 * y.astype(int)) ** 2)
    )
    matches = target_units80 == 16 * best
    result = []
    for metric, lo, hi in (
        ("integer_levels", ilo, ihi),
        ("binary64_inputs", base.lower, base.upper),
    ):
        assert not lo[0] and not hi[0] and lo[-1] and hi[-1] and np.all(lo <= hi)
        assert _monotone(lo, k) and _monotone(hi, k)
        signatures = [to_signature(z) for z in (lo, hi)]
        fit_l, fit_h = [signature_values(i, k, fit_masks) for i in signatures]
        assert integer_brier(fit_l, y) == integer_brier(fit_h, y) == best
        l, h = [signature_values(i, k, masks) for i in signatures]
        assert np.all(l <= h) and np.all(fit_l <= fit_h)
        gap = h - l
        fit_gap = fit_h - fit_l
        minimum_pointwise_excess = (q - np.clip(q, l, h)) ** 2
        target_inside = (
            bool(np.all((l <= q) & (q <= h))) if target is not None else None
        )
        if matches and target is not None and metric == "integer_levels":
            assert target_inside
        fields = {
            "law": law,
            "width": k,
            "data_seed": seed,
            "cut_convention": metric,
            "cut_optimal_units": best,
            "lower_signature": hex(signatures[0]),
            "upper_signature": hex(signatures[1]),
            "upper_lower_different": bool(np.any(lo != hi)),
            "truth_corners_different": int(np.sum(lo != hi)),
            "fit_rows_different": int(np.sum(fit_gap != 0)),
            "fit_mean_squared_gap": np.mean(fit_gap**2),
            "iid_mass_different": np.mean(gap != 0),
            "shift_mass_different": shift @ (gap != 0),
            "iid_mean_squared_gap": np.mean(gap**2),
            "shift_mean_squared_gap": shift @ (gap**2),
            "integer_and_float_extremes_agree": bool(
                np.array_equal(ilo, base.lower) and np.array_equal(ihi, base.upper)
            ),
            "target_in_lattice": target is not None,
            "true_fit_score_equals_cut": matches,
            "true_is_integer_erm": bool(matches and target is not None),
            "true_is_binary64_erm": target_is_b64_erm,
            "any_erm_iid_excess_lower_bound": minimum_pointwise_excess.mean(),
            "any_erm_shift_excess_lower_bound": shift @ minimum_pointwise_excess,
            "true_between_pointwise": target_inside,
            "true_integer_erm_comparison_available": target is not None,
            "true_fit_brier": target_units80 / (6400 * len(y)),
            "true_fit_minus_exact_erm": (target_units80 - 16 * best) / (6400 * len(y)),
        }
        for name, fit_p, p in (
            ("lower", fit_l, l),
            ("upper", fit_h, h),
            ("midpoint_ensemble", (fit_l + fit_h) / 2, (l + h) / 2),
        ):
            err = (p - q) ** 2
            obs = np.mean((fit_p - y) ** 2)
            if name == "midpoint_ensemble":
                assert (
                    abs(
                        obs
                        - (best / (400 * len(y)) - fields["fit_mean_squared_gap"] / 4)
                    )
                    < 1e-13
                )
            result.append(
                {
                    **fields,
                    "candidate": name,
                    "fit_brier": obs,
                    "iid_excess": err.mean(),
                    "shift_excess": shift @ err,
                    "iid_risk": np.mean(err + q * (1 - q)),
                    "shift_risk": shift @ (err + q * (1 - q)),
                }
            )
    return result


def run(folder):
    raw = pd.read_csv(folder / "focal-runs.csv.gz", float_precision="round_trip")
    refs = raw[raw.method.eq("exact_cut")]
    assert len(refs) == 72
    table = pd.DataFrame(
        r
        for row in refs.itertuples()
        for r in inspect_cell(row.width, row.data_seed, row.law, row)
    )
    table.to_csv(folder / "posthoc-cut.csv.gz", index=False)
    cols = [
        "upper_lower_different",
        "truth_corners_different",
        "fit_rows_different",
        "fit_mean_squared_gap",
        "iid_mass_different",
        "shift_mass_different",
        "iid_mean_squared_gap",
        "shift_mean_squared_gap",
        "integer_and_float_extremes_agree",
        "true_is_integer_erm",
        "true_is_binary64_erm",
        "any_erm_iid_excess_lower_bound",
        "any_erm_shift_excess_lower_bound",
        "true_fit_brier",
        "true_fit_minus_exact_erm",
        "fit_brier",
        "iid_risk",
        "shift_risk",
        "iid_excess",
        "shift_excess",
    ]
    table.groupby(["law", "width", "cut_convention", "candidate"])[
        cols
    ].mean().reset_index().to_csv(folder / "posthoc-cut-means.csv", index=False)
    keys = ["law", "width", "data_seed"]
    bounds = table[
        table.candidate.eq("lower") & table.cut_convention.eq("integer_levels")
    ][keys + ["any_erm_iid_excess_lower_bound", "any_erm_shift_excess_lower_bound"]]
    searches = raw[raw.checkpoint.eq("1200") & raw.selector.eq("validation")]
    paired = (
        searches.groupby(keys + ["method"])[["iid_excess", "shift_excess"]]
        .mean()
        .reset_index()
        .merge(bounds, on=keys, validate="many_to_one")
    )
    for domain in ("iid", "shift"):
        paired[domain + "_below_bound"] = (
            paired[domain + "_excess"]
            < paired["any_erm_" + domain + "_excess_lower_bound"]
        )
    paired.to_csv(folder / "posthoc-cut-bounds.csv", index=False)
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, default=Path(__file__).resolve().parent)
    run(parser.parse_args().folder)

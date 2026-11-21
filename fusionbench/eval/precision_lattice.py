"""Native OKAPI selection precision with exact four-provider Brier optima."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations, product
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from scipy.stats import t
from threadpoolctl import threadpool_limits

from fusionbench.eval.native_switch import serialize
from fusionbench.eval.search_graded import (
    IID,
    PROBA,
    SCORES,
    SHIFT,
    brier,
    conditional,
    counts,
    risk,
    sample,
)
from fusionbench.eval.search_regret import Counts
from fusionbench.methods.averaging import TopN
from fusionbench.methods.stacking import StackingLogit

BANKS = tuple(range(6200, 6212))
STARTS = (709, 719)
BUDGETS = (100, 400)


@lru_cache(maxsize=1)
def exact_family():
    """Reconstruct the canonical rule order also used by search_graded.lattice.

    Minimal true conjunctions form an antichain. Four inputs have width six,
    and omitting empty terms and the empty antichain excludes the two constants.
    """
    terms = sorted(c for k in range(1, 5) for c in combinations(range(4), k))
    masks = {term: sum(1 << i for i in term) for term in terms}
    programs = [
        program
        for k in range(1, 7)
        for program in combinations(terms, k)
        if all(
            (masks[a] & masks[b]) not in (masks[a], masks[b])
            for a, b in combinations(program, 2)
        )
    ]
    programs.sort(key=lambda p: (sum(map(len, p)), p))
    conjunction = {term: SCORES[:, term].min(axis=1) for term in terms}
    rules = np.stack(
        [
            np.maximum.reduce([conjunction[term] for term in program])
            for program in programs
        ]
    )
    lookup = {v.tobytes(): i for i, v in enumerate(rules)}
    if len(rules) != 166 or len(lookup) != 166:
        raise AssertionError(
            "The four-input nonconstant lattice must have 166 distinct rules"
        )
    return rules, lookup


def integer_losses(rules, observations):
    ticks = np.rint(20 * np.asarray(rules)).astype(np.int64)
    return (20 - ticks) ** 2 @ observations.pos + ticks**2 @ observations.neg


def _pair_count(values):
    return sum(n * (n - 1) // 2 for n in Counter(values).values())


@dataclass(frozen=True)
class Banks:
    law: str
    seed: int
    order: np.ndarray
    eta: np.ndarray
    fit_cells: np.ndarray
    fit_labels: np.ndarray
    validation: Counts
    iid: Counts
    shift: Counts


def draw_bank(law, seed):
    order = np.random.default_rng([seed, 903]).permutation(4)
    eta = conditional(law, order)
    cells, labels = sample(seed, 0, 256, IID, eta)
    return Banks(
        law,
        seed,
        order,
        eta,
        cells,
        labels,
        counts(*sample(seed, 1, 256, IID, eta)),
        counts(*sample(seed, 2, 2048, IID, eta)),
        counts(*sample(seed, 3, 2048, SHIFT, eta)),
    )


def selection_score(loss_units, n, compressed):
    fitness = 1.0 - loss_units / (400 * n)
    return 1.0 - (1.0 - fitness) * 2**-12 if compressed else fitness


def archive_metrics(indices, fit_units, fit_patterns, selection_grid):
    raw_ties = _pair_count(fit_units[indices])
    fit_equivalent = _pair_count(fit_patterns[i] for i in indices)
    grid_ties = _pair_count(selection_grid[indices])
    return {
        "unique_rules": len(indices),
        "unique_fit_vectors": len({fit_patterns[i] for i in indices}),
        "unique_raw_losses": len(set(fit_units[indices])),
        "unique_selection_scores": len(set(selection_grid[indices])),
        "pairs_same_fit_vector": fit_equivalent,
        "pairs_equal_loss_distinct_fit_vectors": raw_ties - fit_equivalent,
        "pairs_merged_only_by_grid": grid_ties - raw_ties,
    }


def native_search(paths, gt, fit_units, n, seed, decimals, compressed, budgets):
    """Stop inside the final native fitness call, retaining chronological visit order."""
    from okapi import Okapi
    from okapi.globals import use_postprocessing_function
    from okapi.operators import MAX, MIN

    _, lookup = exact_family()
    archive = {}
    raw_digest = hashlib.sha256()
    offered_digest = hashlib.sha256()
    grid_digest = hashlib.sha256()
    snapshots = {}
    scored = 0
    tree_batch = iter(())
    scores = np.asarray(selection_score(fit_units, n, compressed))
    grid = np.round(scores, decimals) if decimals is not None else scores
    physical_end = max(budgets)

    class StopAtBudget(Exception):
        pass

    class Observed(Okapi):
        def _calculate_fitnesses(self, trees=None):
            nonlocal tree_batch
            trees = self.population if trees is None else trees
            tree_batch = iter(trees)
            return super()._calculate_fitnesses(trees)

    def objective(values, _labels):
        nonlocal scored
        tree = next(tree_batch)
        rule = lookup[np.asarray(values).ravel().tobytes()]
        if rule not in archive:
            archive[rule] = (
                int(tree.nodes_count),
                json.dumps(serialize(tree.root), separators=(",", ":")),
            )
        raw_digest.update((repr(tree.structural_key()) + "\n").encode())
        raw_digest.update(struct.pack("<HI", rule, fit_units[rule]))
        offered_digest.update(struct.pack("<d", scores[rule]))
        grid_digest.update(struct.pack("<d", grid[rule]))
        scored += 1
        if scored in budgets:
            snapshots[scored] = (
                archive.copy(),
                raw_digest.hexdigest(),
                offered_digest.hexdigest(),
                grid_digest.hexdigest(),
            )
        if scored == physical_end:
            raise StopAtBudget
        return scores[rule]

    with use_postprocessing_function(np.asarray):
        gp = Observed(
            paths,
            gt,
            4,
            2,
            2,
            backend="numpy",
            seed=seed,
            allowed_ops=(MIN, MAX),
            minimize_node_count=True,
            objective_functions=(objective,),
            operator_point_mutation=False,
            reuse_survivor_fitness=False,
            selection_fitness_decimals=decimals,
        )
        if len(gp.population) != 4 or any(
            tree.nodes_count != 1 for tree in gp.population
        ):
            raise AssertionError("Expected four native singleton starting trees")
        try:
            while scored < physical_end:
                gp.run_iteration()
        except StopAtBudget:
            pass
    if set(snapshots) != set(budgets):
        raise AssertionError("Missing a native fitness-call checkpoint")
    return snapshots, grid


def panel(law, data_seed, *, search_seeds=STARTS, budgets=BUDGETS):
    from loguru import logger

    logger.disable("okapi")
    bank = draw_bank(law, data_seed)
    rules, _ = exact_family()
    vr = integer_losses(rules, bank.validation)
    iid_risk = risk(rules, bank.eta, IID)
    shift_risk = risk(rules, bank.eta, SHIFT)
    common = {
        "law": law,
        "data_seed": data_seed,
        "role_order": "".join(map(str, bank.order)),
        "iid_family_risk": iid_risk.min(),
        "shift_family_risk": shift_risk.min(),
        "iid_bayes_risk": risk(bank.eta, bank.eta, IID),
        "shift_bayes_risk": risk(bank.eta, bank.eta, SHIFT),
    }
    rows = []
    with TemporaryDirectory() as temporary:
        tmp = Path(temporary)
        paths = [tmp / f"p{i}.npy" for i in range(4)]
        for i, path in enumerate(paths):
            np.save(path, SCORES[:, i : i + 1])
        gt = tmp / "labels.npy"

        def evaluate_size(n):
            cells, labels = bank.fit_cells[:n], bank.fit_labels[:n]
            fit_obs = counts(cells, labels)
            # The native tree predicts every feature cell once. The sole objective callback
            # uses the multiplicities and observed labels of fitting specimens only.
            np.save(
                gt,
                fit_obs.pos[:, None]
                / np.maximum((fit_obs.pos + fit_obs.neg)[:, None], 1),
            )
            fit_units = integer_losses(rules, fit_obs)
            minimum = int(fit_units.min())
            fit_patterns = [v.tobytes() for v in rules[:, np.unique(cells)]]
            fit_meta = {
                **common,
                "n_fit": n,
                "family_fit_units": minimum,
                "family_erm_ties": int((fit_units == minimum).sum()),
                "family_erm_rule_ids": json.dumps(
                    np.flatnonzero(fit_units == minimum).tolist()
                ),
                "family_iid_regret_at_erm": iid_risk[np.argmin(fit_units)]
                - iid_risk.min(),
            }

            def save(pred, method, selector, *, rule=None, metadata=None):
                loss = (
                    brier(pred, fit_obs)
                    if rule is None
                    else fit_units[rule] / (400 * n)
                )
                pi = risk(pred, bank.eta, IID)
                ps = risk(pred, bank.eta, SHIFT)
                on_fit = pred[cells] - bank.eta[cells]
                on_labels = labels - bank.eta[cells]
                projected = float(np.mean(on_fit**2))
                label_term = float(-2 * np.mean(on_fit * on_labels))
                noise_energy = float(np.mean(on_labels**2))
                if abs(loss - (projected + label_term + noise_energy)) > 1e-12:
                    raise AssertionError("Observed Brier decomposition failed")
                rows.append(
                    {
                        **fit_meta,
                        "method": method,
                        "selector": selector,
                        "rule": rule,
                        "fit_units": fit_units[rule] if rule is not None else None,
                        "fit_brier": loss,
                        "fit_minus_family_erm": loss - minimum / (400 * n),
                        "fit_hit": int(fit_units[rule] == minimum)
                        if rule is not None
                        else None,
                        "validation_brier": brier(pred, bank.validation),
                        "iid_test_brier": brier(pred, bank.iid),
                        "shift_test_brier": brier(pred, bank.shift),
                        "iid_risk": pi,
                        "shift_risk": ps,
                        "iid_minus_family": pi - common["iid_family_risk"],
                        "shift_minus_family": ps - common["shift_family_risk"],
                        "fit_input_excess": projected,
                        "fit_label_cross_term": label_term,
                        "fit_generator_noise": noise_energy,
                        "iid_minus_fit_input_excess": np.dot(
                            IID, (pred - bank.eta) ** 2
                        )
                        - projected,
                        "iid_excess_bayes": np.dot(IID, (pred - bank.eta) ** 2),
                        "shift_excess_bayes": np.dot(SHIFT, (pred - bank.eta) ** 2),
                        "prediction_sha256": hashlib.sha256(
                            np.asarray(pred, dtype="<f8").tobytes()
                        ).hexdigest(),
                        **(metadata or {}),
                    }
                )

            best = int(np.argmin(fit_units))
            save(rules[best], "exact_erm", "fit", rule=best)
            save(SCORES.mean(axis=1), "mean", "fixed")
            train = PROBA[cells]
            y = labels.astype(int)
            top = TopN("brier").fit(train, y, task="binary", n_classes=2)
            save(
                top.predict(PROBA)[:, 1],
                "top_n",
                "fit",
                metadata={
                    "n_selected": top.n_selected_,
                    "ranking": json.dumps(top.ranking_.tolist()),
                    "prefix_scores": json.dumps(top.prefix_scores_.tolist()),
                },
            )
            stack = StackingLogit().fit(train, y, task="binary", n_classes=2)
            save(stack.predict(PROBA)[:, 1], "stack_logit", "fit")

            modes = [(3, False), (6, False), (None, False)]
            if law == "matched" and n == 256:
                modes.extend([(3, True), (None, True)])
            for gp_seed, (decimals, compressed) in product(search_seeds, modes):
                snaps, grid = native_search(
                    paths, gt, fit_units, n, gp_seed, decimals, compressed, budgets
                )
                for calls, (archive, raw_sha, offered_sha, grid_sha) in snaps.items():
                    ids = np.array(list(archive))
                    fit_id = min(archive, key=lambda i: fit_units[i])
                    val_id = min(archive, key=lambda i: (vr[i], fit_units[i]))
                    diagnostics = archive_metrics(ids, fit_units, fit_patterns, grid)
                    for selector, i in (("fit", fit_id), ("validation", val_id)):
                        save(
                            rules[i],
                            "gp",
                            selector,
                            rule=i,
                            metadata={
                                "gp_seed": gp_seed,
                                "budget": calls,
                                "units": "compressed" if compressed else "nominal",
                                "precision": "full"
                                if decimals is None
                                else f"round{decimals}",
                                "nodes_at_first_visit": archive[i][0],
                                "program": archive[i][1],
                                "raw_prefix_sha256": raw_sha,
                                "offered_prefix_sha256": offered_sha,
                                "selection_prefix_sha256": grid_sha,
                                **diagnostics,
                            },
                        )

        for n in (32, 256):
            evaluate_size(n)
    return rows


METRICS = (
    "fit_hit",
    "fit_brier",
    "fit_minus_family_erm",
    "validation_brier",
    "iid_test_brier",
    "shift_test_brier",
    "iid_risk",
    "shift_risk",
    "iid_minus_family",
    "shift_minus_family",
    "iid_excess_bayes",
    "shift_excess_bayes",
    "unique_rules",
    "unique_raw_losses",
    "unique_selection_scores",
    "pairs_same_fit_vector",
    "pairs_equal_loss_distinct_fit_vectors",
    "pairs_merged_only_by_grid",
    "fit_input_excess",
    "fit_label_cross_term",
    "iid_minus_fit_input_excess",
)


def summarize(raw):
    keys = ["law", "n_fit", "method", "selector", "units", "precision", "budget"]
    per_bank = raw.groupby(keys + ["data_seed"], dropna=False)[list(METRICS)].mean()
    means = (
        per_bank.groupby(level=list(range(len(keys))), dropna=False)
        .mean()
        .reset_index()
    )
    gp = raw.loc[raw.method.eq("gp")].copy()
    gp["arm"] = gp.units + "_" + gp.precision
    pairs = (
        ("nominal_full", "nominal_round3"),
        ("nominal_round6", "nominal_round3"),
        ("compressed_round3", "nominal_round3"),
        ("compressed_full", "nominal_full"),
        ("compressed_full", "compressed_round3"),
    )
    endpoint_keys = ["law", "n_fit", "selector", "budget"]
    results = []
    for endpoint, block in gp.groupby(endpoint_keys):
        table = block.set_index(["data_seed", "gp_seed", "arm"])[list(METRICS)]
        arms = table.index.get_level_values("arm")
        for a, b in pairs:
            if a not in arms or b not in arms:
                continue
            delta = (
                (table.xs(a, level="arm") - table.xs(b, level="arm"))
                .groupby("data_seed")
                .mean()
            )
            for metric in METRICS:
                values = delta[metric]
                estimate = values.mean()
                half = (
                    t.ppf(0.975, len(values) - 1) * values.sem()
                    if len(values) > 1
                    else np.nan
                )
                results.append(
                    {
                        **dict(zip(endpoint_keys, endpoint)),
                        "left": a,
                        "right": b,
                        "metric": metric,
                        "data_banks": len(values),
                        "mean_difference": estimate,
                        "ci_low": estimate - half,
                        "ci_high": estimate + half,
                    }
                )
    return means, pd.DataFrame(results)


def verify_precision_invariants(raw):
    gp = raw.loc[raw.method.eq("gp") & raw.selector.eq("fit")].copy()
    gp["arm"] = gp.units + "_" + gp.precision
    idx = ["law", "n_fit", "data_seed", "gp_seed", "budget"]
    sha = gp.pivot(index=idx, columns="arm", values="raw_prefix_sha256")
    if not sha.nominal_full.eq(sha.nominal_round6).all():
        raise AssertionError(
            "Round-to-six changed an exactly separated fitness ordering"
        )
    compressed = sha.compressed_full.dropna()
    if not compressed.eq(sha.nominal_full.loc[compressed.index]).all():
        raise AssertionError(
            "Monotone rescaling without quantization changed the search prefix"
        )
    if (
        not gp.loc[gp.arm.eq("compressed_round3"), "unique_selection_scores"]
        .eq(1)
        .all()
    ):
        raise AssertionError(
            "Compressed three-decimal selection failed its all-tied control"
        )
    return {
        "round6_full_paired_prefixes": len(sha),
        "compressed_full_nominal_full_paired_prefixes": len(compressed),
        "compressed_round3_single_score_prefixes": int(
            gp.arm.eq("compressed_round3").sum()
        ),
    }


def _job(pair):
    with threadpool_limits(limits=1):
        return panel(*pair)


def run(out, workers=1):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = list(product(("matched", "additive"), BANKS))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        raw = pd.DataFrame(row for block in pool.map(_job, jobs) for row in block)
    checks = verify_precision_invariants(raw)
    means, paired = summarize(raw)
    raw.to_csv(out / "lattice.csv.gz", index=False)
    means.to_csv(out / "lattice-means.csv", index=False)
    paired.to_csv(out / "lattice-paired.csv", index=False)
    (out / "lattice-controls.json").write_text(json.dumps(checks, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    run(args.out, args.workers)

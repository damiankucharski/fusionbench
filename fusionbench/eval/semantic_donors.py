"""Trace actual native MIN/MAX individuals, their achieved rules and counterfactual donors."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import struct
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from functools import lru_cache
from importlib import import_module
from itertools import combinations, product
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

from fusionbench.eval.lattice_signature import lattice_signature, provider_signatures
from fusionbench.eval.native_switch import serialize
from fusionbench.eval.precision_lattice import draw_bank, exact_family, integer_losses
from fusionbench.eval.search_graded import (
    IID,
    PROBA,
    SCORES,
    SHIFT,
    brier,
    counts,
    risk,
)
from fusionbench.methods.averaging import TopN
from fusionbench.methods.stacking import StackingLogit

NAMES = tuple(f"p{i}.npy" for i in range(4))
FOCAL = (1, 5, 10, 15)
SEEDS = (809, 821)
BANKS = range(6600, 6616)


@lru_cache(maxsize=1)
def rule_signatures():
    """Map canonical four-level rules to their 16-bit Boolean-corner certificates."""
    rules, _ = exact_family()
    bits = ((np.arange(16)[:, None] >> np.arange(4)) & 1).astype(bool)
    corners = np.where(bits, SCORES.max(), SCORES.min())
    positions = {tuple(row): index for index, row in enumerate(SCORES)}
    ids = [positions[tuple(corner)] for corner in corners]
    masks = tuple(
        sum(1 << i for i in np.flatnonzero(r[ids] > 0.5).tolist()) for r in rules
    )
    if len(set(masks)) != 166:
        raise AssertionError(
            "Corner certificate failed to distinguish the complete lattice"
        )
    return masks


@lru_cache(maxsize=166)
def noncanalizing(signature):
    """Whether no individual input is necessary and none is sufficient."""
    for bit in provider_signatures(NAMES).values():
        if signature & bit == signature or signature | bit == signature:
            return False
    return True


def root_joins(signatures, targets):
    """Number of pairs of distinct target-free semantic roots yielding any target."""
    remaining = set(signatures) - set(targets)
    unions = intersections = 0
    for a, b in combinations(remaining, 2):
        unions += int((a | b) in targets)
        intersections += int((a & b) in targets)
    return {"union": unions, "intersection": intersections}


def _goals(bank, fit_units):
    rules, _ = exact_family()
    risks = risk(rules, bank.eta, IID)
    # Isolate arithmetic reorderings of the same population risk. The smallest
    # canonical index of the equal-risk group is the single additive target.
    tied = np.flatnonzero(np.isclose(risks, risks.min(), rtol=0, atol=1e-14)).tolist()
    primary = tied[0]
    if bank.law == "matched" and not np.array_equal(rules[primary], bank.eta):
        raise AssertionError(
            "Matched posterior did not resolve to its exact lattice rule"
        )
    return {
        "population_rule": primary,
        "population_optimal_rules": tied,
        "erm_rules": np.flatnonzero(fit_units == fit_units.min()).tolist(),
    }


def trace_search(bank, gp_seed, *, budget=400, focal_generations=FOCAL):
    """Run unchanged native variation; trace ordered genotypes and the exact scored prefix.

    The first generation starts after singleton scoring, immediately before
    crossover. A generation touching the score budget can have proposed children
    beyond the prefix. Survivors are recorded only if native selection completed.
    """
    from loguru import logger
    from okapi import Okapi
    from okapi.globals import use_postprocessing_function
    from okapi.operators import MAX, MIN

    if not isinstance(budget, int) or budget < 1:
        raise ValueError("budget must be a positive integer")
    if any(g < 1 for g in focal_generations):
        raise ValueError("focal generations must be positive")
    logger.disable("okapi")
    final_focal = max(focal_generations, default=0)
    _, lookup = exact_family()
    signatures = rule_signatures()
    by_mask = {mask: i for i, mask in enumerate(signatures)}
    providers = provider_signatures(NAMES)
    obs = counts(bank.fit_cells, bank.fit_labels)
    units = integer_losses(exact_family()[0], obs)
    n = int(obs.size)
    goals = _goals(bank, units)
    targets = {"population": {goals["population_rule"]}, "erm": set(goals["erm_rules"])}
    target_bits = {k: {signatures[r] for r in rs} for k, rs in targets.items()}
    stages = ("parents", "crossover", "mutation", "postmutation", "scored", "survivors")
    first = {k: dict.fromkeys(stages) for k in targets}
    programs, rule_ids, key_ids = [], [], {}
    generations, initial, archived = [], [], []
    seen = set()
    raw_digest = hashlib.sha256()
    generation = actual_calls = 0
    current = None

    def identify(tree):
        key = tree.structural_key()
        if key not in key_ids:
            key_ids[key] = len(programs)
            programs.append(serialize(tree.root))
            rule_ids.append(by_mask[lattice_signature(tree, providers)])
        return key_ids[key]

    def mark(ids, stage, gen):
        actual = {rule_ids[i] for i in ids}
        for label, required in targets.items():
            if first[label][stage] is None and actual & required:
                first[label][stage] = gen

    iterator = iter(())

    def objective(values, _):
        nonlocal actual_calls
        tree = next(iterator)
        i = identify(tree)
        rule = lookup[np.asarray(values).ravel().tobytes()]
        if rule != rule_ids[i]:
            raise AssertionError(
                "Corner signature and native four-level prediction differ"
            )
        actual_calls += 1
        seen.add(rule)
        (initial if current is None else current["scored"]).append(i)
        if actual_calls <= budget:
            archived.append(i)
            raw_digest.update((repr(tree.structural_key()) + "\n").encode())
            raw_digest.update(struct.pack("<HI", rule, int(units[rule])))
            mark([i], "scored", 0 if current is None else generation)
        return 1.0 - units[rule] / (400 * n)

    class Enough(Exception):
        pass

    class Observed(Okapi):
        def _perform_crossovers(self, fitnesses):
            nonlocal current
            pids = [identify(t) for t in self.population]
            root_bits = [signatures[rule_ids[i]] for i in pids]
            archive_bits = {signatures[r] for r in seen}
            current = {
                "generation": generation,
                "calls_before": actual_calls,
                "parents": pids,
                "parent_fitness": np.asarray(fitnesses)[:, 0].tolist(),
                "parent_mutation_chance": [
                    float(t.mutation_chance) for t in self.population
                ],
                "past_rules": sorted(seen),
                "crossover": [],
                "mutation": [],
                "scored": [],
                "survivors": None,
                "complete": False,
                "noncanalizing_live": int(sum(noncanalizing(b) for b in root_bits)),
                "noncanalizing_past": int(sum(noncanalizing(b) for b in archive_bits)),
                "joins": {
                    label: {
                        "live": root_joins(root_bits, bits),
                        "past": root_joins(archive_bits, bits),
                    }
                    for label, bits in target_bits.items()
                },
            }
            generations.append(current)
            if actual_calls < budget:
                mark(pids, "parents", generation)
            if actual_calls >= budget and generation >= final_focal:
                raise Enough
            module = import_module("okapi.okapi")
            native = module.crossover
            by_identity = {id(t): i for i, t in enumerate(self.population)}
            current["sampled_parent_indices"] = []

            def observe_pair(left, right, **kwargs):
                current["sampled_parent_indices"].append(
                    [by_identity[id(left)], by_identity[id(right)]]
                )
                return native(left, right, **kwargs)

            # The research driver runs one search at a time per process. Observe
            # the two references only after the actual native tournaments sampled them.
            with patch.object(module, "crossover", observe_pair):
                count = super()._perform_crossovers(fitnesses)
            current["crossover"] = [identify(t) for t in self.additional_population]
            if actual_calls < budget:
                mark(current["crossover"], "crossover", generation)
            return count

        def _mutate_additional_population(self):
            before = len(self.additional_population)
            count = super()._mutate_additional_population()
            current["mutation"] = [
                identify(t) for t in self.additional_population[before:]
            ]
            if actual_calls < budget:
                mark(current["mutation"], "mutation", generation)
                mark(
                    current["crossover"] + current["mutation"],
                    "postmutation",
                    generation,
                )
            return count

        def _calculate_fitnesses(self, trees=None):
            nonlocal iterator
            cohort = self.population if trees is None else trees
            iterator = iter(cohort)

            # Native _calculate_fitnesses invokes the sole objective for each tree
            # in the unchanged order. Only the final incomplete cohort is shortened.
            must_stop = generation >= final_focal
            left = budget - actual_calls
            if must_stop and left < len(cohort):
                super()._calculate_fitnesses(cohort[: max(0, left)])
                raise Enough
            result = super()._calculate_fitnesses(cohort)
            if current is None and must_stop and actual_calls >= budget:
                raise Enough
            return result

    with TemporaryDirectory() as temp, use_postprocessing_function(np.asarray):
        temp = Path(temp)
        paths = [temp / name for name in NAMES]
        for i, path in enumerate(paths):
            np.save(path, SCORES[:, i : i + 1])
        gt = temp / "labels.npy"
        np.save(gt, obs.pos[:, None] / np.maximum((obs.pos + obs.neg)[:, None], 1))
        gp = Observed(
            paths,
            gt,
            4,
            2,
            2,
            backend="numpy",
            seed=gp_seed,
            allowed_ops=(MIN, MAX),
            minimize_node_count=True,
            objective_functions=(objective,),
            selection_fitness_decimals=3,
            operator_point_mutation=False,
            reuse_survivor_fitness=False,
        )
        try:
            while actual_calls < budget or generation < final_focal:
                generation += 1
                gp.run_iteration()
                current["survivors"] = [identify(t) for t in gp.population]
                current["complete"] = True
                if actual_calls <= budget:
                    mark(current["survivors"], "survivors", generation)
        except Enough:
            pass
    if len(archived) != budget:
        raise AssertionError("Incomplete scored prefix")
    return {
        "law": bank.law,
        "data_seed": int(bank.seed),
        "gp_seed": int(gp_seed),
        "calls": len(archived),
        "real_calls": actual_calls,
        "raw_digest": raw_digest.hexdigest(),
        "programs": programs,
        "rule_ids": rule_ids,
        "goals": goals,
        "initial_scores": initial,
        "archive": archived,
        "generations": generations,
        "stages_first_hit": first,
    }


def stage_rows(trace):
    """Enumerate observed stage membership and actual productive crossover pairs."""
    ids = trace["rule_ids"]
    common = {k: trace[k] for k in ("law", "data_seed", "gp_seed")}
    groups = {
        "population": {trace["goals"]["population_rule"]},
        "erm": set(trace["goals"]["erm_rules"]),
    }
    rows = []
    for g in trace["generations"]:
        for label, targets in groups.items():

            def hits(indices, targets=targets):
                return sum(ids[i] in targets for i in indices)

            productive = 0
            for j, (l, r) in enumerate(g.get("sampled_parent_indices", ())):
                if not hits([g["parents"][l], g["parents"][r]]) and hits(
                    g["crossover"][2 * j : 2 * j + 2]
                ):
                    productive += 1
            row = {
                **common,
                "generation": g["generation"],
                "calls_before": g["calls_before"],
                "target": label,
                "productive_crossover_pairs": productive,
                "actual_crossover_pairs": len(g.get("sampled_parent_indices", ())),
                "parents_hits": hits(g["parents"]),
                "crossover_hits": hits(g["crossover"]),
                "mutant_hits": hits(g["mutation"]),
                "postmutation_hits": hits(g["crossover"] + g["mutation"]),
                "scored_hits": hits(g["scored"]),
                "survivor_hits": None
                if g["survivors"] is None
                else hits(g["survivors"]),
                "complete": g["complete"],
                "scored": len(g["scored"]),
                "noncanalizing_live": g["noncanalizing_live"],
                "noncanalizing_past": g["noncanalizing_past"],
            }
            for pool, detail in g["joins"][label].items():
                for op, v in detail.items():
                    row[f"{pool}_roots_{op}"] = v
            rows.append(row)
    return rows


def focal_rows(trace):
    from fusionbench.eval.donor_crossover import analyze_population, prepare_tree

    sigs = rule_signatures()
    rules = trace["rule_ids"]
    programs = trace["programs"]
    common = {k: trace[k] for k in ("law", "data_seed", "gp_seed")}
    groups = {
        "population": {sigs[trace["goals"]["population_rule"]]},
        "erm": {sigs[r] for r in trace["goals"]["erm_rules"]},
    }
    rows = []
    for g in trace["generations"]:
        if g["generation"] not in FOCAL:
            continue
        parents = [prepare_tree(programs[i]) for i in g["parents"]]
        if [p.signature for p in parents] != [sigs[rules[i]] for i in g["parents"]]:
            raise AssertionError(
                "Independent symbolic donor preparation changed a parent rule"
            )
        for label, target in groups.items():
            detail = asdict(analyze_population(parents, g["parent_fitness"], target))
            row = {
                **common,
                "generation": g["generation"],
                "calls_before": g["calls_before"],
                "target": label,
                "historical_root_union_pairs": g["joins"][label]["past"]["union"],
                "historical_root_intersection_pairs": g["joins"][label]["past"][
                    "intersection"
                ],
            }
            for k, v in detail.items():
                if isinstance(v, dict):
                    row.update({f"{k}_{op}": value for op, value in v.items()})
                elif isinstance(v, (tuple, list, np.ndarray)):
                    row[k] = json.dumps(np.asarray(v).tolist(), separators=(",", ":"))
                else:
                    row[k] = v
            rows.append(row)
    return rows


def _panel(pair):
    law, seed = pair
    bank = draw_bank(law, seed)
    rules, _ = exact_family()
    fit = counts(bank.fit_cells, bank.fit_labels)
    units = integer_losses(rules, fit)
    val = integer_losses(rules, bank.validation)
    expected = risk(rules, bank.eta, IID)
    records = []

    def save(method, pred, **metadata):
        fit_value = brier(pred, fit)
        records.append(
            {
                "law": law,
                "data_seed": seed,
                "method": method,
                "fit_brier": fit_value,
                "validation_brier": brier(pred, bank.validation),
                "iid_test_brier": brier(pred, bank.iid),
                "shift_test_brier": brier(pred, bank.shift),
                "iid_risk": risk(pred, bank.eta, IID),
                "shift_risk": risk(pred, bank.eta, SHIFT),
                "fit_minus_family_erm": fit_value - units.min() / (400 * fit.size),
                "iid_minus_family": risk(pred, bank.eta, IID) - min(expected),
                "prediction_sha256": hashlib.sha256(
                    np.asarray(pred, dtype="<f8").tobytes()
                ).hexdigest(),
                **metadata,
            }
        )

    y = bank.fit_labels.astype(int)
    x = PROBA[bank.fit_cells]
    top = TopN("brier").fit(x, y, task="binary", n_classes=2)
    stack = StackingLogit().fit(x, y, task="binary", n_classes=2)
    save("mean", SCORES.mean(axis=1), selector="fixed")
    save(
        "top_n",
        top.predict(PROBA)[:, 1],
        selector="fit",
        n_selected=top.n_selected_,
        ranking=json.dumps(top.ranking_.tolist()),
        prefix_scores=json.dumps(top.prefix_scores_.tolist()),
    )
    save("stack_logit", stack.predict(PROBA)[:, 1], selector="fit")
    fit_ref = int(np.argmin(units))
    save("exact_erm", rules[fit_ref], selector="fit", rule=fit_ref, fit_hit=True)
    traces, focus, stages = [], [], []
    for gp_seed in SEEDS:
        trace = trace_search(bank, gp_seed)
        attained = list(dict.fromkeys(trace["rule_ids"][i] for i in trace["archive"]))
        for selector, chosen in (
            ("fit", min(attained, key=lambda r: units[r])),
            ("validation", min(attained, key=lambda r: (val[r], units[r]))),
        ):
            program_index = next(
                i for i in trace["archive"] if trace["rule_ids"][i] == chosen
            )
            save(
                "gp",
                rules[chosen],
                selector=selector,
                gp_seed=gp_seed,
                calls=trace["calls"],
                rule=chosen,
                nodes_at_first_visit=sum(
                    1 for _ in _nodes(trace["programs"][program_index])
                ),
                program=json.dumps(
                    trace["programs"][program_index], separators=(",", ":")
                ),
                unique_rules=len(attained),
                fit_units=int(units[chosen]),
                fit_hit=bool(units[chosen] == min(units)),
                target_hit=trace["goals"]["population_rule"] in attained,
                erm_hit=bool(set(trace["goals"]["erm_rules"]) & set(attained)),
                raw_digest=trace["raw_digest"],
            )
        focus.extend(focal_rows(trace))
        stages.extend(stage_rows(trace))
        traces.append(trace)
    return traces, focus, stages, records


def _nodes(node):
    yield node
    for child in node["children"]:
        yield from _nodes(child)


def run(out, workers):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    primary, focal, stages = [], [], []
    pending = out / "traces.partial.jsonl.gz"
    with (
        gzip.open(pending, "wt", encoding="utf8") as stream,
        ProcessPoolExecutor(max_workers=workers) as pool,
    ):
        for traces, focus, events, records in pool.map(
            _panel, product(("matched", "additive"), BANKS)
        ):
            for trace in traces:
                stream.write(json.dumps(trace, separators=(",", ":")) + "\n")
            focal.extend(focus)
            stages.extend(events)
            primary.extend(records)
    for fname, rows in (("runs", primary), ("focal", focal), ("stages", stages)):
        dest = out / f"{fname}.partial.csv.gz"
        pd.DataFrame(rows).to_csv(dest, index=False)
        dest.replace(out / f"{fname}.csv.gz")
    pending.replace(out / "traces.jsonl.gz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--workers", default=1, type=int)
    args = parser.parse_args()
    run(args.out, args.workers)

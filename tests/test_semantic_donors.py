"""Replay score timing, semantic equivalence and visibility of native MIN/MAX donors."""

import inspect
import json
from dataclasses import replace

import numpy as np
import pytest

from fusionbench.eval.precision_lattice import (
    draw_bank,
    exact_family,
    integer_losses,
    native_search,
)
from fusionbench.eval.search_graded import SCORES, counts


@pytest.fixture
def native_sdk():
    sdk = pytest.importorskip("okapi")
    if "selection_fitness_decimals" not in inspect.signature(sdk.Okapi).parameters:
        pytest.skip("requires OKAPI's native selection-precision option")


def as_node(serialized):
    return json.loads(serialized) if isinstance(serialized, str) else serialized


def lattice_values(node):
    """Evaluate serialized programs directly on the four-input grid in value-child order."""
    assert node["type"] == "ValueNode"
    column = int(node["id"].removeprefix("p").removesuffix(".npy"))
    current = SCORES[:, column]
    for operation in node["children"]:
        combine = {"MinNode": np.minimum, "MaxNode": np.maximum}[operation["type"]]
        for operand in operation["children"]:
            current = combine(current, lattice_values(operand))
    return current


def native_reference(bank, gp_seed, tmp_path, budgets):
    paths = []
    for i in range(4):
        path = tmp_path / f"p{i}.npy"
        np.save(path, SCORES[:, i : i + 1])
        paths.append(path)
    gt = tmp_path / "labels.npy"
    np.save(gt, bank.eta[:, None])
    fit_units = integer_losses(
        exact_family()[0], counts(bank.fit_cells, bank.fit_labels)
    )
    results, _ = native_search(
        paths, gt, fit_units, len(bank.fit_labels), gp_seed, 3, False, budgets
    )
    return results, fit_units


def observed_stages(trace, target_ids):
    """Read actual events, keeping initial scores separate from generation-one births."""
    found = dict.fromkeys(("parents", "crossover", "mutation", "scored", "survivors"))

    def read(phase, ids, generation):
        if found[phase] is None and any(
            trace["rule_ids"][i] in target_ids for i in ids
        ):
            found[phase] = generation

    read("scored", trace["initial_scores"], 0)
    for row in trace["generations"]:
        for phase in found:
            read(phase, row[phase] or [], row["generation"])
    births = [g for g in (found["crossover"], found["mutation"]) if g is not None]
    found["postmutation"] = min(births) if births else None
    return found


def check_trace(trace, snapshot, rules, goals, expected_calls):
    programs = [as_node(p) for p in trace["programs"]]
    by_id = trace["rule_ids"]
    assert len(programs) == len(by_id)
    for node, rule in zip(programs, by_id, strict=True):
        np.testing.assert_array_equal(lattice_values(node), rules[rule])

    sequence = list(trace["initial_scores"])
    assert sorted(by_id[i] for i in sequence) == list(range(4))
    parents = list(sequence)
    for generation, row in enumerate(trace["generations"], start=1):
        assert row["generation"] == generation
        assert row["parents"] == parents
        assert row["calls_before"] == len(sequence)
        assert sorted({by_id[i] for i in sequence}) == row["past_rules"]
        assert len(row["crossover"]) == 8
        offered = [*parents, *row["crossover"], *row["mutation"]]
        # Native deduplication retains each first structural occurrence before fitness.
        unique = {}
        for i in offered:
            unique.setdefault(json.dumps(programs[i], sort_keys=True), i)
        cohort = list(unique.values())
        evaluated = row["scored"]
        assert evaluated == cohort[: len(evaluated)]
        sequence.extend(evaluated)
        if row["complete"]:
            assert evaluated == cohort
            parents = row["survivors"]
            assert len(parents) == len(set(parents)) == 4
            assert set(parents) <= set(evaluated)
        else:
            assert generation == len(trace["generations"])
            assert row["survivors"] is None
    assert sequence == trace["archive"]
    assert len(sequence) == trace["calls"] == trace["real_calls"] == expected_calls
    archive, path_digest = snapshot[:2]
    assert trace["raw_digest"] == path_digest
    first_program = {}
    for i in sequence:
        first_program.setdefault(by_id[i], programs[i])
    assert list(first_program) == list(archive)
    for rule in archive:
        assert first_program[rule] == json.loads(archive[rule][1])

    assert trace["goals"]["population_rule"] == goals["population"]
    assert set(trace["goals"]["erm_rules"]) == goals["erm"]
    for name, target in (("population", {goals["population"]}), ("erm", goals["erm"])):
        assert trace["stages_first_hit"][name] == observed_stages(trace, target)


@pytest.mark.parametrize("law", ["matched", "additive"])
def test_instrumented_gp_replays_native_search_at_partial_budgets(
    native_sdk, tmp_path, law
):
    from fusionbench.eval.semantic_donors import trace_search

    budgets = (5, 23, 65)
    seed = 141
    bank = draw_bank(law, 305)
    snapshots, losses = native_reference(bank, seed, tmp_path, budgets)
    rules, _ = exact_family()
    population_rule = int(np.mean((rules - bank.eta) ** 2, axis=1).argmin())
    if law == "matched":
        np.testing.assert_array_equal(rules[population_rule], bank.eta)
    goals = {
        "population": population_rule,
        "erm": set(np.flatnonzero(losses == losses.min())),
    }
    for budget in budgets:
        trace = trace_search(bank, seed, budget=budget, focal_generations=())
        check_trace(trace, snapshots[budget], rules, goals, budget)


def test_offered_rule_can_be_unscored_at_the_endpoint(native_sdk):
    from fusionbench.eval.semantic_donors import trace_search

    # Keep training fixed and change only the known population law. At budget
    # 5 the first join scored one incumbent, while its offspring already exist.
    bank = draw_bank("matched", 307)
    seed = 141
    trial = trace_search(bank, seed, budget=5, focal_generations=())
    generation = trial["generations"][0]
    assert not generation["complete"]
    evaluated = [*trial["initial_scores"], *generation["scored"]]
    scored_rules = {trial["rule_ids"][i] for i in evaluated}
    # An observer on the untouched native first generation finds rule 13 only
    # among its three appended mutant offspring on this controlled start.
    target = 13
    assert target not in scored_rules
    assert target not in {trial["rule_ids"][i] for i in generation["crossover"]}
    assert target in {trial["rule_ids"][i] for i in generation["mutation"]}
    modified = replace(bank, eta=exact_family()[0][target])
    observed = trace_search(modified, seed, budget=5, focal_generations=())
    assert observed["raw_digest"] == trial["raw_digest"]
    assert observed["goals"]["population_rule"] == target
    hits = observed["stages_first_hit"]["population"]
    assert hits == observed_stages(observed, {target})
    assert hits["crossover"] is None and hits["mutation"] == 1
    assert hits["scored"] is None and hits["survivors"] is None

    founder = replace(bank, eta=SCORES[:, 0])
    observed = trace_search(founder, seed, budget=5, focal_generations=())
    hits = observed["stages_first_hit"]["population"]
    assert hits["scored"] == 0 and hits["parents"] == 1
    assert hits["survivors"] is None


def test_later_focal_population_cannot_change_the_primary_prefix(native_sdk):
    from fusionbench.eval.semantic_donors import trace_search

    bank = draw_bank("matched", 307)
    bank = replace(bank, eta=exact_family()[0][13])
    prefix = trace_search(bank, 141, budget=5, focal_generations=())
    extended = trace_search(bank, 141, budget=5, focal_generations=(5,))
    assert extended["generations"][-1]["generation"] == 5
    assert extended["real_calls"] > prefix["real_calls"]
    assert extended["calls"] == prefix["calls"] == 5
    assert extended["raw_digest"] == prefix["raw_digest"]
    assert extended["archive"] == prefix["archive"]
    assert extended["stages_first_hit"] == prefix["stages_first_hit"]
    # The mutation-born rule truly was scored after 5 in the extended
    # trajectory, yet it cannot be a score hit in the primary prefix.
    later_scored = [i for g in extended["generations"] for i in g["scored"]]
    assert 13 in {extended["rule_ids"][i] for i in later_scored}
    assert extended["stages_first_hit"]["population"]["scored"] is None

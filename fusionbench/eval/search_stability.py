"""Cross data seeds with GP seeds, then compare selectors within the same final population."""

from __future__ import annotations

import argparse
import csv
import json
import time
from functools import partial
from pathlib import Path

import numpy as np

from fusionbench.methods.averaging import TopN
from fusionbench.methods.okapi import OKAPIMethod, _identity
from fusionbench.methods.stacking import StackingLogit
from fusionbench.scenarios.correlated import CorrelatedRedundancy
from fusionbench.scenarios.gates import Parity, TwoPathways, WeakestLink


def _macro_f1(predictions, labels):
    prediction = np.asarray(predictions).reshape(-1).astype(bool)
    y = np.asarray(labels).reshape(-1).astype(bool)
    tp, tn = np.sum(y & prediction), np.sum(~y & ~prediction)
    mistakes = np.sum(y != prediction)
    return .5 * (2 * tp / max(1, 2 * tp + mistakes) + 2 * tn / max(1, 2 * tn + mistakes))


def _objective(scores, labels, *, counter):
    counter[0] += 1
    return _macro_f1(scores > .5, labels)


def _scenario_pair(name):
    # The shift preserves model IDs, event definitions and feature dimensions.
    if name == "two_pathways":
        return TwoPathways(rep=2), TwoPathways(rep=2, accuracy=.78)
    if name == "weakest_link":
        return WeakestLink(m=4), WeakestLink(m=4, accuracy=.76)
    if name == "parity":
        return Parity(rep=3), Parity(rep=3, accuracy=.78)
    if name == "correlated":
        return CorrelatedRedundancy(), CorrelatedRedundancy(corr=.10)
    raise ValueError(f"Unknown scenario {name}")


def _split(scenario, n, data_seed, index):
    preds, labels, latent = scenario._generate(n, np.random.default_rng([data_seed, index]))
    return preds, labels, scenario.bayes_oracle(preds, latent)[:, 1]


def _tensor_map(preds):
    return {f"m{i}.npy": preds[:, i, 1:2] for i in range(preds.shape[1])}


def _predict_trees(trees, preds):
    tensors = _tensor_map(preds)
    return np.stack([tree.do_pred_on_another_tensors(current_tensors=tensors).ravel() for tree in trees])


def _group_losses(predictions, y, groups):
    indices = np.array_split(np.arange(len(y)), groups)
    return np.array([[1 - _macro_f1(predicted[group], y[group]) for group in indices]
                     for predicted in predictions])


def _one_search(scenario_name, data_seed, search_seed, generations, fit_n, select_n, test_n, groups, full_fit, checkpoints):
    from okapi.okapi import Okapi
    from okapi.pareto import maximize
    from okapi.selection import select_compact

    native, shifted = _scenario_pair(scenario_name)
    train = _split(native, fit_n, data_seed, 0)
    selection = _split(native, select_n, data_seed, 1)
    trials = {"iid": _split(native, test_n, data_seed, 2),
              "shift": _split(shifted, test_n, data_seed, 3)}
    if full_fit:
        train = tuple(np.concatenate([a, b]) for a, b in zip(train, selection))
    adapter = OKAPIMethod(n_seeds=8)
    count = [0]
    start = time.perf_counter()
    try:
        directory = adapter._write_split(train[0], "fit")
        labels_file = Path(adapter._tmp.name) / "gt.npy"
        np.save(labels_file, train[1])
        # Explicit ordered paths stabilize ID ordering across filesystems.
        paths = [directory / key for key in _tensor_map(train[0])]
        gp = Okapi(paths, labels_file, population_size=min(10, len(paths)),
                   population_multiplier=2, tournament_size=3, minimize_node_count=True,
                   objective_functions=(partial(_objective, counter=count),), objectives=(maximize,),
                   backend="numpy", seed=search_seed, postprocessing_function=_identity)
        adapter._seed_population(gp, np.random.default_rng(search_seed))
        archive, archive_fitness = [], []
        for step in range(1, generations + 1):
            gp.run_iteration()
            if step in checkpoints and step < generations:
                archive.extend(tree.copy() for tree in gp.population)
                archive_fitness.extend(gp.fitnesses[:, 0])
        search_s = time.perf_counter() - start
        pool = archive + gp.population
        fit_f1 = np.concatenate([archive_fitness, gp.fitnesses[:, 0]])
        fit_best = gp.pareto_trees[int(np.argmax(gp.pareto_fitnesses[:, 0]))]
        fit_index = next(i for i, tree in enumerate(pool) if tree is fit_best)
        sizes = [int(t.nodes_count) for t in pool]
        if full_fit:
            selected_loss = None
            choices = {"all_fit_champion": fit_index}
        else:
            selected_loss = _group_losses(_predict_trees(pool, selection[0]) > .5, selection[1], groups)
            compact = select_compact(selected_loss, sizes)
            choices = {"fit_champion": fit_index, "holdout_best": compact.best_index,
                       "holdout_compact": compact.index}
        # Freeze the choices before generating any scores on evaluation splits.
        single_fit = [_macro_f1(train[0][:, j, 1] > .5, train[1]) for j in range(len(paths))]
        best_single = int(np.argmax(single_fit))
        top_n = TopN(metric="macro_f1").fit(train[0], train[1], task="binary", n_classes=2)
        stacker = StackingLogit().fit(train[0], train[1], task="binary", n_classes=2)
        simple_selection = {}
        if not full_fit:
            sx = selection[0]
            simple_selection = {
                "average": sx[:, :, 1].mean(axis=1),
                "single": sx[:, best_single, 1],
                "top_n": top_n.predict(sx)[:, 1],
                "stack_logit": stacker.predict(sx)[:, 1],
            }
        simple_f1 = {method: 1 - _group_losses(np.array([p > .5]), selection[1], groups).mean()
                     for method, p in simple_selection.items()}
        rows, candidates = [], []
        for domain, (x, y, oracle) in trials.items():
            scores = _predict_trees(pool, x)
            accuracy = np.mean((scores > .5) == y, axis=1)
            f1 = np.array([_macro_f1(p > .5, y) for p in scores])
            common = dict(scenario=scenario_name, data_seed=data_seed, search_seed=search_seed, domain=domain,
                          generations=generations, n_fit=len(train[1]), n_select=0 if full_fit else select_n, n_test=test_n, groups=groups,
                          checkpoints=",".join(map(str, checkpoints)), pool=len(pool), fitness_calls=count[0], search_seconds=search_s,
                          oracle_accuracy=np.mean((oracle > .5) == y), positives=int(y.sum()))
            for j, tree in enumerate(pool):
                candidates.append(dict(**common, candidate=j, tree=repr(tree), nodes=sizes[j],
                                       fit_f1=fit_f1[j], selection_f1="" if full_fit else 1-selected_loss[j].mean(),
                                       group_loss="" if full_fit else json.dumps(selected_loss[j].tolist()),
                                       accuracy=accuracy[j], macro_f1=f1[j]))
            for method, j in choices.items():
                rows.append(dict(**common, method=method, candidate=j, nodes=sizes[j], n_selected="", accuracy=accuracy[j],
                                 macro_f1=f1[j], pool_f1_gap=f1.max()-f1[j], fit_f1=fit_f1[j],
                                 selection_f1="" if full_fit else 1-selected_loss[j].mean()))
            for method, p in (("average", x[:, :, 1].mean(axis=1)), ("single", x[:, best_single, 1]),
                              ("stack_logit", stacker.predict(x)[:, 1]), ("top_n", top_n.predict(x)[:, 1]), ("bayes", oracle)):
                rows.append(dict(**common, method=method, candidate="", nodes="",
                                 n_selected=top_n.n_selected_ if method == "top_n" else "", accuracy=np.mean((p > .5) == y),
                                 macro_f1=_macro_f1(p > .5, y), pool_f1_gap="",
                                 fit_f1=top_n.prefix_scores_[top_n.n_selected_-1] if method == "top_n" else "",
                                 selection_f1=simple_f1.get(method, "")))
    finally:
        adapter._tmp.cleanup()
    return rows, candidates


def summarize_restarts(out):
    """Compare policies across all restarts, using only fit or selection scores to pick."""
    import pandas as pd
    from okapi.selection import select_compact

    out = Path(out)
    search = pd.read_csv(out / "search.csv")
    raw = pd.read_csv(out / "candidates.csv")
    rows = []
    for (scenario, data_seed), candidates in raw.groupby(["scenario", "data_seed"], sort=False):
        validation = candidates[candidates.domain == "iid"].reset_index(drop=True)
        runs = search[(search.scenario == scenario) & (search.data_seed == data_seed)]
        iid = runs[runs.domain == "iid"]
        fit = iid[iid.method.isin(["fit_champion", "all_fit_champion"])]
        incumbent = fit.iloc[np.argmax(fit.fit_f1)]
        selected = {"restart_" + incumbent.method: incumbent}
        if validation.group_loss.notna().all():
            group_losses = np.stack(validation.group_loss.map(json.loads))
            compact = select_compact(group_losses, validation.nodes.to_numpy(dtype=int))
            best = validation.iloc[compact.best_index]
            selected["restart_holdout_best"] = best
            selected["restart_holdout_compact"] = validation.iloc[compact.index]
            baselines = iid[iid.method.isin(["single", "average", "top_n", "stack_logit"])].drop_duplicates("method")
            simple = baselines.iloc[np.argmax(baselines.selection_f1)]
            selected["restart_with_baselines"] = best if best.selection_f1 > simple.selection_f1 else simple
        for rule, winner in selected.items():
            for domain in ("iid", "shift"):
                if pd.isna(winner.candidate):
                    rec = runs[(runs.domain == domain) & (runs.method == winner.method)].iloc[0]
                    family = winner.method
                else:
                    rec = candidates[(candidates.domain == domain) & (candidates.search_seed == winner.search_seed)
                                     & (candidates.candidate == winner.candidate)].iloc[0]
                    family = "gp"
                rows.append(dict(scenario=scenario, data_seed=data_seed, domain=domain, policy=rule,
                                 model=family, chosen_search_seed=rec.search_seed, nodes=rec.nodes,
                                 n_selected=rec.get("n_selected", "") if family == "top_n" else "",
                                 accuracy=rec.accuracy, macro_f1=rec.macro_f1,
                                 accuracy_regret=rec.oracle_accuracy - rec.accuracy,
                                 restarts=len(fit), total_fitness_calls=int(fit.fitness_calls.sum()),
                                 total_search_seconds=float(fit.search_seconds.sum())))
    pd.DataFrame(rows).to_csv(out / "restarts.csv", index=False)


def run_search_stability(out, data_seeds=tuple(range(4)), search_seeds=tuple(range(4)),
                         scenarios=("two_pathways", "weakest_link", "correlated", "parity"),
                         generations=12, fit_n=96, select_n=384, test_n=4096, groups=8, full_fit=False, checkpoints=()):
    if generations < 1 or min(fit_n, test_n) < 1 or not 2 <= groups <= select_n:
        raise ValueError("Use positive generations/sample sizes and 2 <= groups <= select_n")
    if not data_seeds or not search_seeds or not scenarios:
        raise ValueError("Pass at least one scenario, data seed and search seed")
    if any(point < 1 or point > generations for point in checkpoints):
        raise ValueError("Checkpoints must fall within this search")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with (out/"search.csv").open("w", newline="") as results, (out/"candidates.csv").open("w", newline="") as raw:
        writers = []
        for name in scenarios:
            for data_seed in data_seeds:
                for search_seed in search_seeds:
                    records = _one_search(name, data_seed, search_seed, generations, fit_n, select_n, test_n, groups, full_fit, checkpoints)
                    if not writers:
                        for stream, rows in zip((results, raw), records):
                            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
                            writer.writeheader()
                            writers.append(writer)
                    for stream, writer, rows in zip((results, raw), writers, records):
                        writer.writerows(rows)
                        stream.flush()

    summarize_restarts(out)


def add_stability_arguments(parser):
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--data-seeds", type=int, nargs="+", default=list(range(4)))
    parser.add_argument("--search-seeds", type=int, nargs="+", default=list(range(4)))
    parser.add_argument("--scenarios", nargs="+", default=["two_pathways", "weakest_link", "correlated", "parity"])
    parser.add_argument("--generations", type=int, default=12)
    parser.add_argument("--fit-n", type=int, default=96)
    parser.add_argument("--select-n", type=int, default=384)
    parser.add_argument("--test-n", type=int, default=4096)
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--checkpoints", nargs="*", type=int, default=[], help="Add intermediate populations to the selection pool.")
    parser.add_argument("--full-fit", action="store_true", help="Use both labelled splits for GP fit; select by fit only.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_stability_arguments(parser)
    run_search_stability(**vars(parser.parse_args()))

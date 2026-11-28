"""Run the preregistered native replacement policies on the 24 standard scenarios."""

import argparse
import gzip
import json
import platform
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import sklearn
from inputs import REPEATS, SCENARIOS, controls, fingerprint, load_bank, metrics
from threadpoolctl import threadpool_limits

from fusionbench.eval.suite_point_gp import predict_tree

HERE = Path(__file__).resolve().parent
NATIVE = ("ordinary", "open_raw", "open_kl", "strict_kl", "ordinary_full")
DISCOVERY = 256
LOOKS = 6
GATE_ROWS = 64


def evaluate_bank(job):
    from fusionbench.eval.replacement_search import run_replacement_search

    scenario, rep = job
    law, data_seed, search_seed, x, y, test, labels = load_bank(scenario, rep)
    key = {
        "scenario": scenario,
        "rep": rep,
        "data_seed": data_seed,
        "search_seed": search_seed,
    }
    blocks = [
        (x[i : i + GATE_ROWS], y[i : i + GATE_ROWS])
        for i in range(DISCOVERY, len(y), GATE_ROWS)
    ]
    assert len(blocks) == LOOKS and all(len(by) == GATE_ROWS for _, by in blocks)
    bank = {
        **key,
        "n_providers": len(law.model_ids()),
        "n_classes": law.n_classes,
        "n_source": len(y),
        "n_test": len(labels),
        "discovery_rows": DISCOVERY,
        "gate_looks": LOOKS,
        "gate_rows": GATE_ROWS,
        "source_inputs_sha256": fingerprint(x),
        "source_labels_sha256": fingerprint(y),
        "test_inputs_sha256": fingerprint(test),
        "test_labels_sha256": fingerprint(labels),
        "gate_blocks": [
            {"input_sha256": fingerprint(bx), "label_sha256": fingerprint(by)}
            for bx, by in blocks
        ],
    }
    rows, look_rows = [], []
    # The held-out tensors are only queried after an arm's entire fitted search has returned.
    with threadpool_limits(limits=1):
        for n in (DISCOVERY, len(y)):
            tick = perf_counter()
            fitted = controls(x[:n], y[:n], law.task, law.n_classes)
            for name, model in fitted.items():
                if name == "mean" and n == DISCOVERY:
                    continue
                label = name if name == "mean" else f"{name}_{n}"
                rows.append(
                    {
                        **key,
                        "method": label,
                        **metrics(model.predict(test), labels),
                        "discovery_rows": 0 if name == "mean" else n,
                        "gate_rows": 0,
                        "top_n": getattr(model, "n_selected_", None),
                    }
                )
            errors = [metrics(x[:n, i], y[:n])["brier"] for i in range(x.shape[1])]
            single = int(np.argmin(errors))
            rows.append(
                {
                    **key,
                    "method": f"best_single_{n}",
                    **metrics(test[:, single], labels),
                    "discovery_rows": n,
                    "gate_rows": 0,
                    "providers_used": 1,
                    "singleton": single,
                }
            )
            bank[f"controls_seconds_{n}"] = perf_counter() - tick

        trajectories = {}
        for method in NATIVE:
            full = method == "ordinary_full"
            policy = "ordinary" if full else method
            n = len(y) if full else DISCOVERY
            tick = perf_counter()
            result = run_replacement_search(
                x[:n],
                y[:n],
                None if full else blocks,
                seed=search_seed,
                policy=policy,
                incumbent="best_single",
            )
            seconds = perf_counter() - tick
            pred = predict_tree(result.tree, test)
            stats, history = search_statistics(result, test, labels)
            for stage in history:
                look_rows.append({**key, "method": method, **stage})
            rows.append(
                {
                    **key,
                    "method": method,
                    **metrics(pred, labels),
                    "discovery_rows": n,
                    "gate_rows": 0 if full else LOOKS * GATE_ROWS,
                    "nodes": int(result.tree.nodes_count),
                    "providers_used": int(result.unique_providers),
                    "champion_id": result.champion_id,
                    "path_digest": result.path_digest,
                    "discovery_evaluations": int(result.discovery_evaluations),
                    "proposal_count": int(result.proposal_count),
                    "seconds": seconds,
                    "scored_discovery_row_instances": n
                    * (int(result.discovery_evaluations) + int(result.incumbent_scan)),
                    "scored_gate_row_instances": GATE_ROWS * stats["gate_evaluations"],
                    **stats,
                }
            )
            trajectories[method] = result.path_digest
        if (
            not trajectories["ordinary"]
            == trajectories["open_raw"]
            == trajectories["open_kl"]
        ):
            raise AssertionError(
                "The three open policies did not see the identical native proposal/breeder path"
            )
    return rows, look_rows, bank


def search_statistics(result, test, test_labels):
    # These look-by-look diagnostic scores are taken only after the full search returns.
    retrospective = {}

    def test_brier(birth_id):
        if birth_id not in retrospective:
            tree = result.records[birth_id].tree
            retrospective[birth_id] = metrics(predict_tree(tree, test), test_labels)[
                "brier"
            ]
        return retrospective[birth_id]

    rows = []
    for record in result.generations:
        row = asdict(record)
        row["test_brier_before"] = test_brier(record.incumbent_before)
        row["test_brier_after"] = test_brier(record.incumbent_after)
        rows.append(row)
    champion = result.records[result.champion_id]
    return {
        "gate_comparisons": sum(g.gate_comparisons for g in result.generations),
        "gate_evaluations": sum(
            g.gate_comparisons + int(g.gate_comparisons > 0) for g in result.generations
        ),
        "gate_moves_helped_test": sum(
            r["promoted"] is not None
            and r["test_brier_after"] - r["test_brier_before"] < -1e-12
            for r in rows
        ),
        "gate_moves_harmed_test": sum(
            r["promoted"] is not None
            and r["test_brier_after"] - r["test_brier_before"] > 1e-12
            for r in rows
        ),
        "promotions": sum(g.promoted is not None for g in result.generations),
        "incumbent_scan": int(result.incumbent_scan),
        "distinct_genotypes": len({r.genotype for r in result.records}),
        "archived_records": len(result.records),
        "champion_sha256": champion.genotype,
        "initial_incumbent_sha256": result.records[
            result.initial_incumbent_id
        ].genotype,
        "deployed_changed": result.displaced,
        "deployed_has_unconfirmed_ancestor": champion.has_unconfirmed_ancestor,
    }, rows


def report(out, frame):
    means = frame.groupby("method")[["brier", "accuracy"]].mean()
    means.to_csv(out / "means.csv")
    values = frame.pivot(index=["scenario", "rep"], columns="method", values="brier")
    comparisons = (
        ("open_kl", "open_raw"),
        ("open_kl", "strict_kl"),
        ("open_kl", "ordinary"),
        ("open_raw", "ordinary"),
        ("strict_kl", "ordinary"),
        ("ordinary_full", "ordinary"),
        ("open_kl", "ordinary_full"),
        ("open_kl", "best_single_256"),
        ("open_kl", "mean"),
        ("open_kl", "top_n_640"),
        ("open_kl", "prob_lr_640"),
        ("open_kl", "logit_lr_640"),
    )
    paired = pd.DataFrame(
        {a + "_minus_" + b: values[a] - values[b] for a, b in comparisons}
    )
    paired.to_csv(out / "paired.csv")
    paired.groupby("scenario").mean().to_csv(out / "scenario-contrasts.csv")
    frame.groupby(["scenario", "method"])[["brier", "accuracy"]].mean().to_csv(
        out / "scenario-means.csv"
    )
    rng = np.random.default_rng(5230999)
    draws = rng.integers(REPEATS, size=(10000, len(SCENARIOS), REPEATS))
    stats = []
    for name, r in paired.items():
        v = r.unstack("rep").to_numpy()
        if v.shape != (len(SCENARIOS), REPEATS):
            raise ValueError("The panel is incomplete")
        sampled = v[np.arange(len(SCENARIOS))[None, :, None], draws].mean((1, 2))
        lo, hi = np.quantile(sampled, [0.025, 0.975])
        avg = v.mean(1)
        stats.append(
            {
                "contrast": name,
                "mean": v.mean(),
                "ci025": lo,
                "ci975": hi,
                "wins": (v < -1e-12).sum(),
                "ties": (np.abs(v) <= 1e-12).sum(),
                "losses": (v > 1e-12).sum(),
                "scenario_wins": (avg < -1e-12).sum(),
                "scenario_ties": (np.abs(avg) <= 1e-12).sum(),
                "scenario_losses": (avg > 1e-12).sum(),
            }
        )
    pd.DataFrame(stats).to_csv(out / "contrasts.csv", index=False)
    budget = [
        "proposal_count",
        "incumbent_scan",
        "discovery_evaluations",
        "gate_comparisons",
        "gate_evaluations",
        "scored_discovery_row_instances",
        "scored_gate_row_instances",
        "nodes",
        "providers_used",
        "promotions",
        "seconds",
    ]
    frame[frame.method.isin(NATIVE)].groupby("method")[budget].mean().to_csv(
        out / "mean-budgets.csv"
    )


def serial(value):
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Unsupported result field {type(value).__name__}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "results")
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=2)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    jobs = [(sc, i) for sc in SCENARIOS for i in range(REPEATS)]
    scores, looks, banks = [], [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for rows, look_rows, bank in pool.map(evaluate_bank, jobs):
            scores.extend(rows)
            looks.extend(look_rows)
            banks.append(bank)
    frame = pd.DataFrame(scores)
    if (
        frame.duplicated(["scenario", "rep", "method"]).any()
        or len(banks) != 24 * REPEATS
    ):
        raise ValueError("Unexpected duplicate or missing bank rows")
    frame.to_csv(args.out / "scores.csv", index=False)
    (args.out / "banks.json").write_text(
        json.dumps(banks, indent=2, default=serial) + "\n"
    )
    with (
        (args.out / "generations.jsonl.gz").open("wb") as output,
        gzip.GzipFile(fileobj=output, mode="wb", mtime=0) as compressed,
    ):
        for row in looks:
            compressed.write((json.dumps(row, default=serial) + "\n").encode())
    report(args.out, frame)
    (args.out / "run.json").write_text(
        json.dumps(
            {
                "n_banks": len(banks),
                "n_scores": len(scores),
                "n_generation_records": len(looks),
                "seconds": perf_counter() - start,
                "workers": args.workers,
                "python": platform.python_version(),
                "numpy": np.__version__,
                "sklearn": sklearn.__version__,
                "standard_scenarios": SCENARIOS,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

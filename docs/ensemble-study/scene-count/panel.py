"""Evaluate the fixed standard-v4 panel and the paired clone-by-reversal experiment."""

import argparse
import hashlib
import json
import platform
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from runpy import run_path
from time import perf_counter

import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits

from fusionbench import generate_dataset, get_scenario
from fusionbench.eval.suite_point_gp import predict_tree
from fusionbench.eval.transfer_grid import brier
from fusionbench.methods.averaging import SimpleAverage, TopN
from fusionbench.methods.native_pattern import RegularizedNativePatternOKAPI
from fusionbench.methods.stacking import StackingLR
from fusionbench.transfer import load_preset

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "seeded-checkpoint"
sys.path.insert(0, str(PREVIOUS))
previous = run_path(str(PREVIOUS / "panel.py"))
search, tree_digest = previous["search"], previous["digest"]

METHODS = ("seeded", "mean", "top_n", "prob_lr", "regularized")
STANDARD_SEEDS = (286201, 286202)
FACTORIAL_SEEDS = tuple(range(286300, 286324))
SCENARIO = "cloned_shortcut"
CONDITIONS = {
    (0, 0): "neither",
    (1, 0): "redundancy_only",
    (0, 1): "shift_only",
    (1, 1): "both",
}
B = 10_000
BOOTSTRAP_SEED = 97823


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def fit_models(x, y):
    """Freeze exactly one seeded GP and four references using fit inputs alone."""
    _, providers, classes = x.shape
    start = perf_counter()
    found = search(x, y, "seeded", 0)
    tree, fit_loss, _ = found["archive"][found["best"]]
    last = found["snapshots"][-1]
    context = {
        "n_fit": len(y),
        "pool_size": providers,
        "n_classes": classes,
        "search_seed": 0,
        "fit_inputs_sha256": digest(x),
        "fit_labels_sha256": digest(y),
    }
    fits = {
        "seeded": (
            partial(predict_tree, tree),
            {
                **context,
                "fit_brier": fit_loss,
                "fit_seconds": perf_counter() - start,
                "unique_providers": len(tree.unique_value_node_ids),
                "objective_calls": last["calls"],
                "offspring_proposals": last["proposals"],
                "family": "seeded",
                "selected_digest": tree_digest(found["best"]),
            },
        )
    }
    path = [
        {
            "generation": snap["generation"],
            "n_survivors": len(snap["keys"]),
            "objective_calls": snap["calls"],
            "offspring_proposals": snap["proposals"],
            "mean_exact_present": found["mean_key"] in snap["keys"],
            "best_active_fit_brier": min(
                found["archive"][key][1] for key in snap["keys"]
            ),
            "survivors_digest": tree_digest(snap["keys"]),
        }
        for snap in found["snapshots"]
    ]
    task = "binary" if classes == 2 else "multiclass"
    for name, model in (
        ("mean", SimpleAverage()),
        ("top_n", TopN(metric="brier")),
        ("prob_lr", StackingLR(c_value=1.0)),
        ("regularized", RegularizedNativePatternOKAPI(seed=0, split_seed=0)),
    ):
        start = perf_counter()
        model.fit(x, y, task=task, n_classes=classes)
        seconds = perf_counter() - start
        stats = {
            **context,
            "fit_seconds": seconds,
            "unique_providers": providers,
            "fit_brier": brier(model.predict(x), y),
        }
        if name == "top_n":
            stats["unique_providers"] = model.n_selected_
        if name == "regularized":
            chosen = model.trees_[model.chosen_]
            stats.update(
                unique_providers=len(chosen.unique_value_node_ids),
                family=model.chosen_,
                selected_digest=tree_digest(chosen.structural_key()),
                objective_calls=model.counts_["total_score_calls"],
            )
        fits[name] = (model.predict, stats)
    return fits, path


def score_models(fits, x, y, context):
    test = {
        "n_test": len(y),
        "test_inputs_sha256": digest(x),
        "test_labels_sha256": digest(y),
    }
    return [
        {
            **context,
            **stats,
            **test,
            "method": method,
            "test_brier": brier(predict(x), y),
        }
        for method, (predict, stats) in fits.items()
    ]


def evaluate_standard(job):
    scenario, seed = job
    with threadpool_limits(limits=1):
        bank = generate_dataset(scenario, seed=seed, n_per_split=512)
        context = {"scenario_id": scenario, "data_seed": seed}
        fits, path = fit_models(bank.fit_preds, bank.fit_y)
        scores = score_models(fits, bank.test_preds, bank.test_y, context)
        return scores, [{**context, **r} for r in path]


def factorial_bank(seed, clones, reversal):
    law = get_scenario(
        SCENARIO,
        clone_probability=float(clones),
        test_reversal_probability=float(reversal),
    )
    rng = np.random.default_rng(seed)
    fit_x, fit_y, _ = law.sample_split(1024, rng, "fit")
    test_x, test_y, _ = law.sample_split(4096, rng, "test")
    return fit_x, fit_y, test_x, test_y


def evaluate_factorial(seed):
    scores, traces = [], []
    stable_test = stable_fit = shared_fit_y = shared_test_y = None
    with threadpool_limits(limits=1):
        for clones in (0, 1):
            x, y, aligned, test_y = factorial_bank(seed, clones, 0)
            other_x, other_y, reversed_x, reversed_y = factorial_bank(seed, clones, 1)
            np.testing.assert_array_equal(other_x, x)
            np.testing.assert_array_equal(other_y, y)
            np.testing.assert_array_equal(reversed_y, test_y)
            np.testing.assert_array_equal(reversed_x[:, :2], aligned[:, :2])
            np.testing.assert_allclose(
                reversed_x[:, 2:, 1], 1 - aligned[:, 2:, 1], rtol=0, atol=1e-15
            )
            if clones:
                np.testing.assert_array_equal(stable_fit, x[:, :2])
                np.testing.assert_array_equal(stable_test, aligned[:, :2])
                np.testing.assert_array_equal(shared_fit_y, y)
                np.testing.assert_array_equal(shared_test_y, test_y)
                # Exact duplicates in the active cell, including after reversal.
                for bank in (x, aligned, reversed_x):
                    np.testing.assert_array_equal(
                        bank[:, 2:], np.broadcast_to(bank[:, 2:3], bank[:, 2:].shape)
                    )
            else:
                stable_fit, stable_test, shared_fit_y, shared_test_y = (
                    x[:, :2],
                    aligned[:, :2],
                    y,
                    test_y,
                )

            fits, path = fit_models(x, y)
            base = {
                "scenario_id": SCENARIO,
                "data_seed": seed,
                "clone_probability": clones,
            }
            traces.extend({**base, **r} for r in path)
            for reversal, inputs in enumerate((aligned, reversed_x)):
                context = {
                    **base,
                    "test_reversal_probability": reversal,
                    "condition": CONDITIONS[clones, reversal],
                }
                scores.extend(score_models(fits, inputs, test_y, context))
    return scores, traces


def panel_analysis(scores, scenarios):
    index = pd.MultiIndex.from_product(
        [scenarios, STANDARD_SEEDS], names=["scenario_id", "data_seed"]
    )
    wide = scores.pivot(
        index=index.names, columns="method", values="test_brier"
    ).reindex(index)
    if (
        set(wide.columns) != set(METHODS)
        or not np.isfinite(wide).all().all()
        or len(scores) != wide.size
    ):
        raise ValueError("Incomplete standard-v4 score grid")
    means = wide.groupby(level="scenario_id").mean().reindex(scenarios)
    rows, aggregate = [], []
    for name, subset in (("standard-v4", scenarios), ("historical-24", scenarios[:-1])):
        draws = np.random.default_rng(BOOTSTRAP_SEED).integers(
            2, size=(B, len(subset), 2)
        )
        for method in METHODS:
            aggregate.append(
                {
                    "suite": name,
                    "scenarios": len(subset),
                    "method": method,
                    "mean_test_brier": means.loc[subset, method].mean(),
                }
            )
        for ref in METHODS[1:]:
            vs = (
                wide["seeded"]
                .sub(wide[ref])
                .unstack()
                .loc[subset, list(STANDARD_SEEDS)]
                .to_numpy()
            )
            boot = vs[np.arange(len(subset))[None, :, None], draws].mean(axis=(1, 2))
            lo, hi = np.quantile(boot, [0.025, 0.975])
            rows.append(
                {
                    "suite": name,
                    "scenarios": len(subset),
                    "method": "seeded",
                    "reference": ref,
                    "estimate": vs.mean(),
                    "ci_low": lo,
                    "ci_high": hi,
                }
            )
    return {
        "panel-family-means": means.reset_index(),
        "panel-means": pd.DataFrame(aggregate),
        "panel-contrasts": pd.DataFrame(rows),
    }


def factorial_analysis(scores):
    wide = scores.pivot(
        index=["data_seed", "method"], columns="condition", values="test_brier"
    )
    expected = pd.MultiIndex.from_product(
        [FACTORIAL_SEEDS, METHODS], names=wide.index.names
    )
    wide = wide.reindex(expected)
    if (
        set(wide.columns) != set(CONDITIONS.values())
        or not np.isfinite(wide).all().all()
        or len(scores) != wide.size
    ):
        raise ValueError("Incomplete 2x2 score grid")
    r0, r1 = wide.redundancy_only - wide.neither, wide.both - wide.shift_only
    s0, s1 = wide.shift_only - wide.neither, wide.both - wide.redundancy_only
    wide = wide.assign(
        redundancy_main=(r0 + r1) / 2, reversal_main=(s0 + s1) / 2, interaction=s1 - s0
    )
    bank = wide.rename_axis(columns="effect").stack().rename("value").reset_index()
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(
        len(FACTORIAL_SEEDS), size=(B, len(FACTORIAL_SEEDS))
    )
    rows = []
    for (method, effect), group in bank.groupby(["method", "effect"]):
        values = (
            group.set_index("data_seed").loc[list(FACTORIAL_SEEDS), "value"].to_numpy()
        )
        lo, hi = np.quantile(values[draws].mean(axis=1), [0.025, 0.975])
        rows.append(
            {
                "method": method,
                "effect": effect,
                "estimate": values.mean(),
                "ci_low": lo,
                "ci_high": hi,
                "n_banks": len(values),
            }
        )
    return {"factorial-bank-effects": bank, "factorial-effects": pd.DataFrame(rows)}


def run(design, scenarios, workers, out):
    start = perf_counter()
    if design == "panel":
        worker = evaluate_standard
        jobs = [(s, seed) for s in scenarios for seed in STANDARD_SEEDS]
        fit_blocks = len(jobs)
    else:
        worker, jobs = evaluate_factorial, FACTORIAL_SEEDS
        fit_blocks = 2 * len(jobs)
    scores, traces = [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for result, path in pool.map(worker, jobs):
            scores.extend(result)
            traces.extend(path)
    scores, traces = pd.DataFrame(scores), pd.DataFrame(traces)
    if (
        len(traces) != 17 * fit_blocks
        or not np.isfinite(scores[["fit_brier", "test_brier"]]).all().all()
    ):
        raise ValueError("Incomplete trajectory or nonfinite score")
    if design == "panel":
        tables = panel_analysis(scores, scenarios)
    else:
        tables = factorial_analysis(scores)
    out.mkdir(exist_ok=True, parents=True)
    for name, frame in {
        f"{design}-scores": scores,
        f"{design}-trajectories": traces,
        **tables,
    }.items():
        frame.to_csv(out / f"{name}.csv", index=False)
    report = {
        "protocol": "541b4c3",
        "okapi_revision": "1d7ad2d1660627585329bc584a9c4634915e8bb1",
        "suite": "standard-v4",
        "design": design,
        "seeds": list(STANDARD_SEEDS if design == "panel" else FACTORIAL_SEEDS),
        "methods": list(METHODS),
        "workers": workers,
        "threads_per_worker": 1,
        "bootstrap_draws": B,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "fit_blocks_per_method": fit_blocks,
        "score_rows": len(scores),
        "trajectory_rows": len(traces),
        "seconds": perf_counter() - start,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "accounting": "Only the seeded GP arm is run. Each factorial clone level has one fit shared across both reversal tests; each test score repeats that fit's cost.",
    }
    (out / f"{design}-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


def plot_factorial(out):
    """Draw the frozen four cell means with their bank-bootstrap intervals."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    table = pd.read_csv(out / "factorial-effects.csv").set_index(["method", "effect"])
    panels = {
        "mean": "Arithmetic mean",
        "prob_lr": "Logistic stacking",
        "seeded": "Seeded native GP",
    }
    series = (
        (("neither", "shift_only"), "Independent votes", "#21756e", "o", "-"),
        (("redundancy_only", "both"), "Cloned votes", "#9a572b", "s", "--"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(10.8, 4.7), sharey=True)
    figure.subplots_adjust(left=0.083, right=0.97, top=0.675, bottom=0.19, wspace=0.16)
    figure.text(
        0.035,
        0.936,
        "Reversing a shortcut under independent and cloned votes",
        color="#172836",
        fontsize=16.5,
        fontweight="semibold",
    )
    figure.text(
        0.035,
        0.865,
        "Six shortcut providers and two stable providers. All methods train on aligned votes.",
        color="#405367",
        fontsize=10.5,
    )
    for ax, (method, title) in zip(axes, panels.items(), strict=True):
        for cells, label, color, marker, line in series:
            rows = table.loc[[(method, cell) for cell in cells]]
            estimates, low, high = (
                rows[key].to_numpy() for key in ("estimate", "ci_low", "ci_high")
            )
            ax.plot(
                [0, 1],
                estimates,
                color=color,
                linestyle=line,
                marker=marker,
                markersize=5,
                linewidth=1.8,
                label=label,
            )
            ax.vlines([0, 1], low, high, color=color, linewidth=1.2)
            ax.plot([0, 1], low, "_", color=color, markersize=7.0)
            ax.plot([0, 1], high, "_", color=color, markersize=7.0)
        ax.set_title(title, fontsize=12, fontweight="semibold", loc="left", pad=11)
        ax.set_xticks([0, 1], ["Aligned", "Reversed"])
        ax.set_xlim(-0.22, 1.22)
        ax.set_ylim(0, 2)
        ax.set_yticks(np.linspace(0, 2, 5))
        ax.grid(axis="y", color="#dae2e6", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#a6b5be")
        ax.tick_params(axis="y", length=0, labelsize=9, colors="#394e5a")
        ax.tick_params(axis="x", length=3, labelsize=10, colors="#394e5a")
    axes[0].set_ylabel("Test Brier (lower is better)", fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.026, 0.83),
        ncol=2,
        frameon=False,
        fontsize=10,
    )
    figure.text(
        0.035,
        0.045,
        "Points show mean full-class Brier across 24 paired banks. Bars show 95% bank-bootstrap intervals.",
        color="#405367",
        fontsize=9.4,
    )
    figure.savefig(out / "compound-interaction.png", dpi=180, facecolor="white")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--design", choices=["panel", "factorial", "all"], default="all"
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--out", type=Path, default=HERE / "results")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Redraw the compound figure from completed CSVs",
    )
    args = parser.parse_args()
    if args.plot_only:
        plot_factorial(args.out)
        return
    scenarios = list(load_preset("standard-v4")["scenarios"])
    if (
        len(scenarios) != 25
        or scenarios[-1] != SCENARIO
        or scenarios[:-1] != list(load_preset("standard-v3")["scenarios"])
    ):
        raise RuntimeError(
            "Expected standard-v3 followed by the new cloned_shortcut scenario"
        )
    for design in ("panel", "factorial") if args.design == "all" else (args.design,):
        run(design, scenarios, args.workers, args.out)
        if design == "factorial":
            plot_factorial(args.out)


if __name__ == "__main__":
    main()

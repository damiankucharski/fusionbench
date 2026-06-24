"""Sweeps over base-model competence and validation reliability.

These turn the benchmark from a single table into a *map*: how each method's test
performance varies as (a) base-model accuracy changes, and (b) the validation
(fit) signal is weakened. The reliability sweep is where parameter-light methods
(averaging, fixed operators, parsimonious OKAPI trees) are expected to overtake
flexible meta-learners that overfit a small/noisy fit split.
"""

from __future__ import annotations

import pandas as pd

from fusionbench.eval.degrade import add_optimism, degrade_fit
from fusionbench.eval.runner import evaluate_dataset


def reliability_sweep(scenario, methods, *, fit_ns, noises=(0.0,), n_per_split=50_000,
                      seed=0) -> pd.DataFrame:
    ds = scenario.build(n_per_split=n_per_split, seed=seed)
    frames = []
    for fit_n in fit_ns:
        for noise in noises:
            dd = degrade_fit(ds, fit_n=fit_n, label_noise=noise, seed=seed)
            df = evaluate_dataset(dd, methods)
            df["fit_n"] = fit_n
            df["label_noise"] = noise
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def optimism_sweep(scenario, methods, *, deltas, n_per_split=50_000, seed=0) -> pd.DataFrame:
    """Vary validation optimism (verification gap); test split stays clean."""
    ds = scenario.build(n_per_split=n_per_split, seed=seed)
    frames = []
    for delta in deltas:
        dd = add_optimism(ds, delta_max=delta, seed=seed) if delta > 0 else ds
        df = evaluate_dataset(dd, methods)
        df["optimism"] = delta
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def competence_sweep(scenario_cls, methods, *, accuracies, n_per_split=50_000, seed=0,
                     **fixed_knobs) -> pd.DataFrame:
    frames = []
    for acc in accuracies:
        try:
            scenario = scenario_cls(accuracy=acc, **fixed_knobs)
        except TypeError:
            raise SystemExit(
                f"{scenario_cls.__name__} has no 'accuracy' knob; competence sweep "
                "unsupported for this scenario."
            )
        df = evaluate_dataset(scenario.build(n_per_split=n_per_split, seed=seed), methods)
        df["accuracy"] = acc
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def sweep_table(df: pd.DataFrame, axis: str) -> pd.DataFrame:
    """Pivot: rows = axis value, cols = method, values = test primary metric."""
    comp = df[(df.method != "oracle") & (df.error.fillna("") == "")].copy()
    primary = comp["primary_metric"].iloc[0]
    comp["test_primary"] = comp[f"test_{primary}"]
    return comp.pivot_table(index=axis, columns="method", values="test_primary")


def save_sweep(df: pd.DataFrame, axis: str, out_dir, title: str = "") -> None:
    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "sweep_raw.csv", index=False)
    table = sweep_table(df, axis)
    table.to_csv(out / f"sweep_by_{axis}.csv")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ax = table.plot(marker="o", figsize=(8, 5))
        ax.set_xlabel(axis)
        ax.set_ylabel("test primary metric")
        ax.set_title(title or f"sweep over {axis}")
        if axis == "fit_n":
            ax.set_xscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
        plt.tight_layout()
        plt.savefig(out / f"sweep_by_{axis}.png", dpi=120)
        plt.close()
    except Exception:
        pass

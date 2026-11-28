"""Inspect the full surviving native GP population on synthetic clustered units."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from okapi import Okapi

from fusionbench.eval.suite_point_gp import predict_tree
from fusionbench.methods.averaging import TopN
from fusionbench.methods.okapi import OKAPIMethod, _task_postprocessing
from fusionbench.methods.stacking import StackingLR

HERE = Path(__file__).resolve().parent


def draw(seed, stream, units, copies, signal):
    rng = np.random.default_rng([seed, stream])
    latent = rng.dirichlet(np.ones(5), size=units)
    x = rng.gamma(0.2 + 3 * latent[:, None, :], size=(units, 8, 5))
    x /= x.sum(axis=2, keepdims=True)
    q = x.mean(axis=1)
    if signal:
        q = 0.2 * q + 0.8 * x[:, 0]
    private = rng.random((units, copies))
    shared = rng.random((units, 1))
    uniform = np.where(rng.random((units, copies)) < 0.8, shared, private)
    y = (uniform[:, :, None] > q.cumsum(axis=1)[:, None, :-1]).sum(axis=2)
    return np.repeat(x, copies, axis=0), y.ravel(), np.repeat(q, copies, axis=0)


def loss(p, y):
    return (p * p).sum(axis=1) - 2 * p[np.arange(len(y)), y] + 1


def run(job):
    iteration, bank, signal, policy, seed_mean = job
    seed = 4720000 + 10000 * (iteration - 1) + bank
    search = seed + 3000000
    x, y, train_posterior = draw(seed, 0, 24, 8, signal)
    t, ty, posterior = draw(seed, 1, 2048, 1, signal)
    mean = x.mean(axis=1)
    test_mean = t.mean(axis=1)
    controls = {"mean": test_mean}
    for key, model in (("top_n", TopN("brier")), ("lr", StackingLR())):
        model.fit(x, y, task="multiclass", n_classes=5)
        controls[key] = model.predict(t)
    base_loss = loss(mean, y).mean()
    if policy == "population":

        def objective(prediction, gt):
            conditional = 1 + (
                prediction * prediction - 2 * prediction * train_posterior
            ).sum(axis=1)
            return 1.0 - conditional.mean() / 2
    elif policy == "raw":

        def objective(prediction, gt):
            return 1.0 - loss(prediction, gt.ravel()).mean() / 2
    else:
        from okapi.paired_fitness import PairedGainFitness

        groups = np.repeat(np.arange(24), 8) if policy == "groups" else None
        guard = PairedGainFitness(mean, y, loss, z=1.0, groups=groups)

        def objective(prediction, gt):
            return np.clip(1.0 + (guard(prediction, gt.ravel()) - base_loss) / 2, 0, 1)

    rows = []
    adapter = OKAPIMethod(seed=search)
    try:
        with _task_postprocessing("multiclass"):
            directory = adapter._write_split(x, "fit")
            labels = Path(adapter._tmp.name) / "labels.npy"
            np.save(labels, y)
            gp = Okapi(
                directory,
                labels,
                8,
                2,
                3,
                objective_functions=(objective,),
                backend="numpy",
                seed=search,
            )
            if seed_mean:
                from okapi.paired_fitness import arithmetic_mean_tree

                gp.population[-1] = arithmetic_mean_tree(gp.train_tensors)
            for generation in range(17):
                if generation:
                    gp.run_iteration()
                if generation not in (0, 4, 16):
                    continue
                for i, tree in enumerate(gp.population):
                    pred = predict_tree(tree, t)
                    fit = np.asarray(tree.predict())
                    rows.append(
                        dict(
                            round=iteration,
                            bank=bank,
                            law="signal" if signal else "mean",
                            policy=policy,
                            generation=generation,
                            candidate=i,
                            fit_brier=loss(fit, y).mean(),
                            fitness=objective(fit, y),
                            test_brier=loss(pred, ty).mean(),
                            excess=np.square(pred - posterior).sum(axis=1).mean(),
                            **{
                                k + "_excess": np.square(v - posterior)
                                .sum(axis=1)
                                .mean()
                                for k, v in controls.items()
                            },
                            **{
                                k + "_brier": loss(v, ty).mean()
                                for k, v in controls.items()
                            },
                        )
                    )
    finally:
        adapter._tmp.cleanup()
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("iteration", type=int, choices=(1, 2))
    parser.add_argument("--seed-mean", action="store_true")
    parser.add_argument("--population-control", action="store_true")
    args = parser.parse_args()
    r = args.iteration
    if args.seed_mean and r != 2:
        parser.error("The starting-rule control belongs to round two")
    policies = ("raw",) if r == 1 else ("raw", "rows", "groups")
    if args.seed_mean:
        policies = ("raw", "groups")
    prefix = f"r{r}" + ("-seed-mean" if args.seed_mean else "")
    jobs = [
        (r, i, s, p, args.seed_mean)
        for i in range(12)
        for s in (False, True)
        for p in policies
    ]
    if args.population_control:
        if r != 2 or not args.seed_mean:
            parser.error("The population control requires 2 --seed-mean")
        jobs = [(2, i, False, "population", True) for i in range(12)]
        prefix += "-population"
    with ProcessPoolExecutor(3) as workers:
        data = pd.DataFrame(row for block in workers.map(run, jobs) for row in block)
    data.to_csv(
        HERE / f"{prefix}-survivors.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    rows = []
    for (bank, law, policy, g), d in data.groupby(
        ["bank", "law", "policy", "generation"]
    ):
        raw = d.loc[d.fit_brier.idxmin()]
        chosen = d.loc[d.fitness.idxmax()]
        rows.append(
            {
                "bank": bank,
                "law": law,
                "policy": policy,
                "generation": g,
                "median": d.excess.median(),
                "best": d.excess.min(),
                "worst": d.excess.max(),
                "raw_selected": raw.excess,
                "selected": chosen.excess,
                "selected_test": chosen.test_brier,
                "mean": raw.mean_excess,
                "top_n": raw.top_n_excess,
                "lr": raw.lr_excess,
            }
        )
    pd.DataFrame(rows).to_csv(HERE / f"{prefix}-banks.csv", index=False)


if __name__ == "__main__":
    main()

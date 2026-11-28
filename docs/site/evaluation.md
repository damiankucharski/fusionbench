# Evaluate a fusion method

A paired comparison starts with one bank. Every candidate fits on the same provider predictions and labels and is evaluated on the same held-out examples. A separate `oracle` row may appear when the scenario supplies a reference. The runner captures method-level exceptions in its `error` column, so inspect that column before aggregating.

## Start with three inexpensive references

```python
from fusionbench import evaluate_dataset, generate_dataset
from fusionbench.methods.averaging import SimpleAverage, TopN
from fusionbench.methods.stacking import StackingLR

bank = generate_dataset("deadweight", seed=11, n_per_split=512, n_bad=12)
baselines = [SimpleAverage(), TopN(metric="brier"), StackingLR(c_value=1.0)]
scores = evaluate_dataset(bank, baselines)
assert scores.error.eq("").all()
print(scores[["method", "fit_brier", "test_brier"]])
```

| Method | What gets fitted |
| --- | --- |
| `simple_average` | Give every provider equal weight on every row. |
| `top_n` | Rank individual providers by fit Brier. Evaluate the first provider, then the equal average of the first two, then the first three and so on. Keep the prefix with the lowest fit Brier and use that same prefix on test. |
| `stacking_lr` | Fit regularized logistic regression to the flattened probability vectors and source labels. Its coefficients may depend on the provider, input class and output class; this allows cross-class correction. |

Use `TopN(metric="brier")` to lock the selection criterion even on a scenario with a different primary metric. With `TopN()`, the dataset evaluator supplies that scenario's primary metric. Logistic regression fits its own logistic objective. All three above are *evaluated* by the same held-out Brier, the average of the per-example sum of squared errors over all classes. Lower values are better. For binary classification, both channels contribute, so this is twice the positive-class-only squared error.

A candidate can subclass `fusionbench.methods.base.Method`, implement `fit(fit_preds, fit_y, *, task, n_classes)` and return a `(rows, classes)` probability array from `predict(preds)`. Pass its instance alongside the baselines to `evaluate_dataset`. An apparent advantage on fit that disappears on test can come from finite samples, adaptive selection, or a change of distribution. Separate these possibilities with independent draws and a controlled intervention.

## Run a named panel from the CLI

```sh
fusionbench run --suite smoke --methods simple_average top_n stacking_lr --n 128 --seed 7 --out results/smoke
fusionbench run --suite standard-v6 --group context_routing --methods simple_average top_n stacking_lr --n 256 --seed 7 --out results/routing
```

Read `results.csv` for individual scenario/method rows and `mechanisms.csv` for mean test Brier by group. `run.json` records the scenario weights used. Omitting `--methods` runs the three inexpensive references. Requesting `--methods okapi` includes them alongside OKAPI. In ordinary suite evaluation the `top_n` registry method follows its configured selection metric; the Python example above explicitly locks both ranking and prefix selection to Brier.

## Repeat across settings

Use `fusionbench.profiles.list_profiles("correlated")` to inspect named changes to that generator. Its `reference` profile uses the constructor defaults. The `profiles` command evaluates predefined interventions across data seeds.

```sh
fusionbench profiles --scenario correlated --profile independent_noise strong_dependence --seeds 11 12 --fit-rows 128 --test-rows 256 --out results/noise
fusionbench transfer-grid --preset smoke --out results/transfer-smoke
```

The profiles run compares a correlated block under independent errors and strong shared noise. In its `paired.csv`, each difference is the method's Brier minus the reference method's Brier on a matched draw. With two seeds the interval is illustrative.

The transfer grid creates training data, an independent labeled source-validation sample for choosing among fusion methods, and a deployment test sample. This **cross-method selection** is a separate operation from a fusion method's own internal training or selection on its fit bank. The grid can perturb development labels, swap provider reports or add log-probability jitter. Some scenarios also expose a change of generator parameters at deployment. Record the resolved configuration and report the test risk of the method selected using source evidence.

## Optional OKAPI comparison

To run the basic OKAPI adapter from the source checkout, use `uv sync --group okapi` or install the pinned Git dependency together with the package.

```sh
python -m pip install . 'torch>=2.2' 'okapi @ git+https://github.com/damiankucharski/OKAPI@9871dac460eb295a9935fd1d28acf2ea54bf7f73'
```

The core baselines run with the ordinary FusionBench dependencies. Evaluate OKAPI on the same splits, with its search budget, fitness and optimizer seed recorded. Results from one native-tree configuration are evidence about that configuration and those data-generating laws. [Group and score scenarios](scoring.md)

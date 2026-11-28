# Evaluate fusion methods

A fusion method learns from the probabilities of the same ordered providers that it will receive at deployment. Keep a test split separate from the data used to choose coefficients, hyperparameters, prefixes or model structures. For real base models, fit the fusion method on out-of-fold predictions or predictions from a separate calibration split.

## Compare three baselines

This example compares the arithmetic mean, learned ranked-prefix Top-N and regularized logistic regression on provider log odds. Top-N ranks providers once on development labels and evaluates every equal-weight prefix. The example explicitly selects N by Brier. The evaluator returns fit/test metrics, the scenario's primary metric, error information and any available oracle comparison.

```python
from fusionbench import generate_dataset
from fusionbench.eval.runner import evaluate_dataset
from fusionbench.methods.averaging import SimpleAverage, TopN
from fusionbench.methods.stacking import StackingLogit

bank = generate_dataset("correlated", n_per_split=600, seed=11)
methods = [SimpleAverage(), TopN("brier"), StackingLogit()]
scores = evaluate_dataset(bank, methods)
assert scores["error"].eq("").all()
print(scores[["method", "test_brier", "test_accuracy"]].sort_values("test_brier"))
```

The reported Brier score is the mean over rows of the sum of squared errors across all classes. Lower is better. For a binary classifier it is twice the mean squared error of the positive-class probability. The known posterior minimizes expected Brier for its law; the observed Brier difference on a particular finite sample can have either sign.

## Submit a fusion method

Implement `fit(fit_preds, fit_y, *, task, n_classes)` when training is needed. Return one normalized class-probability vector per row from `predict(preds)`. The input shape is `(N, K, C)`; the output shape is `(N, C)`. This example adds a componentwise median pool and falls back to the arithmetic mean on rows where every class median is zero.

```python
import numpy as np
from fusionbench import generate_dataset
from fusionbench.eval.runner import evaluate_dataset
from fusionbench.methods.averaging import SimpleAverage
from fusionbench.methods.base import Method

class MedianPool(Method):
    name = "median_pool"

    def predict(self, preds):
        pooled = np.median(preds, axis=1)
        mass = pooled.sum(axis=1, keepdims=True)
        return np.divide(pooled, mass, out=preds.mean(axis=1), where=mass > 0)

bank = generate_dataset("multiclass_committee", n_per_split=300, seed=21)
results = evaluate_dataset(bank, [SimpleAverage(), MedianPool()])
assert results["error"].eq("").all()
print(results[["method", "test_brier"]])
```

For the generic transfer grid, register a zero-argument factory under a unique key in `fusionbench.methods.registry.FACTORIES`, for example `FACTORIES["median_pool"] = MedianPool`. Include that ID in the configuration and call `run_transfer` in the same Python process. The runner creates fresh instances per data bank and noise level. A registry addition in Python is local to that process. [Transfer configuration and results](transfer.md)

Built-ins include averaging and fixed operators, Top-N and stackers on raw probabilities or logits. Additional algorithms can require extras; `fusionbench list` shows availability. Learned OKAPI trees are one application of the same evaluation interface. [OKAPI case study](okapi.md)

## Compare an OKAPI change across the full standard suite

`fusionbench run --methods okapi_native_pattern_l2 --seed 1729 --n 2048 --out results/okapi-standard` evaluates all 30 standard scenarios, including `correlated`, `parity` and `similar_classes_38`. The runner includes simple averaging, ranked-prefix Top-N and probability logistic stacking with each explicit OKAPI comparison. Use the same seeds and row counts for both revisions and compare the per-scenario test scores in `results.csv`.

# Generate and store datasets

A `Dataset` contains two aligned banks, `fit_preds` and `test_preds`, with shape `(examples, providers, classes)`. Class probabilities sum to one on the last axis. Binary banks have both class channels. The integer vectors `fit_y` and `test_y` index that class axis. The list `model_ids` names providers in axis order. Optional oracle arrays describe the scenario's reference prediction and are used at evaluation time.

## Inspect a packaged bank

```python
from fusionbench import load_reference_dataset

bank = load_reference_dataset("similar_classes_38")
assert bank.fit_preds.shape == (128, 3, 10)
assert len(bank.test_y) == 128
print(bank.meta["reference_bank"], bank.meta["reference_sha256"])
```

The package includes one seed-1729 reference per standard scenario, available offline. Reusing it gives every implementation the same test observations. For inference about variation across data draws, generate additional banks. The [reference provenance](reference-provenance.md) lists the archived source revisions and reproduction procedure.

## Draw a configurable bank

```python
import numpy as np
from fusionbench import generate_dataset

def draw(seed):
    return generate_dataset(
        "correlated", n_per_split=512, seed=seed,
        n_block=4, n_diverse=2, accuracy=0.75, corr=0.9,
    )

bank = draw(37)
np.testing.assert_array_equal(draw(37).fit_preds, bank.fit_preds)
assert bank.fit_preds.shape == (512, 6, 2)
print(bank.meta["effective_parameters"])
```

At a fixed package version, scenario implementation, parameters, seed and sample size define the draw. Fit and test use separate samples. Ordinary stationary scenarios use the same law on both splits. Shift scenarios can configure a different law per split. For example, set `shortcut_fit` and `shortcut_test` independently in `rare_shortcut_noise`. Reusing a seed across two parameter settings can pair their latent draws when the generator implements that coupling. Check the generator or the resulting labels before treating examples as matched.

`list_scenarios()` includes all registered generators, including local extensions. `scenario_weights()` in `fusionbench.taxonomy` identifies exactly the 30 standard IDs.

## Save and reload

```python
from tempfile import TemporaryDirectory

import numpy as np
from fusionbench import generate_dataset, load_dataset, save_dataset

bank = generate_dataset("committee", seed=7, n_per_split=64, k=3)
with TemporaryDirectory() as directory:
    path = save_dataset(bank, directory)
    restored = load_dataset(path)
    assert restored.model_ids == bank.model_ids
    np.testing.assert_array_equal(restored.test_y, bank.test_y)
    np.testing.assert_array_equal(restored.test_preds, bank.test_preds.astype(np.float32))
```

Pass a project directory in place of the temporary directory to retain the bank. `save_dataset` returns its actual named subdirectory. The on-disk format uses one float32 probability array per provider per split, int64 labels, and a JSON metadata record.
## Bring predictions from real models

Collect predictions from every provider for the same examples in each split. Align examples by ID and classes by label before stacking. The fit split should use out-of-fold or otherwise held-out base predictions. Keep the test labels separate until scoring.

```python
import numpy as np
from fusionbench import Dataset

def make_bank(fit_by_provider, fit_y, test_by_provider, test_y):
    providers = list(fit_by_provider)
    if set(providers) != set(test_by_provider):
        raise ValueError("The two splits must contain the same providers")
    fit = np.stack([fit_by_provider[k] for k in providers], axis=1)
    test = np.stack([test_by_provider[k] for k in providers], axis=1)
    classes = fit.shape[-1]
    return Dataset(
        name="my_classifiers", task="binary" if classes == 2 else "multiclass",
        n_classes=classes, model_ids=providers, fit_preds=fit, fit_y=np.asarray(fit_y),
        test_preds=test, test_y=np.asarray(test_y), meta={"primary_metric": "brier"},
    )
```

Each dictionary value is one provider’s `(examples, classes)` matrix of probabilities. The resulting bank can go directly to `evaluate_dataset`, or through `save_dataset` for reuse. It has no analytic oracle. [Evaluate a fusion method](evaluation.md)

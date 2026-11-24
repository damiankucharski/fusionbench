# Generate and save data

A dataset contains the per-provider probability vectors and class labels on separate fit and test splits. You can load a bundled sample for a quick comparison, generate new rows from a built-in mechanism, or reload a dataset you saved locally.

## Load a ready-made sample

The installed package includes one versioned reference dataset for each of the 24 standard scenarios (19 original banks, the class-pair bank and four version-3 additions). Each uses default parameters and seed 1729, with 128 examples per split. The loader reads the packaged files offline. Use these small fixed banks to check an implementation or demonstrate an API. Repeated analyses of the same reference bank share the same test examples. Additional registered generators, including `multiclass_provider_panel`, are available through `generate_dataset()`.

```python
from fusionbench import list_scenarios, load_reference_dataset

available = list_scenarios()
assert "committee" in available
reference = load_reference_dataset("committee")
assert reference.fit_preds.shape[0] == reference.test_preds.shape[0] == 128
print(reference.model_ids, reference.fit_preds.shape)
print(reference.meta["reference_bank"], reference.meta["reference_sha256"])
```

The reference-bank version and archive SHA-256 identify the exact saved observations, including their storage precision. Retain both when reporting a result obtained on a reference dataset.

## Generate a different sample

`list_scenarios()` lists registered scenario IDs in insertion order. `get_scenario(id, **parameters)` creates a configured generator, so you can inspect its providers and class count before sampling. `generate_dataset(id, *, n_per_split, seed, **parameters)` returns a new `Dataset`. The catalog gives each generator's accepted parameters and the interpretation of its oracle. [Scenario catalog](scenarios.md)

```python
from fusionbench import generate_dataset, get_scenario

scenario = get_scenario("committee", k=5, accuracy=0.8)
print(scenario.model_ids(), scenario.n_classes)
data = generate_dataset("committee", n_per_split=512, seed=7, k=5, accuracy=0.8)
assert data.fit_preds.shape == data.test_preds.shape == (512, 5, 2)
assert data.fit_y.shape == data.test_y.shape == (512,)
```

The same experiment can be written directly to disk from the CLI. Each `--param` value is a JSON value. The command prints the dataset directory.

```sh
fusionbench generate --scenarios committee --param k=5 --param accuracy=0.8 \
  --n 512 --seed 7 --out data
```

Record the scenario ID, parameter values, seed, sample size and software version for reproduction. Using the same generator, versions and seed reproduces the bank. Use new seeds for independent replications. The ordinary generated dataset contains fit and test rows from the same law; matching source distributions alone does not guarantee a ranking on a finite test sample.

Datasets from the public generator include the caller's constructor arguments in `meta["parameters"]`. Built-ins also record defaults in `meta["effective_parameters"]`. These describe the configuration at construction. Create a new configured scenario when changing an experiment. For a custom generator, also retain the code revision defining its sampling law. The saved recipe for a built-in can be replayed directly:

```python
import numpy as np
from fusionbench import generate_dataset

bank = generate_dataset("correlated", corr=0.4, n_per_split=256, seed=31)
recipe = bank.meta
replay = generate_dataset(
    recipe["scenario_id"], seed=recipe["seed"], n_per_split=recipe["n_per_split"],
    **recipe["parameters"],
)
np.testing.assert_array_equal(bank.test_preds, replay.test_preds)
```

| Dataset attribute | Content |
|---|---|
| `fit_preds`, `test_preds` | Arrays of shape `(rows, providers, classes)`. Entry `[i, j, c]` is provider j's probability for class c on observation i. Each vector is finite, nonnegative and sums to one. |
| `fit_y`, `test_y` | Integer class indices of shape `(rows,)` in `0, …, n_classes - 1`. |
| `model_ids`, `n_classes`, `task` | Ordered provider identifiers, class count and binary or multiclass task. Provider order and class semantics must agree across splits. |
| `fit_oracle`, `test_oracle` | Optional clean-domain posterior arrays of shape `(rows, classes)`. |
| `meta` | Generation and evaluation metadata, including the primary metric and provenance for newly generated datasets. |

## Save and reload a bank

`save_dataset(dataset, root)` creates a named dataset directory under `root` and returns its actual path. Pass that path to `load_dataset`. Each split stores one NumPy probability file per provider. Labels are stored as int64, while probabilities and optional oracle arrays are stored as float32. Compare reloaded probabilities to the float32 cast, particularly when the original generator used float64.

```python
from tempfile import TemporaryDirectory

import numpy as np
from fusionbench import generate_dataset, load_dataset, save_dataset

bank = generate_dataset("committee", n_per_split=128, seed=7, k=3)
with TemporaryDirectory() as directory:
    path = save_dataset(bank, directory)
    restored = load_dataset(path)
    assert restored.model_ids == bank.model_ids
    np.testing.assert_array_equal(restored.test_y, bank.test_y)
    np.testing.assert_array_equal(restored.test_preds, bank.test_preds.astype(np.float32))
```

For a persistent copy, pass a working directory such as `"data"` and retain the returned path. To compare existing external predictions, build a `Dataset` with the same axes and label encoding. Fit rows should contain held-out or out-of-fold base predictions. Keep a separate untouched test split. [Evaluate fusion methods](evaluate.md)

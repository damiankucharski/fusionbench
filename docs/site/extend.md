# Add a scenario

A scenario defines a reproducible joint distribution of provider probabilities and labels. Begin by stating which failure mode you want to vary and which aspects should remain constant as its parameter changes. In this example, two sensors have unequal signal strengths. Increasing the shared-noise fraction increases their dependence while leaving each sensor's class-conditional marginal distribution unchanged.

## Implement the generator

A `Scenario` declares its task, class count and ordered provider IDs. Implement the public `sample(n, rng)` hook. Return probabilities of shape `(n, K, C)`, class indices of shape `(n,)`, and a dictionary of any metadata needed for an oracle. Draw all randomness from `rng`. An empty metadata dictionary is sufficient for an empirical benchmark.

```python
import numpy as np
from fusionbench import Scenario, generate_dataset, register_scenario

class PairedSensors(Scenario):
    task = "binary"
    n_classes = 2
    primary_metric = "brier"

    def __init__(self, shared=0.5):
        if not 0 <= shared <= 1:
            raise ValueError("shared must lie in [0, 1]")
        super().__init__(shared=shared)
        self.shared = shared

    def model_ids(self):
        return ["strong_sensor", "weak_sensor"]

    def sample(self, n, rng):
        y = rng.integers(0, 2, size=n)
        common = rng.standard_normal((n, 1))
        private = rng.standard_normal((n, 2))
        noise = np.sqrt(self.shared) * common + np.sqrt(1 - self.shared) * private
        scores = (2 * y[:, None] - 1) * np.array([1.1, 0.7]) + noise
        positive = 1 / (1 + np.exp(-scores))
        return np.stack((1 - positive, positive), axis=-1), y, {}

register_scenario("paired_sensors", PairedSensors)
bank = generate_dataset("paired_sensors", n_per_split=256, seed=37, shared=0.8)
assert bank.fit_preds.shape == (256, 2, 2)
assert bank.test_oracle is None
```

Registration makes the class discoverable by name in the current Python process, including through `get_scenario`. The default built-in benchmark keeps its original membership after registration. Select your new ID explicitly to evaluate it. Existing IDs are protected against accidental replacement. Run the registration once during application startup. For command-line distribution, integrate the class in the package registry, or invoke the Python runner after registration in your own script.

## Check the scientific contract

- Give each provider one stable ID. Keep identifiers, their order and class meanings fixed across splits and shift endpoints. To add a provider, extend both the generated probability tensor and `model_ids()`.
- Check shape, normalization, class range, determinism at a fixed seed and the intended marginal behavior. Evaluate changes on fresh data seeds. A second run of a fixed sample is a reproducibility check.
- If the clean posterior is known, optionally implement `bayes_oracle(preds, latent)`. A Bayes posterior used as a fusion benchmark must condition on information available from the provider predictions. Document any additional information in a privileged reference. Leave the oracle absent when you cannot justify it.
- For the transfer grid, select a unique `seed_slot` and set `knobs` to the constructor arguments. For the example above, an endpoint of `{"shared": 0.9}` changes the dependence while keeping both provider identities. Report the parameters for both domains. [Transfer benchmark](transfer.md)

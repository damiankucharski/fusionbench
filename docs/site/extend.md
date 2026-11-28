# Add a scenario

A scenario defines a joint distribution over provider probabilities and class labels. Start with the effect you want to isolate and name the quantities that should stay fixed. The example below changes the correlation between two noisy sensors. The marginal distribution of each sensor's logit within each class stays the same because its common and private noise variances sum to one.

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
        y = rng.integers(0, 2, n)
        common = rng.standard_normal((n, 1))
        private = rng.standard_normal((n, 2))
        noise = np.sqrt(self.shared) * common + np.sqrt(1 - self.shared) * private
        logits = (2 * y[:, None] - 1) * np.array([1.1, 0.7]) + noise
        p = 1 / (1 + np.exp(-logits))
        return np.stack((1 - p, p), axis=2), y, {}

register_scenario("paired_sensors", PairedSensors)
bank = generate_dataset("paired_sensors", n_per_split=256, seed=37, shared=0.8)
assert bank.fit_preds.shape == (256, 2, 2)
assert bank.test_oracle is None
```

The class supplies stable provider IDs and implements `sample(n, rng)`, returning a probability tensor, integer labels and optional latent information for evaluation. Draw every random quantity from the generator supplied by FusionBench. The base class creates the fit and test samples. To implement a scenario with an intentional change between splits, specialize `sample_split(n, rng, split)` and record the settings for both domains. An analytic reference is optional. If you implement `bayes_oracle(preds, latent)`, document exactly what information it conditions on.

Registration lasts for the current Python process. Import and register your class before calling `generate_dataset` or the Python evaluator, then select its ID explicitly. Standard-v6 membership remains versioned independently. The example is also available as `examples/paired_sensors.py` in the checkout.

## Check the scientific claim

1. Rebuild with the same parameters and seed, and compare probabilities, labels and provider order on both splits.
2. Probe a control value with a known interpretation. At `shared=0` the two sensors' noise terms are independent given the label. At `shared=1` both use the same noise draw while retaining different signal strengths.
3. Verify the intended mechanism over fresh draws. Here, compare within-class residual correlation and marginal variances across the two settings. Pooling classes would introduce a second source of correlation from the shared label.
4. Evaluate candidate fusions and the three references on paired banks. Retain the parameter settings, seeds and any uncertainty over independent draws. [Evaluation guide](evaluation.md)

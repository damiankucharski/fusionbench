"""Add a generator whose two sensors have fixed marginal quality and shared noise."""

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
        shared_noise = rng.standard_normal((n, 1))
        private_noise = rng.standard_normal((n, 2))
        noise = (
            np.sqrt(self.shared) * shared_noise
            + np.sqrt(1 - self.shared) * private_noise
        )
        logits = (2 * y[:, None] - 1) * np.array([1.1, 0.7]) + noise
        positive = 1 / (1 + np.exp(-logits))
        return np.stack((1 - positive, positive), axis=2), y, {}


if __name__ == "__main__":
    register_scenario("paired_sensors", PairedSensors)
    data = generate_dataset("paired_sensors", seed=37, n_per_split=256, shared=0.8)
    print(data.model_ids, data.fit_preds.shape)

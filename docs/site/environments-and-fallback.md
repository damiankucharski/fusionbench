# Shifting shortcuts and selective fallback

These binary scenarios keep labels balanced in expectation and provide an exact probability of class 1 conditional on each row's observed reports. Both use full-class Brier as their primary metric. Each dataset also carries evaluator-only group IDs in `bank.meta["evaluation_groups"]["fit"]` and `["test"]`. Methods fit on `fit_preds` and `fit_y` and predict from `test_preds`.

## Environment shortcut

`environment_shortcut` has two providers. The stable provider votes for the correct label with probability 0.75. The shortcut votes correctly with probabilities 0.90, 0.50 and 0.10 in environments 0 (aligned), 1 (independent) and 2 (reversed). Its reported confidence in its vote stays at 0.90. Set `stable_accuracy` and `shortcut_accuracy` to change the two signal strengths. The aligned and reversed accuracies sum to one.

The default mixture changes from `fit_mix=(0.8, 0.1, 0.1)` to `test_mix=(0.1, 0.1, 0.8)`. The shortcut's pooled accuracy consequently changes from 0.78 to 0.22, while the stable provider's class-conditional signal law stays fixed. For a stationary control, pass the same mixture for both splits. With `stable_only=True`, the shortcut emits `[0.5, 0.5]` on every row and the stable provider is itself the posterior. The same seed preserves labels and stable reports across these choices.

```python
from fusionbench import generate_dataset

bank = generate_dataset("environment_shortcut", n_per_split=256, seed=17)
env = bank.meta["evaluation_groups"]["test"]["environment"]
assert len(env) == len(bank.test_preds) == 256
```

The posterior averages the shortcut's correctness over the known **split mixture**. It combines that vote likelihood with the stable provider's posterior. Environment IDs are available to score each group's loss. To draw fresh rows explicitly, use `scenario.sample_split(n, rng, split="fit")` or `split="test"`. The ordinary `sample(n, rng)` draws the fit law. In the transfer grid, the clean test condition also uses the fit law. The canonical shift at severity 1 draws the configured test law. Override its mixture through `canonical_shift: {test_mix: [0, 0, 1]}` in a scenario spec. [Transfer configuration](transfer.md)

## Selective fallback

`selective_fallback` has an always-available reference and a second provider that can return `[0.5, 0.5]` on unresolved cases. The reference is correct on 82% of easy cases and 60% of hard cases in expectation. The second provider's underlying vote is correct with probabilities 0.94 and 0.72 respectively. Active reports display confidence 0.85, with an independent chance `neutral_rate=0.20` of reporting `[0.5, 0.5]` naturally. A neutral report can therefore come from either process. Set `expose_flag=True` to add an explicit third provider named `available`, whose positive coordinate is 1 for an active report and 0 for fallback.

In each split of size `n`, the generator rounds `n * hard_fraction` and `n * fallback_rate` to the nearest integers, rounding halves up. Defaults are 0.30 and 0.25. The default `mechanism="difficulty"` censors the hardest rows first. The `"random"` treatment selects the same number using an independent random ordering. Using the same seed holds labels, difficulty and underlying raw reports fixed. Set `fallback_rate=0` for an always-available control. The evaluator exposes `fallback` and `hard` Boolean masks for each split.

```python
import numpy as np
from fusionbench import generate_dataset

def bank_for(mechanism):
    return generate_dataset("selective_fallback", mechanism=mechanism,
                            n_per_split=100, seed=28)

hard, random = bank_for("difficulty"), bank_for("random")
assert sum(hard.meta["evaluation_groups"]["test"]["fallback"]) == 25
assert sum(random.meta["evaluation_groups"]["test"]["fallback"]) == 25
np.testing.assert_array_equal(hard.test_y, random.test_y)
np.testing.assert_array_equal(hard.test_preds[:, 0], random.test_preds[:, 0])
```

The posterior sums over difficulty and availability. For `H` hard and `F` censored rows, a row has probability `min(H, F) / n` of being both under difficulty-based fallback and `(H / n) * (F / n)` under random fallback. The likelihood accounts for natural neutral reports and, when supplied, the availability flag. This gives a row-level posterior for the actual finite split.

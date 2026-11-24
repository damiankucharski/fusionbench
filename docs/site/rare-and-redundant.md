# Rare expertise and redundant provider pools

These two binary scenarios ask whether fusion can identify when a provider adds label information. Both supply the exact posterior given the observed provider columns and the configured data-generating law. Their primary metric is the full-class Brier score, with range 0 to 2. The examples below use fixed random seeds to pair cases across conditions.

## Rare conditional expertise

`rare_specialist` draws a balanced class label and an independent region indicator. The generalist casts a noisy vote for the label with fixed accuracy. Outside the region the specialist casts a fair-coin vote. Inside the region its accuracy is `0.5 + specialist_gain`. The third provider supplies an imperfect binary cue for region membership. Each of the three votes has independent noise conditional on the label and region.

| Knob | Default | Effect |
|---|---:|---|
| `region_prevalence` | 0.10 | Probability of a specialist-region case, from 0 to 1 |
| `specialist_gain` | 0.40 | Increment in specialist accuracy in the region, from 0 to 0.5 |
| `cue_accuracy` | 0.85 | Symmetric probability that the cue reports the correct region, from 0.5 to 1 |
| `generalist_accuracy` | 0.75 | Generalist vote accuracy for both classes and both regions, from 0.5 to 1 |

The three provider IDs are `generalist`, `specialist`, and `routing_cue`. The generalist emits its calibrated class-one probability. The specialist vote and routing cue each encode a bit as 0.25 or 0.75 in the class-one column. The cue alone carries zero information about the class, but knowing it can change the reliability of the specialist. The specified numeric encoding lets any fusion method consume the cue through the usual provider input.

Given the observed cue, let `w` be the conditional probability of the hidden region. The prevalence and cue accuracy determine `w`. The specialist's implied accuracy given the cue is `0.5 + w × specialist_gain`. The oracle combines the two class-conditional vote likelihoods using the balanced prior. Zero specialty gain recovers exactly the generalist posterior. A chance-level cue makes the oracle invariant to the cue while the specialist can still contribute marginal information.

```python
import numpy as np
from fusionbench import get_scenario

s = get_scenario("rare_specialist", region_prevalence=0.025, cue_accuracy=0.85)
p, y, markers = s.sample(5_000, np.random.default_rng(42))
q = s.bayes_oracle(p, {})
loss = ((np.eye(2)[y] - q) ** 2).sum(axis=1)
for region in (False, True):
    rows = markers["region"] == region
    print(region, rows.sum(), loss[rows].mean())
```

The private boolean marker is available to the evaluator through `sample` or, on a built or saved Dataset, `dataset.meta["evaluation_groups"]["test"]["region"]` and its `"fit"` counterpart. Record the number of rare examples in development and evaluation alongside slice scores. The observed posterior minimizes expected Brier over the full population. Scores conditioned on the hidden region answer a different question, so another rule can have lower loss within one slice.

## Redundant provider pools

`redundant_pool` draws a calibrated anchor probability *p* from a Beta(2, 2) distribution and a binary label from Bernoulli(*p*). In the null condition every other provider is generated from the anchor and independent randomness. Adding providers increases the number of candidate rules that could appear successful on a small fit split. The exact optimal forecast remains *p*.

| Knob | Default | Effect |
|---|---:|---|
| `n_candidates` | 15 | Number of non-anchor providers. The total number of providers is one larger |
| `positive_control` | `False` | Informs a single candidate with a noisy observation of the true label |
| `candidate_accuracy` | 0.80 | Reliability used to encode every vote. In the positive control it is the informed vote's actual accuracy |
| `role_seed` | 0 | Permutes anonymous provider columns. Keep fixed between fit and test, and vary across banks to randomize the layout |

Let *a* be vote accuracy and *p* the anchor. The probability of a genuinely informed vote being one given the anchor is `p × a + (1 − p) × (1 − a)`. Each null candidate independently draws a sham vote with that same probability. All candidates report the class-one probability obtained by updating the anchor as if the vote had accuracy *a*. Their score distributions conditional on the anchor therefore match across arms. In the positive control a single candidate receives the genuine vote. Its score equals the posterior given all observed columns. The other candidates are conditionally uninformative. With `candidate_accuracy=0.5`, every candidate reports the anchor. With accuracy 1, the informed candidate reveals the label.

```python
from fusionbench import get_scenario
import numpy as np

settings = dict(n_candidates=31, candidate_accuracy=0.8, role_seed=73)
null = get_scenario("redundant_pool", **settings)
informed = get_scenario("redundant_pool", positive_control=True, **settings)
a = null.build(n_per_split=256, seed=42)
b = informed.build(n_per_split=256, seed=42)
np.testing.assert_array_equal(a.test_y, b.test_y)
np.testing.assert_array_equal(a.test_oracle, a.test_preds[:, null.anchor_index])
np.testing.assert_array_equal(b.test_oracle, b.test_preds[:, informed.candidate_indices[0]])
```

Provider IDs are `model_0`, `model_1` and so on. The integer `anchor_index` and ordered `candidate_indices` describe the configured layout for evaluator-side checks. The learner receives only ordinary provider columns. The exact oracle assumes the fixed layout of the configured generator. Across pool sizes the original cases, labels, anchor and common candidates remain paired by their underlying roles. The numerical column indices can change with pool size or role seed.

[Scenario catalog](scenarios.md)

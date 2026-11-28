# Scenario catalog

Choose a scenario by the assumption you want to test. Each generator returns class probabilities from multiple providers for the same examples, together with class labels. The standard suite contains 30 scenarios, including 27 binary scenarios and three multiclass scenarios. The registry also includes the opt-in three-provider panel below. Use these IDs as registry keys in Python or as scenario names in a transfer configuration.

To select a disjoint group of standard scenarios or give different mechanisms different shares of an aggregate score, see [Choose and weight mechanism groups](mechanism-groups.md).

## Independent evidence and calibration

Start with `committee` to check whether fusion combines independent evidence. Use `deadweight` or `needle` to introduce irrelevant inputs, and `miscalibration` to preserve each provider's binary decision while changing its confidence.

| ID | Mechanism | Constructor defaults |
| --- | --- | --- |
| `committee` | Equally informative binary providers, independent conditional on the label. | `k=10` providers, `accuracy=0.75` per provider |
| `deadweight` | Informative providers mixed with label-independent scores concentrated near 0.5. | `n_good=4`, `n_bad=6`, `accuracy=0.82` for good providers |
| `needle` | The same construction with sparse signal among 100 providers. | `n_good=3`, `n_bad=97`, `accuracy=0.80` for good providers |
| `miscalibration` | Each provider's log odds are multiplied by a different fixed factor, evenly spaced from 0.5 to 2.5 at the default pool size. | `k=6`, `accuracy=0.78` before distortion |
| `multiclass_committee` | Independent class-logit noise with a boost for the correct class. | `k=8`, `n_classes=5`, `accuracy=0.60` |

For example, keep the sparse pool at 100 providers while increasing its informative fraction. Both splits below have 256 examples. The last axis contains the two class probabilities.

```python
from fusionbench.scenarios import SCENARIOS

pool = SCENARIOS["needle"](n_good=8, n_bad=92)
data = pool.build(n_per_split=256, seed=42)
assert data.fit_preds.shape == (256, 100, 2)
```

## Dependence and faults

| ID | Mechanism | Constructor defaults |
| --- | --- | --- |
| `similar_classes_38` | On true class 3 or 8, all three providers assign their largest incorrect probability to the other member of the pair (ties are possible). Other labels follow the ordinary provider panel. | 10 classes, shared pair active on fit and test, finite votes (31, 7) |
| `correlated` | One block shares Gaussian noise. The remaining providers have independent noise and the same marginal accuracy. | `n_block=8`, `n_diverse=2`, `accuracy=0.75`, `corr=0.85` within the block |
| `saturated` | Strong multiclass providers share most of their noise, leaving little independent evidence. | `k=6`, `n_classes=4`, `accuracy=0.95`, `corr=0.95` shared noise fraction |
| `calibrated_suppressor` | Two individually calibrated binary sources with unequal signal strengths. Their conditional correlation controls the weaker source's joint weight. | `corr=0.85`, strictly between −1 and 1 |
| `polarity_fault` | One of seven fixed-strength sensors occasionally reports its evidence with the wrong sign. The six stable sensors share noise. | `fault_rate=0.10` for the first sensor, `correlation=0.65` among the six stable sensors |

For a sign-change control, build `SCENARIOS["calibrated_suppressor"](corr=0.0)` and `SCENARIOS["calibrated_suppressor"](corr=0.85)`. The weaker source's population coefficient passes through zero at 0.5 for these fixed logit variances. For the two v1 multiclass scenarios, `accuracy` controls a heuristic logit boost. Measure realized top-1 accuracy from the generated labels.

## Three-provider multiclass panel

`multiclass_provider_panel` tests fusion when three providers differ in strength, confidence and reporting precision. It first draws a uniform class label and marks each example hard with probability `hard_fraction` (default 0.15). For each class, the three providers receive unit-variance Gaussian logit noise with pairwise correlation `shared_error` (0.65). Each provider adds `separation × strength` to the correct class's logit. Hard rows multiply that boost by `hard_scale` (0.55). The default strengths in provider order are `(1, 0.9, 0.8)`.

On a hard row with label `y`, provider `j` adds `confusion_bias` to class `(y + 1 + j % (C - 1)) % C`, using zero-based indices. A positive bias creates a systematic preference for a particular wrong class. Each provider divides its logits by its own temperature (defaults `(1.5, 1.0, 0.75)`) before softmax. Provider 0 reports the resulting probabilities. Providers 1 and 2 can draw independent multinomial votes and report observed class frequencies, including exact zeros. Each entry of `votes` sets that provider's count, with zero selecting continuous output.

These controls were specified independently of digits data. The default class count is 10 and any integer of at least 3 is supported. The higher separation is intended to produce strong providers. Realized accuracy and calibration depend on the class count and the other settings. The scenario supplies labels for scoring and omits an analytic posterior conditioned on reported scores.

| Setting | `separation` | `votes` | `confusion_bias` |
| --- | ---: | --- | ---: |
| A · Soft continuous | 2.5 | `(0, 0)` | 0 |
| B · High separation, finite votes (defaults) | 6.0 | `(31, 7)` | 0 |
| C · High separation, continuous | 6.0 | `(0, 0)` | 0 |
| D · Soft, finite votes | 2.5 | `(31, 7)` | 0 |
| E · High separation, finite votes, directed confusion | 6.0 | `(31, 7)` | 3.0 |

Keep the seed, class count and row counts fixed to pair the labels, difficulty flags and Gaussian draws across settings. Separate voting streams preserve the other providers when one count changes.

```python
from fusionbench import generate_dataset

options = dict(n_per_split=256, seed=42, n_classes=10, separation=6.0)
voted = generate_dataset("multiclass_provider_panel", **options, votes=(31, 7))
continuous = generate_dataset("multiclass_provider_panel", **options, votes=(0, 0))
assert (voted.test_y == continuous.test_y).all()
assert (voted.test_preds[:, 0] == continuous.test_preds[:, 0]).all()
```

```sh
fusionbench generate --scenarios multiclass_provider_panel --param n_classes=3 \
  --param 'votes=[31,7]' --param confusion_bias=3 --n 256 --seed 42 --out data
```

### Direct wrong-class preferences with paired controls

`shared_confusion_panel` takes the same generator settings and applies a class-identity intervention after all three providers report their probabilities. Supply a `shared_matrix` of shape `(C, C)`. Row `y` gives the relative weights of wrong destinations for examples with true class `y`. Weights must be finite and nonnegative, and the diagonal must be zero. A whole row of zeros leaves examples from that class untouched.

On each example and provider independently, `strength` sets the probability of applying the intervention. The generator draws a destination according to the row weights and swaps its probability with the largest probability among the wrong classes. It chooses uniformly among tied maxima. To allow differences between providers, pass `provider_matrices` of shape `(3, C, C)`. On each active draw, `deviation_mix` sets the probability of using that provider's weights. At zero all providers use the shared weights. At one each uses its own weights. Omitted provider weights reuse the shared matrix. The default `strength=0` exactly reproduces the original panel.

The example makes 3 and 8 each other's most favored wrong class. The private alternative sends both true classes toward wrong class 0, 1 or 2 according to the provider. These are illustrative class identities.

```python
import numpy as np
from fusionbench import generate_dataset

shared = np.zeros((10, 10))
shared[3, 8] = shared[8, 3] = 1
private = np.zeros((3, 10, 10))
for j in range(3):
    private[j, [3, 8], j] = 1

options = dict(seed=42, n_per_split=256, shared_matrix=shared,
               provider_matrices=private, strength=1)
common = generate_dataset("shared_confusion_panel", **options, deviation_mix=0)
per_provider = generate_dataset("shared_confusion_panel", **options, deviation_mix=1)
np.testing.assert_array_equal(common.fit_y, per_provider.fit_y)
```

For paired draws, keep the seed, split sizes and base panel settings fixed. The true-class probability and every wrong probability value are preserved for each example and provider. Consequently each provider retains its per-example Brier and log loss, its margins and its number of exact zeros. Fusion loss can change as providers put their wrong probability mass on different class identities. Shared preferences control the destination of errors. The inherited `shared_error` controls correlation in the original provider logits. When probabilities have exact top-1 ties, the class selected by lowest-index tie-breaking can change.

### Measure wrong-class preferences on fit rows

Call `runner_up_profile(bank.fit_preds, bank.fit_y)` on a generated or external bank. Supply a nonempty array shaped `(rows, providers, classes)`, with at least two classes and one integer label per row in `0..C-1`. Each probability must be finite and in `[0, 1]`, and each provider's class probabilities must sum to one within `1e-5`.

- `top_two_margin` averages the gap between the largest and second-largest probabilities across rows and providers. A tie for first contributes zero.
- `wrong_class_repeatability` measures whether a provider favors the same wrong classes across two groups of fit rows. Each row votes for the highest-probability incorrect class, sharing the vote equally across ties. This picks the runner-up when the true class wins, and a winning wrong class when it loses.

Within each true class, alternate rows in their supplied order between two halves. For each provider and half, add two votes spread uniformly over the wrong classes and subtract the uniform distribution from the resulting frequencies. Take the dot product of those deviations across halves. Average equally over providers and true classes with at least two rows, divide by `1 - 1/(C - 1)`, then clip to `[0, 1]`. Values near one indicate a consistent, concentrated preference. The two added votes dampen results for small classes. Binary banks and banks with no repeated true class return zero for repeatability.

Keep class identities aligned across rows. Applying one permutation to every probability vector and relabeling the targets accordingly preserves both scores. Sorting each row's probabilities independently discards the identity of the preferred wrong class.

```python
import numpy as np
from fusionbench.eval.confusion_profile import runner_up_profile

# One provider prefers class 1 among the wrong classes on all four rows.
p = np.array([[0.65, 0.30, 0.05], [0.30, 0.55, 0.15]] * 2)[:, None, :]
profile = runner_up_profile(p, np.array([0, 0, 0, 0]))
np.testing.assert_allclose(profile["top_two_margin"], 0.30)
np.testing.assert_allclose(profile["wrong_class_repeatability"], 0.25)
```

## Hidden conditions and specialists

The six Boolean tasks draw independent, balanced latent bits. A detector estimates its own condition, and the Boolean rule gives the observed label. Here `rep` counts independent noisy detectors **per condition**. Their accuracy refers to the latent condition. Increasing the number of conditions in an AND or OR also changes the frequency of the positive class.

| ID | Label or routing rule | Constructor defaults |
| --- | --- | --- |
| `hardveto` | Presence AND safety. One detector group measures each. | `acc_detect=0.85`, `acc_veto=0.88` for safety, `rep=3` |
| `hardrescue` | Any one of the latent conditions can make the label positive. | `n_detect=3` conditions, `accuracy=0.88`, `rep=2` |
| `weakest_link` | Every latent condition must hold. | `m=4` conditions, `accuracy=0.85`, `rep=2` |
| `two_pathways` | (A AND B) OR (C AND D). Four conditions and their pairings are fixed. | `accuracy=0.90`, `rep=3` |
| `parity` | Exactly one of two latent bits is true (XOR). | `accuracy=0.92`, `rep=2` |
| `compound` | (Core OR rescue) AND safety. Each group has its own detector strength. | `acc_core=0.78`, `acc_rescue=0.90`, `acc_safe=0.90`, `rep=3` |
| `gated_experts` | A hidden region chooses the informative specialist. The other specialist tracks an independent distractor. A third provider estimates the region. | `acc_select=0.92` for the region detector, `acc_expert=0.90` for each specialist on its active region |

Use `SCENARIOS["parity"](rep=1)` to test the XOR interaction with only two inputs. To test sensitivity to identifying the active specialist, use `SCENARIOS["gated_experts"](acc_select=0.65)` and keep expert accuracy at its default.

## Label laws defined directly on provider scores

These three scenarios draw six anonymous discrete probability scores, compute a positive-class probability from selected columns, then draw a Bernoulli label. The default operators deliberately match particular fusion trees. A win in these cases measures the ability to recover that structure under the chosen sampling law.

| ID | Positive-class probability at the default | Constructor defaults |
| --- | --- | --- |
| `native_switch` | Of two designated scores, return the one farther from the fixed cutoff 27/64. Ties choose the first operand. | `mixing=0.0`, `role_seed=0` |
| `native_close_switch` | Return the score closer to the fixed cutoff 37/64, with the same tie rule. | `mixing=0.0`, `role_seed=0` |
| `compositional_switch` | Return `max(min(a, b), min(c, d))` on four designated scores. | `law="composition"`, `role_seed=0`, `corner_probability=0.5` |

For FAR and CLOSE, `mixing` is the weight of an affine-logit probability in the label law. At 0 the gate supplies the entire probability and at 1 the logit rule does. For the composition task, `law="logit"` uses the first two designated columns. With probability `corner_probability` all six scores are drawn from the endpoints 1/64 and 63/64. Other rows draw each score from all 63 ticks. In all three families `role_seed` assigns columns to roles, while the `seed` supplied to `build` draws examples. The cutoffs and the six-column layout are fixed by the constructors.

```python
from fusionbench.scenarios import SCENARIOS

build = SCENARIOS["compositional_switch"]
kwargs = dict(role_seed=17, corner_probability=0.0)
composition = build(law="composition", **kwargs).build(n_per_split=64, seed=9)
logistic = build(law="logit", **kwargs).build(n_per_split=64, seed=9)
assert (composition.test_preds == logistic.test_preds).all()
```

This comparison holds provider inputs and roles fixed while changing the conditional label law.

## Reading the oracle

`Dataset.test_oracle`, when available, is a row-by-class reference under the generating law. The independent binary committee uses a sum of log odds under balanced class priors. The correlated Gaussian model accounts for its covariance, and the logical tasks combine evidence using the known latent grouping. For the two v1 multiclass models the reference uses the generator's logit strength and, for Saturated, its shared noise. Keep the oracle and known roles on the evaluation side. The three-provider panel and custom scenarios may return `None`. After provider corruption, the appropriate reference would condition on the corrupted observations. [Version-1 generator implementations](https://github.com/damiankucharski/fusionbench/tree/05fa4ec364e545671402ec1ff14392b818fc4d6a/fusionbench/scenarios)

## Four standard-v3 scenarios

| ID | Controlled mechanism | Evaluation question |
| --- | --- | --- |
| `rare_specialist` | A specialist supplies additional information in a rare region. A third input gives a noisy cue about the region. | Does a learned rule exploit the specialist on rare examples without degrading performance elsewhere? |
| `redundant_pool` | An optimal anchor is accompanied by noisy alternatives. A positive control gives one candidate genuine additional evidence. | How much does searching many candidates cost when the anchor is already sufficient? |
| `environment_shortcut` | The relation between a shortcut and the label depends on the environment. The population mixture changes between fitting and deployment. | Does validation reward a provider whose apparent reliability changes at deployment? |
| `selective_fallback` | A provider falls back to a neutral score, with the event associated with example difficulty. | Does fusion use the remaining evidence on cases where a source loses resolution? |

[Generate rare specialists and redundant pools](rare-and-redundant.md) · [Control environments and fallback](environments-and-fallback.md) · [Run mechanism stress profiles](stress-profiles.md) · [Change only the class prior](prior-shift.md)

For a nested size comparison, `fusionbench.interventions.paired_pool_sizes("needle", [10, 97], n_per_split=512, seed=7)` returns two banks with identical rows, informative providers and oracle. For committee scenarios the size counts independent inputs; for needle and deadweight it counts irrelevant inputs. All sizes are prefixes of the same maximum-size bank. For correlated and saturated pools, `duplicate_provider(dataset, provider=0, copies=4)` in the same module creates a paired control with identical information and posterior. Each additional copy receives a distinct provider ID.

## Cloned shortcut in standard-v4

`cloned_shortcut` lets you vary how many independent signals an ensemble receives and whether those signals still point in the same direction at deployment. Two stable providers vote correctly with probability 0.75 in both domains. Six shortcut providers each vote correctly with probability 0.80 during fitting. Every provider reports its source confidence, including on test. Labels are balanced in expectation.

| Setting | `clone_probability` | `test_reversal_probability` |
| --- | ---: | ---: |
| Independent, aligned votes | 0 | 0 |
| Cloning alone | 1 | 0 |
| Reversal alone | 0 | 1 |
| Both mechanisms (default) | 1 | 1 |

On a cloned row all six shortcut reports copy the first shortcut's vote, so their agreement supplies one independent piece of evidence. On a reversed test row every shortcut reports the opposite vote while retaining its confidence. The stable providers keep their independent evidence. Each knob also accepts intermediate probabilities. The hidden events are independent Bernoulli draws per row.

```python
from itertools import product

import numpy as np

from fusionbench import generate_dataset

banks = {
    (c, r): generate_dataset(
        "cloned_shortcut", clone_probability=c, test_reversal_probability=r,
        n_per_split=512, seed=17,
    )
    for c, r in product((0, 1), repeat=2)
}
clean = banks[0, 0]
for bank in banks.values():
    np.testing.assert_array_equal(bank.test_y, clean.test_y)
    np.testing.assert_array_equal(bank.test_preds[:, :2], clean.test_preds[:, :2])
np.testing.assert_array_equal(banks[1, 0].fit_preds, banks[1, 1].fit_preds)
```

The matched seed keeps labels, underlying votes and stable reports paired across all four settings. Fit and test in each bank use separate rows. The oracle uses the eight reported votes and the known domain law, integrating over the hidden events. At complete reversal it counts an inverted shortcut vote as positive evidence for the opposite class. The source rows alone cannot identify whether this reversal will occur at deployment. Group flags for evaluating subsets are in `bank.meta["evaluation_groups"]`. [Generate and save datasets](generate.md)

## Rare specialist, shortcut and noise in standard-v5

`rare_shortcut_noise` has four binary providers. A generalist votes correctly on 75% of cases before corruption. A specialist votes correctly on 95% inside its region and 50% outside. A routing cue identifies the region. An active shortcut votes correctly on 90% of cases, while an inactive shortcut emits a fair vote with the same reported confidence. The label and provider noise controls use independent draws. All eight settings accept probabilities from zero to one.

| Mechanism | Fit knob and default | Test knob and default | Meaning |
| --- | --- | --- | --- |
| Region prevalence | `region_fit=0.05` | `region_test=0.50` | Probability of drawing a case from the specialist's region |
| Shortcut availability | `shortcut_fit=1` | `shortcut_test=0` | Probability that the shortcut is active on a row |
| Provider noise | `provider_noise_fit=0` | `provider_noise_test=0.20` | Probability of independently flipping each generalist or specialist vote |
| Label noise | `label_noise_fit=0` | `label_noise_test=0` | Probability of flipping the observed label after generating the reports |

The following comparison uses the same random draws to isolate the combined deployment change. The source-like bank retains the fit settings at test. Both banks have exactly the same fit rows.

```python
import numpy as np
from fusionbench import generate_dataset, load_reference_dataset

options = {"n_per_split": 512, "seed": 41}
shifted = generate_dataset("rare_shortcut_noise", **options)
source_like = generate_dataset(
    "rare_shortcut_noise", region_test=0.05, shortcut_test=1,
    provider_noise_test=0, **options,
)
np.testing.assert_array_equal(shifted.fit_preds, source_like.fit_preds)
np.testing.assert_array_equal(shifted.fit_y, source_like.fit_y)
reference = load_reference_dataset("rare_shortcut_noise")
```

`fit_oracle` and `test_oracle` give the exact probability of the **observed** label given the four reports under the corresponding split law. For an exact finite-support calculation, `get_scenario("rare_shortcut_noise").support("test")` returns every report vector, its mass and the corresponding positive-class posterior. The region membership for subgroup scoring is available in dataset metadata under `evaluation_groups`.

The standard data generator uses the fit law for fitting and the test law for held-out evaluation. In the optional transfer grid, the clean condition instead draws fresh rows from the fit law. Canonical-shift severity 1 uses the full test law. Intermediate severity selects complete rows from a mixture of these two domains. The benchmark varies the region prevalence, shortcut and noise independently to measure which changes transfer. [Factorial experiment](https://github.com/damiankucharski/fusionbench/tree/research/ensemble-mechanisms-stability/docs/ensemble-study/transfer-factorial) · [Transfer-grid settings](transfer.md)

### Multiple labelled source environments

The same rare-shortcut scenario can generate several training environments for a single fusion model. A source environment is an observed batch, such as an acquisition site. Changing both case mix and shortcut activity can make a useful specialist appear unreliable across sites. The following named source profiles vary these factors independently.

| Profile | Specialist-region prevalence at sites 0 / 1 | Fraction of active shortcut cases at sites 0 / 1 |
|---|---:|---:|
| `homogeneous` | .50 / .50 | .50 / .50 |
| `varied` | .50 / .50 | .00 / 1.00 |
| `case_mix` | .15 / .85 | .50 / .50 |
| `mixed` | .15 / .85 | .00 / 1.00 |

```python
import numpy as np
from fusionbench.scenarios.source_environments import source_environment_preset

bank = source_environment_preset("mixed", n_rows=512, seed=7)
assert np.bincount(bank.env).tolist() == [256, 256]
assert bank.x.shape == (512, 4, 2)
test = bank.target(region=.5, shortcut=0)

# Exact population Brier for the arithmetic mean across the four providers.
p = test.x[:, :, 1].mean(axis=1)
loss = 2 * (test.mass @ ((p - test.q) ** 2 + test.q * (1 - test.q)))
```

Fitting may use reports `x`, class labels `y`, the source identifiers `env`, and the observed per-row cue `context`. The simulator exposes provider role indices and the exact target probabilities for scoring. Keep these outside fitting and configuration. Reusing a seed and row counts pairs the labels, random draws, and provider ordering across profiles. The two-site `mixed` profile and its equal four-site subdivision share all observations when each original site is split into two equal blocks at a quarter of the total row count.

For a custom design, import `source_environment_bank` from the same module and pass parallel sequences `region` and `shortcut`. Pass a total `n_rows` for balanced site counts, or a sequence for explicit counts. The default normalizes every provider's confidence to .8 while preserving its vote. Set `original_scale=True` to restore the original report magnitudes. Set `context_noise` to the desired probability of flipping the supplied source context bit. This affects only that side channel. The cue provider's reports and the exact test population stay fixed.

## Same-population stratified pool in standard-v5

`stratified_pool` emits eight binary forecasts with independent, equally likely signs before stratification. On each row all eight positive-class probabilities are either 0.5 ± 0.15 or 0.5 ± 0.35. The probability of a positive label is a weighted average of the arithmetic mean and the first provider. The parameter `specialization` controls the first provider's weight in this mixture (default 0.25, range 0–1). At zero the mean is the exact posterior. At one, the first provider is the exact posterior.

Fit and test use the same distribution. Both splits balance the four strata formed by report amplitude and realized label. When the sample size is not divisible by four, a random subset of strata receives one extra observation each. This lets you vary training size and specialization while holding the population and its composition fixed. The row-level group `high_amplitude` is saved for evaluation. The 512 report vectors have equal population mass, so methods can also be scored against their exact population Brier loss.

```python
from fusionbench import generate_dataset, get_scenario

bank = generate_dataset("stratified_pool", specialization=0.25, n_per_split=512, seed=7)
reports, mass, positive_probability = get_scenario("stratified_pool").support()
assert reports.shape == (512, 8, 2)
```

The [matched-population study and directional transfer experiment](https://github.com/damiankucharski/fusionbench/tree/research/ensemble-mechanisms-stability/docs/ensemble-study/transfer-factorial) use these two standard-v5 scenarios.

## Rare events in correlated scenes (standard-v6)

Three ready-made IDs appear in the standard suite and use the same configurable generator.

| Scenario | Mechanism at its default settings |
| --- | --- |
| `grouped_rare_event` | Provider accuracy differs and all eight providers share some error. A specialist is active in 20% of scenes. |
| `shared_error_bloc` | Six of eight equally accurate providers share 85% of their error variance. The other two have independent errors. Specialization is off. |
| `rare_scene_specialist` | All eight providers are equally noisy outside the specialist region. One reports the exact probability in the active 20% of scenes. There is no shared residual across providers. |

Use `grouped_rare_event` to ask what a fusion method can learn when the training set covers few independent scenes. Eight anonymous providers estimate a binary event probability. Observations from one scene share a signal and a component of each provider's error. Fit and test contain independently drawn scenes from the same population. Scene identifiers are local to each split. The default configuration yields about 10% positives in expectation; a small training set can have a markedly different event rate.

| Parameter | Default | Effect |
| --- | ---: | --- |
| `provider_signal_slope` | 1.3 | Positive finite coefficient of the latent signal in ordinary provider logits. The true event logit uses slope 1.8. |
| `provider_order_seed` | None | A nonnegative integer permutes the anonymous provider columns. The order depends only on this seed and is fixed across train and test. The default retains the original order. |
| `scene_size` | 8 | Positive integer count of rows per scene. The last scene may contain fewer rows. |
| `event_offset` | 2 | Subtracted from the latent signal; larger values make events rarer. |
| `heterogeneity` | 0.5 | Changes the ratio of the largest to smallest provider residual standard deviation from 1 at zero to exp(2) at one. Average pre-specialist residual variance is fixed. |
| `common_error_share` | 0.35 | Fraction of a provider's standardized residual variance shared across providers. |
| `bloc_size` | 8 | Number of providers participating in the shared residual, starting with provider zero. The other providers draw independent residuals. Accepts integers from zero through eight. |
| `scene_error_share` | 0.85 | Fraction of this residual variance persistent across observations of a scene. |
| `specialist_rate` | 0.2 | Probability that a scene activates the specialist, independently of its signal. |
| `specialist_strength` | 0.75 | Interpolates provider 3's logit toward the true event logit in active scenes. At one this provider reports the exact probability in those scenes. |

Each row has latent signal `z = 0.9 * scene_signal + 0.8 * row_signal - event_offset`, using independent standard normal signals. The event probability is `expit(1.8*z)`. Ordinary provider logits use `provider_signal_slope*z` plus Gaussian error. Noise controls range from zero to one. At a fixed seed, row count and scene size, changing provider properties preserves the same labels, signals and underlying noise draws. These switches let you compare the loss due to a common provider error with the benefit of a local specialist on the very same events.

```python
from fusionbench import generate_dataset, get_scenario, load_reference_dataset

bank = generate_dataset(
    "grouped_rare_event", seed=34, n_per_split=128,
    scene_size=8, common_error_share=0.7, specialist_strength=1.0,
)
scenes = bank.meta["evaluation_groups"]["fit"]["scene_id"]
specialist_active = bank.meta["evaluation_groups"]["test"]["specialist_region"]
reference = load_reference_dataset("grouped_rare_event")
```

To compare how providers scale their evidence, vary the provider slope and keep the data seed fixed:

```python
slope_banks = {
    slope: generate_dataset(
        "rare_scene_specialist", provider_signal_slope=slope,
        seed=34, n_per_split=128,
    )
    for slope in (1.3, 1.8, 2.3)
}
```

A slope of 1.8 gives ordinary provider logits the same signal coefficient as the true logit. The providers still carry residual errors. At 2.3 their logits respond more strongly to the latent signal. All three settings share labels, residual draws and active scenes. With specialist strength one, the specialist reports the same true probability inside its active scenes at every slope.

To check sensitivity to column order, keep the data seed and all other knobs fixed and vary `provider_order_seed`. The permutation applies to both splits. With `sc = get_scenario("shared_error_bloc", provider_order_seed=7)`, `sc.provider_order[j]` gives the latent role behind output column `j`. The returned tuple is immutable. Model IDs remain `provider_0` through `provider_7` in output order. A method that receives only the reports cannot read the latent role from these IDs.

The prediction tensor has shape (rows, 8, 2). Scene labels, context and latent probability are available for evaluation. The generator's `sample()` method returns `latent["posterior"]`, the probability conditional on the underlying signal. The posterior conditional on the reports is generally different, so the dataset leaves `fit_oracle` and `test_oracle` empty. This control models within-scene dependence without reproducing image geometry or a particular sensor. [Evaluating grouped data](validation-curse.md)

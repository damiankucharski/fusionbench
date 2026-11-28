# Locate overfitting during fusion search

A high validation score can arise because a particular rule fitted well, or because the search compared many rules on a small common set of cases. FusionBench offers two diagnostics. A fixed-menu audit measures optimism from repeated validation comparisons. A population trace records the individual programs surviving each stage of an actual search. These measurements answer different questions.

## Fixed-menu audit

`fusionbench.eval.validation_curse.audit_menus` takes finite row-by-candidate loss matrices from separate validation, same-law audit, IID test and shifted test sets. Column zero is a prespecified baseline. Keep candidate order fixed, fit candidates independently of all four matrices and construct disjoint splits at the level of independent examples. Specify prefixes to compare larger menus and validation samples on common cases. For correlated observations such as multiple samples from one acquisition, aggregate paired losses per independent unit for a unit-weighted evaluation.

```python
from fusionbench.eval.validation_curse import audit_menus

# Each column corresponds to the same frozen predictor on all four splits.
cells = audit_menus(validation_losses, audit_losses, test_losses, shift_losses,
                    menu_sizes=[1, 4, 16], validation_sizes=[32, 128])
```

Provide `population_risk` only when the generating law supplies it. On real held-out samples report empirical test loss. Program choices in adaptive GP also depend on earlier generations, so a fixed menu alone cannot identify when useful candidates disappear. [Finite-menu example and independent replay](https://github.com/damiankucharski/fusionbench/tree/research/ensemble-mechanisms-stability/docs/ensemble-study/validation-curse/round1)

## Trace the population

Record the active population as well as the cumulative archive. For each checkpoint compare its empirical champion, typical survivor and the best available rule. Choose a retrospective winner on an independent audit and score it on a separate test, or integrate expected loss when the generating posterior is known. Keep all audit scores outside parent selection, mutation and survival. A useful positive control inserts a known correct candidate at initialization and tracks whether it survives. [Native trajectory audit](https://github.com/damiankucharski/fusionbench/tree/research/ensemble-mechanisms-stability/docs/ensemble-study/validation-curse/trajectory) · [Clustered multiclass example](https://github.com/damiankucharski/fusionbench/tree/research/ensemble-mechanisms-stability/docs/ensemble-study/validation-curse/grouped)

## Evaluate a conservative training objective

`fusionbench run --methods okapi_search_guard --seed 1729 --n 640 --out results/search-guard` runs the opt-in native training objective over all standard scenarios with averaging, ranked-prefix Top-N and probability stacking. The objective penalizes uncertainty of paired improvement over averaging during GP and parameter fitting. The existing discovery/validation partition and final validation rule are retained. In Python, `SearchGuardedNativeOKAPI(z=0)` gives the incumbent control and `z=1` runs the experiment. This FusionBench adapter treats rows as independent. The native OKAPI primitive also supports group identifiers for clustered losses. [OKAPI usage](https://github.com/damiankucharski/OKAPI/blob/64ee67e/docs/api/core.md)

## Separate breeding from deployment

The replacement experiment runs six native generations of twelve offspring. The discovery set determines breeding fitness. A fresh, independent block of labels in each generation determines which tree can replace the incumbent. In the open variant, offspring can also breed before they have earned a positive lower bound on their Brier gain. The strict variant requires that bound for admission to breeding. Both deploy a single native tree.

```python
from fusionbench.eval.replacement_search import run_replacement_search
from fusionbench.eval.suite_point_gp import predict_tree

# Each (X, y) block contains distinct independent examples from development data.
result = run_replacement_search(
    X_discovery, y_discovery, six_confirmation_blocks,
    seed=17, policy="open_kl", incumbent="best_single",
)
probabilities = predict_tree(result.tree, X_test)
```

A comparison with `open_raw` measures the effect of using a confidence bound for promotion on the same proposal trajectory. Comparing `open_kl` with `strict_kl` measures how admitting unconfirmed offspring changes exploration and eventual deployment. Record the label budget and the final tree's provider dependencies. The confidence calculation assumes a fresh independent block at each step. [Protocol, controls and measured outcomes](https://github.com/damiankucharski/fusionbench/tree/research/ensemble-mechanisms-stability/docs/ensemble-study/replacement-experiment)

## Reuse scenes for choosing provider checkpoints

A staged checkpoint bank exposes the observations available *before* choosing which prediction column to keep for each provider. Generate two independent source blocks (A and B) and a fresh audit (I). A and B each contain 24 scenes of eight observations. The audit contains 256 new scenes. All three obey the same law. Each provider has eight correlated checkpoint alternatives with equal marginal competence.

```python
import numpy as np
from fusionbench.scenarios.checkpoint_bank import generate_checkpoint_bank, select_checkpoints

bank = generate_checkpoint_bank(seed=42)
slots = select_checkpoints(bank["A"], k=8)
providers = np.arange(len(slots))

def frozen_inputs(block):
    positive = block.P[:, providers, slots]
    return np.stack([1.0 - positive, positive], axis=-1)

reused_fit = frozen_inputs(bank["A"])
independent_fit = frozen_inputs(bank["B"])
fresh_test = frozen_inputs(bank["I"])
```

Fit one copy of a fusion method on each source block and evaluate both on the same fresh scenes. Then exchange A and B and repeat the comparison. Keep all eight observations from a scene together when creating independent blocks. Setting `k=1` removes checkpoint selection while preserving the predictions of the first slot. This staged protocol complements the 24 standard-v3 fixed-provider scenarios. [Paired study, baselines and replay](https://github.com/damiankucharski/fusionbench/tree/research/ensemble-mechanisms-stability/docs/ensemble-study/checkpoint-reuse)

### Give one provider the true conditional probability

Use the paired law `generate_checkpoint_bank(seed, law="signal")` to test whether fusion can leave averaging when one provider is more informative. It preserves the labels, scenes and checkpoint outputs of the other seven providers generated with the same seed under `law="equal"`. The designated provider returns the true conditional probability for every checkpoint. Freeze one column per provider before passing inputs to a learner.

Every block exposes `posterior` for evaluation. For binary full-class Brier, the conditional excess loss of a frozen positive-class prediction `p` on the audit inputs is `2 * np.mean((p - bank["I"].posterior) ** 2)`. This averages over possible labels while retaining the finite audit scene sample. On the equal law the posterior uses a latent signal. On the signal law an observable provider attains it. Withhold the posterior and provider identity from model fitting. [Mean founder and positive-control experiment](https://github.com/damiankucharski/fusionbench/tree/research/ensemble-mechanisms-stability/docs/ensemble-study/seeded-checkpoint)

### Change the number of independent scenes at a fixed row budget

A scene is a group of observations that share part of their underlying signal and provider errors. `generate_scene_pool` draws separate scenes for checkpoint selection (`A`), fusion training (`F`) and evaluation (`I`). Each scene has eight observations. The fit and audit pool sizes are configurable. Growing a pool preserves its previous rows, labels and scene identifiers.

```python
import numpy as np
from fusionbench.scenarios.checkpoint_bank import (
    generate_scene_pool, scene_sample_indices, select_checkpoints,
)

pool = generate_scene_pool(42, law="signal", n_fit_scenes=1536, n_audit_scenes=512)
slots = select_checkpoints(pool["A"], 8)
providers = np.arange(len(slots))
fit_p = pool["F"].P[:, providers, slots]  # positive probabilities, one column per model
fit_y = pool["F"].y
audit_p = pool["I"].P[:, providers, slots]

for scenes, rows_per_scene in [(24, 8), (192, 1), (192, 8), (1536, 1)]:
    rows = scene_sample_indices(scenes, rows_per_scene)
    train_p, train_y = fit_p[rows], fit_y[rows]
    assert len(np.unique(pool["F"].scene[rows])) == scenes
    # Fit a fresh fusion rule to train_p and train_y, then score it on audit_p.
```

The first two designs each contain 192 training rows. Comparing them estimates the effect of sharing scenes under this generator. The middle two designs contain exactly the same 192 scenes. Comparing them measures the value of adding seven observations per scene. For a new fusion method, use the same checkpoint choices and audit inputs across designs and estimate uncertainty over independently generated pool seeds. [Scene-count study](https://github.com/damiankucharski/fusionbench/tree/research/ensemble-mechanisms-stability/docs/ensemble-study/scene-count)

### Vary the provider error shared by observations from one scene

Use `generate_scene_pool(42, law="signal", scene_error_share=0.5)` to place half of each provider's residual variance at scene level. Values from zero to one preserve the distribution of the provider's checkpoint predictions for a single row. They change how much error persists across the other rows in the same scene. The scene's influence on the event probability remains present at zero. `None` uses the original coefficients, which allocate different shares to static errors and checkpoint-specific errors.

For a paired experiment, use the same seed and freeze the selected checkpoint columns before changing the share. Compare training on one observation per scene with training on several observations from each scene. The variance of the difference in losses for a specific pair of rules can respond differently from the test loss of an adaptive fusion procedure. [Paired study and data](https://github.com/damiankucharski/fusionbench/tree/research/ensemble-mechanisms-stability/docs/ensemble-study/scene-dependence)

### Check whether a candidate improves both the pooled loss and several folds

The four-fold study compares candidate trees with the same initial reference tree. A candidate can improve in three folds and still worsen the pooled loss when it loses heavily on the fourth. The optional OKAPI helper exposes both gains before rounding. To test a terminal rule, evolve one population, keep its final survivors fixed and compare three choices: take the largest fold gain, return the anchor when that winner fails the pooled check, or reconsider the remaining eligible survivors. To test the effect on breeding, rerun the evolution with the smaller of the two gains as its fitness and keep the final rule fixed. A known correct rule that is absent from the initial population makes it possible to measure whether a stricter rule blocks a necessary change. [Three-series experiment and reconstruction code](https://github.com/damiankucharski/fusionbench/tree/research/ensemble-mechanisms-stability/docs/ensemble-study/joint-gain)

### Check whether search ever proposes the correct rule

With a FusionBench generator whose posterior is a known function of two providers, record three events separately. Did the search score the correct rule? Did it survive? Did fit-set ranking select it? A matched single-provider control measures the cost of offering irrelevant pairs. On the full benchmark, give the pair-proposal arm and ordinary evolution the same number of scored offspring and compare both fit and test Brier. In one 25-family study, pair proposals lowered mean test loss by 0.01593 while the test-minus-fit gap went from 0.03302 to 0.03450. Better search coverage and resistance to optimistic validation can therefore be measured separately. [Protocols, scenarios and paired results](https://github.com/damiankucharski/fusionbench/tree/research/ensemble-mechanisms-stability/docs/ensemble-study/pair-coverage)

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

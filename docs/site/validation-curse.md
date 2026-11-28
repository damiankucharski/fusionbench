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

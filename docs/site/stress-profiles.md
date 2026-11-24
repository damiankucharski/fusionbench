# Run mechanism stress profiles

Every standard scenario has a `reference` profile using its constructor defaults and two named alternatives. A full run evaluates 72 settings across 24 scenarios. Each method learns from the same rows within a setting and receives the same held-out provider probabilities. Full-class Brier is the average sum of squared probability errors per row. Lower values are better. [Scenario definitions](scenarios.md)

```sh
fusionbench profiles --list --scenario weakest_link similar_classes_38
fusionbench profiles --scenario weakest_link similar_classes_38 \
  --fit-rows 640 --test-rows 2048 --seeds 11 12 13 --search-seed 5 \
  --out results/mechanism-profiles
```

Omit `--scenario` to evaluate all standard families. Choose a subset of settings using `--profile reference` or a name printed by `--list`. The default comparison fits the regularized native OKAPI tree, arithmetic mean, Brier-ranked Top-N and logistic regression on probabilities. Top-N evaluates all ranked prefixes on the fit set. Requesting OKAPI includes these three controls. For a run using the base installation, pass `--methods simple_average top_n stacking_lr`. The default has 384 fit rows, 1,024 test rows, data seed 0 and search seed 0. If a rare positive class is absent throughout a binary training bank, logistic regression requires a larger fit sample or a different seed. An empty evaluation subgroup has count zero and a blank loss.

## What changes

| Scenario | Alternatives to the reference |
| --- | --- |
| `committee`, `multiclass_committee` | Weaken each provider at fixed pool size, or retain three providers at the original strength. |
| `deadweight`, `needle` | Vary the number of irrelevant providers while retaining the original informative sources. |
| `correlated`, `saturated` | Move to independent noise or strong dependence. Saturated's correlation-1 setting yields exactly identical probability vectors. |
| `calibrated_suppressor` | Correlations 0.15 and 0.50 give a positive and a zero coefficient for the weaker provider, respectively. The reference gives it a negative coefficient. |
| `polarity_fault`, `miscalibration` | Compare zero or frequent sign flips in the former and identity or half-strength logit distortions in the latter. |
| `hardveto`, `hardrescue`, `two_pathways`, `compound` | Change how often the decisive branch occurs within its eligible label class, holding the target population prevalence fixed. |
| `weakest_link` | Compare six conditions at the four-condition reference prevalence of 1/16, or lower prevalence to 0.02 at the reference depth of four. Both changes retain the reference share of negatives with exactly one unsatisfied condition (4/15). |
| `parity`, `gated_experts` | For parity, lower the second bit detector's strength. For gated experts, make the selector nearly uninformative or weaken the specialists. |
| `native_switch`, `native_close_switch` | Vary the population fraction with an absolute gap of at least 0.25 between gate operands, holding the gate fixed. |
| `compositional_switch` | Draw all rows from the full grid, or force all scores onto the two endpoints. |
| `similar_classes_38` | The reference has a shared 3↔8 preference. One alternative mixes shared and provider-private destinations equally. The other uses private destinations throughout. Each provider's true-label probability and per-row Brier stay fixed across arms. |
| `rare_specialist` | Reduce region prevalence from 0.10 to 0.025, or make its routing cue uninformative at reference prevalence. |
| `redundant_pool` | Increase the null pool from 15 to 63 candidates, or give one of the original 15 an informative signal. The fixed role seed keeps the provider order reproducible. |
| `environment_shortcut` | Retain the training environment mixture at test, or neutralize the shortcut while retaining the default mixture shift. |
| `selective_fallback` | Compare difficulty-dependent losses of availability with the same quota allocated randomly, or keep the provider available for all rows. |

## Reading the outputs

`cells.csv` records the settings, seeds, provider count, observed label counts, fit and test losses, and the observed-input oracle's test loss where available. `groups.csv` records counts and conditional test loss for the applicable diagnostic groups, including decisive cases, sign flips, rare regions, environment, fallback and classes 3/8. A method's aggregate loss includes every test example once at its sampled frequency.

`by_profile.csv` averages repeated data seeds within each setting. `by_scenario.csv` averages those settings within a scenario. `macro.csv` then gives each included scenario equal weight. Each summary panel identifies whether it covers reference settings, stress settings, or all selected settings. `paired.csv` reports OKAPI minus each control in test Brier, with a two-sided 95% t interval across the per-seed macro differences when at least two data seeds are supplied. The interval measures variability across generated banks for the fixed scenarios and chosen search seed.

The manifest records profile version, exact constructor overrides, fit/test sizes and RNG streams. Fit and test use separate child seeds. Each setting reuses a given data seed. The pool-size comparisons preserve row labels and shared provider prefixes on both splits. Some controlled Boolean settings change the sampling law and can produce different latent cases and labels from the historical reference. Set `--search-seed` separately to vary OKAPI's search while keeping its discovery/validation split fixed by the data seed.

```python
from fusionbench.profiles import list_profiles, run_profiles

settings = list_profiles("parity")
tables = run_profiles(
    scenarios=["parity"], seeds=[11, 12],
    methods=["simple_average", "top_n", "stacking_lr"],
)
print(tables["by_profile"][["profile", "method", "mean_brier"]])
```

# Group and score scenarios

The `mechanisms-v1` taxonomy assigns the 30 `standard-v6` generators to seven disjoint mechanism groups. Each group appears in the [catalog](scenarios.md). A separate set of purpose tags can overlap. Choose the score that answers the scientific question before examining method results.

| Summary | What receives equal weight? |
| --- | --- |
| Standard macro | All 30 scenario means. This preserves comparisons with the same standard panel. |
| One mechanism | Members of that group. For example, `context_routing` covers `gated_experts`, `rare_specialist` and `selective_fallback`. |
| Equal-group macro | Each of the seven groups. Within each group its members share the weight equally. |
| Purpose tag | All scenarios carrying the selected tag. `capability_probe` tests access to a planted rule, `operational_stress` captures a recognizable practical difficulty, and `reference` selects comparison cases. |

A purpose filter takes the union of the requested tags. A scenario with both tags appears once. Supplying both a group filter and a purpose filter takes their intersection.

## Choose explicit priorities

```python
from fusionbench.taxonomy import (
    APPLICATION_PROFILES, MECHANISM_GROUPS, scenario_weights,
)

standard = scenario_weights()
rules = scenario_weights(groups="structured_rules")
stress = scenario_weights(purposes="operational_stress")
equal_groups = scenario_weights(group_weights=dict.fromkeys(MECHANISM_GROUPS, 1))
multi_site = scenario_weights(group_weights=APPLICATION_PROFILES["multi_site"])

assert len(standard) == 30 and len(rules) == 9 and len(multi_site) == 9
print(multi_site)
```

The illustrative `multi_site` profile assigns 50% to transfer shift, 30% to grouped sampling and 20% to context routing. `rule_design` assigns 75% to structured rules and 25% to context routing. These shares describe evaluator priorities. Estimating the prevalence of each mechanism in an application requires measurements from that application.

## Aggregate an evaluation

First average the chosen metric across seeds within each scenario for each method. Then pass those means and a named weight mapping to `weighted_score`. All methods should use identical banks and the same profile. Missing scores on a positively weighted scenario raise an error.

```python
from fusionbench import evaluate_dataset, load_reference_dataset
from fusionbench.methods.averaging import SimpleAverage
from fusionbench.taxonomy import scenario_weights, weighted_score

weights = scenario_weights(groups="context_routing")
means = {}
for scenario in weights:
    scores = evaluate_dataset(load_reference_dataset(scenario), [SimpleAverage()])
    means[scenario] = scores.set_index("method").at["simple_average", "test_brier"]
print(weighted_score(means, weights))
```

The example uses one fixed bank per scenario. Fresh data seeds give a basis for reporting uncertainty in scenario means. The checkout example `python examples/compare_groups.py` reports the three ordinary reference methods on the full panel and three narrower profiles. Save the taxonomy version, membership, weights and per-scenario scores with a published aggregate.

For generated data, the public suite runner can combine repeated draws before summarizing the selected panel and its mechanism groups.

```python
import pandas as pd
from fusionbench import run_suite, summarize_suite

rows = pd.concat(
    [run_suite(["correlated", "deadweight"], n_per_split=128, seed=seed) for seed in (17, 18)],
    ignore_index=True,
)
print(summarize_suite(rows))
```

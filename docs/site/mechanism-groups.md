# Choose and weight scenarios by mechanism

An algorithm that handles a planted XOR rule has demonstrated a useful capability. A practitioner combining classifiers from several acquisition sites may care more about correlated observations and changing site composition. FusionBench can report either view of the same evaluation. Its standard score still gives each of the 30 scenarios equal weight.

Version `mechanisms-v1` assigns each `standard-v6` scenario one primary mechanism. The assignment follows the generator's construction. Keep the version, chosen scenario weights and per-scenario results alongside an aggregate so another researcher can reproduce its interpretation. [Scenario generators and parameter defaults](scenarios.md)

## Which mechanisms are covered?

| Group | Question | Scenario IDs |
| --- | --- | --- |
| `evidence_calibration` | Can the method combine conditionally independent evidence and handle changes in the scale of reported probabilities? | `committee`, `miscalibration`, `multiclass_committee` |
| `pool_selection` | Can it select or average providers when a pool contains weak, irrelevant or conditionally redundant inputs? How much does searching this pool cost on fresh data from the same population? | `deadweight`, `needle`, `redundant_pool`, `stratified_pool` |
| `error_dependence` | What happens when providers share errors, reverse the sign of an observation, or confuse the same classes? | `saturated`, `correlated`, `calibrated_suppressor`, `polarity_fault`, `similar_classes_38` |
| `structured_rules` | Can a method represent and learn an explicit rule on conditions or scores? The targets include AND, OR and XOR of latent conditions, selection by distance to a threshold, and nested minima and maxima. | `hardveto`, `hardrescue`, `weakest_link`, `two_pathways`, `parity`, `compound`, `native_switch`, `native_close_switch`, `compositional_switch` |
| `context_routing` | Can the method use an observable cue or a provider's fallback behavior to decide whose prediction to trust for this example? | `gated_experts`, `rare_specialist`, `selective_fallback` |
| `transfer_shift` | Does a rule still work when shortcut reliability, region frequency, or observation quality changes between fitting and deployment? | `environment_shortcut`, `cloned_shortcut`, `rare_shortcut_noise` |
| `grouped_sampling` | How well does selection transfer to new scenes when examples within one scene share signal and error? | `grouped_rare_event`, `shared_error_bloc`, `rare_scene_specialist` |

Some generators deliberately combine mechanisms. For example, `shared_error_bloc` has both provider dependence and within-scene dependence. It contributes to the grouped-sampling score because the generator draws independent scenes with repeated observations. The ordinary `correlated` generator isolates provider dependence across independently sampled examples. Likewise, `rare_specialist` tests selection using a regional cue, while `rare_shortcut_noise` explicitly manipulates which aspects of the rare-region source transfer. The primary partition counts each scenario exactly once.

## What is the evaluation for?

A second annotation records how a scenario can be used. These tags **overlap**.

- `capability_probe` selects a controlled law with a specified statistical, logical or routing operation to recover. For example, the negative weak-provider coefficient in `calibrated_suppressor` tests signed evidence handling. The FAR/CLOSE scenarios directly plant threshold-based selectors. Those scores can be reported separately when comparing fusion methods with different primitive operations.
- `operational_stress` selects a recognizable difficulty such as shared or unequal errors, confidence distortion, an unreliable shortcut, weak signals, or dependence between rows. Each law is synthetic. The tag describes the question the simulation asks. Establishing how often the mechanism occurs in a particular deployment requires evidence from that deployment.
- `reference` selects four anchor cases for interpreting an intervention. The two committees supply independent-evidence references, `redundant_pool` provides a null for searching redundant inputs, and `stratified_pool` fixes the population across training and test. A reference can also be a capability probe or a stress case.

The tags describe the default generator. A researcher changing the parameters or combining perturbations should state which property remains under test. Constructor variations are documented separately. [Mechanism stress profiles](stress-profiles.md)

## Specify a scoring profile

`scenario_weights` returns an ordered dictionary from registered scenario IDs to normalized weights. Its keys can select or generate benchmark data. Its values weight a **per-scenario result**, after averaging data seeds within that scenario. In the original standard macro, the nine structured-rule scenarios receive 9/30 of the total mass. Under equal group weighting each of the seven groups receives 1/7. Both aggregates can be shown alongside the same individual measurements.

~~~python
from fusionbench.taxonomy import (
    APPLICATION_PROFILES, MECHANISM_GROUPS, PURPOSE_TAGS, scenario_weights,
)

standard = scenario_weights()
equal_groups = scenario_weights(group_weights=dict.fromkeys(MECHANISM_GROUPS, 1))
multi_site = scenario_weights(group_weights=APPLICATION_PROFILES["multi_site"])
print(PURPOSE_TAGS["capability_probe"])
~~~

The two named examples state research preferences. `multi_site` gives 50% to transfer shift, 30% to grouped sampling and 20% to context routing, which selects nine scenarios. `rule_design` gives 75% to structured rules and 25% to routing, which selects twelve. A deployment-specific use should justify its weights from the intended cost or available observations.

- `groups` and `purposes` each accept one ID or several. Names in one argument form a union. The arguments intersect, so a case carrying two chosen tags still occurs only once.
- `scenarios` accepts selected IDs, or a mapping from individual IDs to nonnegative relative weights. This can include a locally registered scenario. Membership in the published taxonomy stays fixed after registration. To mix a new local case with a named profile, expand the profile to individual scenario weights first.
- `group_weights` limits selection to the named groups and allocates their relative total masses. Within each represented group, selected cases share its mass in proportion to the individual weights (equally by default). Zero weights omit a choice. A group with no selected cases drops out before renormalization. An entirely empty selection raises an error.

For example, this custom profile gives 40% of its mass to a three-provider multiclass panel and retains the relative weights of the nine multi-site cases in the remaining 60%. The same expression accepts a registered lab-specific generator in place of `multiclass_provider_panel`. [Register a local scenario](extend.md)

~~~python
custom = scenario_weights(scenarios={
    **{name: 0.60 * weight for name, weight in multi_site.items()},
    "multiclass_provider_panel": 0.40,
})
~~~

## Apply weights to existing results

The mechanism stress runner writes `by_scenario.csv` with a mean loss per scenario, method and panel. The example uses only the reference setting of each generator and scores methods on the same selected cases. Missing or nonfinite scores for any positively weighted case raise an error. For a new standard evaluation, pass `scenarios=list(multi_site)` to `run_profiles` to generate just those cases. Custom registered cases can use the ordinary dataset generator and evaluation API. [Evaluate fusion methods](evaluate.md)

~~~python
import pandas as pd
from fusionbench.taxonomy import weighted_score

rows = pd.read_csv("results/by_scenario.csv")
reference = rows[rows.panel == "reference"]
for method, frame in reference.groupby("method"):
    means = frame.set_index("scenario", verify_integrity=True)["mean_brier"].to_dict()
    print(method, weighted_score(means, multi_site))
~~~

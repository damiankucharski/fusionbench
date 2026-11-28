# Scenario catalog

The `standard-v6` panel fixes 30 scenario IDs. The taxonomy assigns each ID to one primary mechanism based on the generator. A single scenario can contain more than one difficulty. For instance, `rare_shortcut_noise` contains both regional expertise and a shortcut whose usefulness can change between splits. The intervention column below gives an example of something you can vary. To inspect predefined settings for any standard ID, use `fusionbench.profiles.list_profiles(id)`. [Weighting the groups](scoring.md)

## Evidence and calibration

Group ID: `evidence_calibration`.

| ID | What the method sees | Example intervention |
| --- | --- | --- |
| `committee` | Binary providers offering independent evidence conditional on the label. | Reduce each provider's accuracy or the number of providers. |
| `miscalibration` | Providers report probabilities with different log-odds scales. | Change the scale range while preserving the binary decisions. |
| `multiclass_committee` | Multiclass providers with independent logit noise and evidence for the correct class. | Change the number of classes or the strength of that evidence. |

## Pool selection

Group ID: `pool_selection`.

| ID | What the method sees | Example intervention |
| --- | --- | --- |
| `deadweight` | Informative providers mixed with irrelevant reports near 0.5. | Increase the number of irrelevant providers. |
| `needle` | A small informative subset inside a large pool. | Reduce the pool while keeping the number of informative providers fixed. |
| `redundant_pool` | Many candidate reports can look useful by chance alongside an informative source. | Increase the search space or enable a genuinely informative candidate. |
| `stratified_pool` | Fit and test both balance class labels and evidence strength. The true event probability combines the provider mean with one provider. | Vary specialization or available fit examples to study selection under sampling uncertainty. |

## Dependence between errors

Group ID: `error_dependence`.

| ID | What the method sees | Example intervention |
| --- | --- | --- |
| `correlated` | One block shares errors while other providers contribute independent evidence. | Increase `corr` while holding marginal provider accuracy fixed. |
| `saturated` | Strong multiclass providers with little independent evidence. | Vary their shared noise. |
| `calibrated_suppressor` | Two calibrated sources of unequal strength. A weak source's joint coefficient can become negative as their positive correlation rises. | Compare `corr=0.0` with `corr=0.85`. |
| `polarity_fault` | One sensor sometimes reverses its evidence; six others share noise. | Vary the probability of a sign reversal. |
| `similar_classes_38` | Three providers share a confusion between classes 3 and 8 in a ten-class problem. | Vary how much of the confusion is shared among providers. |

## Structured rules

Group ID: `structured_rules`.

| ID | What the method sees | Example intervention |
| --- | --- | --- |
| `hardveto` | Positive classification requires both presence and safety. | Vary the frequency of cases on which the safety condition is decisive. |
| `hardrescue` | Any one of multiple conditions can make the label positive. | Change the number of conditions or the frequency of decisive rescues. |
| `weakest_link` | All latent conditions must hold. | Increase rule depth or change decisive-case prevalence. |
| `two_pathways` | The label follows (A AND B) OR (C AND D). | Make the second pathway rarer. |
| `parity` | The label follows XOR of two latent bits observed through noisy detectors. | Weaken evidence for the second bit. |
| `compound` | A core or rescue condition must hold, subject to a safety veto. | Vary the support for rescues and vetoes. |
| `native_switch` | Of two designated scores, the one farther from a cutoff determines the event probability. | Mix the switching law with a control. |
| `native_close_switch` | The score closer to the cutoff determines the event probability. | Change the roles of providers. |
| `compositional_switch` | Four designated scores enter a max of two pairwise minima. | Change the frequency of informative corner cases. |

## Context routing

Group ID: `context_routing`.

| ID | What the method sees | Example intervention |
| --- | --- | --- |
| `gated_experts` | One provider indicates which of two other providers is informative for a row. | Weaken the selector while keeping the experts. |
| `rare_specialist` | A specialist is useful in an uncommon region, signaled by a noisy cue. | Change the region prevalence or the quality of the cue. |
| `selective_fallback` | A specialist can return a fallback report, and availability can depend on difficulty. | Compare difficulty-dependent and random missingness. |

## Transfer shift

Group ID: `transfer_shift`.

| ID | What the method sees | Example intervention |
| --- | --- | --- |
| `environment_shortcut` | The relationship between a shortcut and the label depends on the environment mixture. | Use the same mixture in fit and test as a control. |
| `cloned_shortcut` | Providers can repeat a shortcut that reverses in the test population. | Change the fraction of clones or the test reversal probability. |
| `rare_shortcut_noise` | A generalist, regional specialist, cue and shortcut face several possible changes at deployment. | Change test regional prevalence, shortcut activity, provider noise or label noise independently. |

## Grouped sampling

Group ID: `grouped_sampling`.

| ID | What the method sees | Example intervention |
| --- | --- | --- |
| `grouped_rare_event` | Rows share scenes, providers can differ in quality and residual correlation, and a specialist can activate on some scenes. | Change one of these effects at a time, then combine them. |
| `shared_error_bloc` | Six members of an eight-provider pool share residual error. | Change the block size or set the shared error to zero. |
| `rare_scene_specialist` | A specialist can be accurate on a minority of scenes. | Alter its strength or the frequency of those scenes. |

Scene IDs are recorded per split for the three grouped generators. Holding out complete scenes tests transfer to a new scene. Holding out rows from a scene already used in training addresses a different question.

## Additional generators

The registry also provides `multiclass_provider_panel` for varying provider quality, reporting precision and shared difficulty, and `shared_confusion_panel` for configurable cross-class confusions. Select either explicitly with `generate_dataset`. The 30 standard IDs and their packaged reference banks stay fixed when additional generators are registered.

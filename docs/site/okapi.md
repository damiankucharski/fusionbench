# Diagnosing a fusion tree with FusionBench

OKAPI uses genetic programming to build a tree that combines provider predictions. If its test loss is high, three questions lead to different experiments. Can the allowed operations express the generating rule? Does a finite search visit a tree that computes it? Does validation choose a good tree from those visited? FusionBench can answer each question separately when the conditional probability of the label is known.

## What can one operation express?

In a binary test with six anonymous providers, four contain the signals `A`, `B`, `C` and `D`. The true probability is `max(min(A, B), min(C, D))`. A pair supplies strong positive evidence only when both of its scores are high. A tree with two minimum operations and one maximum reproduces this probability on every input.

There is also a **proved limit** for a shallower tree. The generator draws half its examples uniformly from 64 corners where each of the six scores is either 1/64 or 63/64. On these corners one threshold operation reduces to a minimum, a maximum or its first input. A normalized weighted mean reduces to a convex average. Enumerating the subsets and optimizing the convex weights gives a minimum squared error of 1/8 on the rescaled zero–one corners. The corner component alone therefore contributes at least **½ × (62/64)² × ⅛ = 961/16384 ≈ 0.05865** to the population squared error relative to the true probability. This bound covers a lone provider or one application of minimum, maximum, mean, normalized weighted mean, nearest-to-threshold or farthest-from-threshold. The three-operation witness achieves zero. The status of two-operation trees is open in this comparison. [Proof and native witness](https://github.com/damiankucharski/OKAPI/blob/c91dd94ffc432b08a882fd64f281966bc47a5348/experiments/compositional_control/README.md) · [Independent certificate](https://github.com/damiankucharski/fusionbench/blob/05fa4ec364e545671402ec1ff14392b818fc4d6a/docs/ensemble-study/search-regret/organic-composition/verify.py)

## Did search find it, and did validation return it?

A separate experiment hid the same two-pair rule among eight providers. The evaluator could compare a candidate's function with the target on the entire finite input grid. Each run used 1,200 training-score evaluations and a separate set of 256 validation labels. The two deep initializations used the same provider occurrences, operators and node counts, arranged as two branches or as a chain.

| Starting trees | Exact rule present at initialization | Exact rule visited by 1,200 evaluations | Exact rule returned by validation |
| --- | ---: | ---: | ---: |
| Two branches | 1/24 | 21/24 | 14/24 |
| Matched chains | 0/24 | 11/24 | 8/24 |
| Ordinary shallow trees | 0/24 | 2/24 | 2/24 |

With branched starts, three runs never visited the rule and seven visited it but returned something else. Increasing the search budget addresses a different failure from improving how the final tree is selected. The 24 runs used two search seeds on each of 12 sampled data sets. The counts describe this target, initialization and budget. [Experiment and protocol](https://github.com/damiankucharski/fusionbench/blob/05fa4ec364e545671402ec1ff14392b818fc4d6a/docs/ensemble-study/seed-geometry/README.md) · [Per-run results](https://github.com/damiankucharski/fusionbench/blob/05fa4ec364e545671402ec1ff14392b818fc4d6a/docs/ensemble-study/seed-geometry/focal-runs.csv.gz)

## Does the validation choice transfer?

To limit the GP programs compared on a small validation split, construct
`RegularizedNativePatternOKAPI(seed=0, split_seed=0, gp_validation_top_k=8)`.
`NativePatternOKAPI` accepts the same keyword. At each checkpoint, the
method admits the eight structurally distinct programs with the best
discovery scores and uses their validation scores to choose the GP
representative. Equal discovery scores retain first-visit order. The
default discovery score favors lower unrounded Brier. A custom fitness
uses its higher-is-better discovery score. Both native head families
continue to compete with that representative. The full archive remains
available for feature screening. Omitting the option gives unrestricted
GP validation. With the option enabled, inspect `counts_["gp_final_eligible"]` for the final
shortlist and `counts_["score_calls"]["gp_validation"]` for actual GP
validation calls, including the initial checkpoint.

Keep the candidates fixed and change the test inputs. Across 19 scenarios and eight data sets per scenario, four native trees competed on 192 clean validation labels in each run. The validation winner tied for the lowest test Brier among the four in **65.1%** of clean test sets and **27.6%** under the strongest tested log-probability noise. Its average excess over the best of those four rose from **0.00338** to **0.06271** in full-class Brier. These comparisons use the same trees before refitting. Test labels identify the best available candidate retrospectively. [Transfer results and definitions](https://github.com/damiankucharski/fusionbench/blob/05fa4ec364e545671402ec1ff14392b818fc4d6a/docs/ensemble-study/transfer-grid/README.md)

You can apply that last diagnostic to other registered fusion methods. Choose on source validation labels, freeze the candidates and compare their rankings under paired test disturbances. [Run the transfer benchmark](transfer.md)

## Class-specific fusion on multiclass probabilities

The method IDs `okapi_class_diagonal` and `okapi_class_full` fit one native softmax node directly to provider probabilities. The diagonal mode scores each class using that class's probability from every input. The full mode also uses probabilities assigned to other classes. Both fit by derivative-free search. Choose the logistic controls `stacking_diagonal` or `stacking_shared_bias` to compare the effect of own-class slopes and class-specific intercepts.

To let validation also consider evolved subtree inputs, use `NativePatternOKAPI(class_head="full", class_head_max_evaluations=4096)` from `fusionbench.methods.native_pattern`. Each head fit has its own evaluation budget. The deployed result is one tree. [Paired study and baselines](https://github.com/damiankucharski/fusionbench/blob/research/ensemble-mechanisms-stability/docs/ensemble-study/class-capacity/README.md)

## Fit a native convex pool as a low-variance comparison

With the [simplex fitter in OKAPI](https://github.com/damiankucharski/OKAPI/blob/d1afe2b2164e67df5c9e157b84d8729a3850f65e/docs/api/simplex.md) on the Python path, the FusionBench adapter returns one native probability tree. The fit uses no derivatives. The weights are shared by classes, constrained to be nonnegative and sum to one. The default penalty favors similar weights and its strength decreases as more training observations become available.

```python
from fusionbench import generate_dataset
from fusionbench.methods.simplex_tree import SimplexTree

bank = generate_dataset("stratified_pool", seed=9, n_per_split=128)
model = SimplexTree(regularized=True).fit(
    bank.fit_preds, bank.fit_y, task="binary", n_classes=2
)
probabilities = model.predict(bank.test_preds)
weights = model.weights_
```

Use `regularized=False` to remove the penalty while retaining the same search schedule. A weighted average can lose substantial accuracy when the target calls for a minimum, maximum or nonlinear interaction. Compare it to the full native-pattern pipeline in those cases. [Five benchmark-wide experiments](https://github.com/damiankucharski/fusionbench/tree/research/ensemble-mechanisms-stability/docs/ensemble-study/stability-five)

## Choose a fusion configuration from source predictions

`ConfiguredOKAPI` takes a prediction bank from a new task and selects one of five configurations. In order of preference in close comparisons, these are the arithmetic mean, Brier-ranked Top-N, a temperature-calibrated arithmetic mean, a derivative-free regularized logit head, and a calibrated mean-seeded GP tree. The arithmetic mean and Top-N can both be the final prediction rule.

Install the optional dependency with `pip install '.[okapi]'` from this FusionBench checkout. It pins the OKAPI revision that supplies pool diagnostics, the prediction-pull objective and the learned-policy reader. [Pinned OKAPI source](https://github.com/damiankucharski/OKAPI/tree/25947c4a695c64c85922d18ac605b8eb2a61d4d7)

```python
from fusionbench import generate_dataset
from fusionbench.methods.auto_configuration import ConfiguredOKAPI

bank = generate_dataset("correlated", n_per_split=256, seed=11)
model = ConfiguredOKAPI(policy="source_cv")
model.fit(bank.fit_preds, bank.fit_y, task="binary", n_classes=2)
print(model.action_, model.selection_["oof_brier"])
predictions = model.predict(bank.test_preds)
```

With `source_cv`, each row gets one prediction from a model fit on the other two folds. If rows come from repeated measurements of the same scene or subject, pass `groups=scene_ids` to `fit`. The folds keep groups intact and give each group's average Brier loss equal weight. The selector takes the first configuration whose excess loss over the empirical winner is within one standard error of their paired differences. This rule expects at least two providers and three independent units. It uses the arithmetic mean when either is missing. It refits the chosen configuration on every available source row for prediction. Independently changing target behavior can still reverse the choice.

To apply an existing shallow policy, pass its JSON-decoded mapping as `ConfiguredOKAPI(policy=mapping)`. The model computes the pool measurements and fits only the resulting action. Inspect `selection_["path"]` for the diagnostic values and thresholds that led to it. The action names, fitted rule and training-family coverage belong with any published evaluation. [Study protocol](https://github.com/damiankucharski/fusionbench/blob/research/ensemble-mechanisms-stability/docs/ensemble-study/autoconfig-three/protocol.md)

## Include two arithmetic pools in native pattern selection

`PoolCandidateNativePatternOKAPI` lets the regularized native pattern search compare two more trees on its existing held-out rows. The benchmark method ID is `okapi_native_pattern_l2_pools`. One tree averages every provider. The other ranks the providers by discovery-set Brier score, evaluates the average of each prefix in that ranking and retains the best prefix. Each gets one held-out score alongside the flat logit head, the head with evolved inputs and the GP finalist. An exact tie keeps the original winner, followed in priority by the mean and Top-N.

```python
from fusionbench import generate_dataset
from fusionbench.methods.native_pool_candidates import PoolCandidateNativePatternOKAPI

bank = generate_dataset("correlated", n_per_split=256, seed=11)
model = PoolCandidateNativePatternOKAPI(seed=0, split_seed=0, gp_validation_top_k=8)
model.fit(bank.fit_preds, bank.fit_y, task="binary", n_classes=2)
probabilities = model.predict(bank.test_preds)
print(model.action_, model.chosen_, model.best_tree)
```

The selected method deploys one native tree. If Top-N wins, the method ranks providers and chooses the prefix again on **all** source rows. To compare against the original winner from exactly the same GP evaluations, call `model.predict_incumbent(bank.test_preds)`. Inspect `model.counts_["pool_candidates"]` to see which providers Top-N selected on discovery and after a winning refit. For M providers the extra work amounts to 2M discovery Brier evaluations, two held-out scores and, when Top-N wins, another 2M evaluations at refit. The historical 70/30 split samples rows. Multiple observations of the same scene can therefore occur on both sides.

## Choose between regularized OKAPI, mean and Top-N using source folds

When the question is whether a simpler pool performs well enough on a particular source bank, compare predictions for the same held-out source rows. `fit_source_choice` shares three outer folds across three actions. The actions are the arithmetic mean, Brier-ranked Top-N, and the regularized native pattern method with eight discovery-ranked GP finalists. The native search uses the supplied seed and fixes its internal discovery split seed at zero.

```python
from fusionbench import generate_dataset
from fusionbench.methods.native_choice import fit_source_choice

bank = generate_dataset("correlated", n_per_split=256, seed=11)
model, action, evidence = fit_source_choice(
    bank.fit_preds, bank.fit_y, rule="paired", seed=0
)
probabilities = model.predict(bank.test_preds)
print(action, evidence["oof_brier"])
```

For repeated observations, pass `groups=scene_ids`. Outer folds keep each scene intact. Each scene's mean full-class Brier loss receives equal weight in the comparison. Without groups each row counts as one unit. On each outer training sample the native candidate keeps its usual internal 70/30 row split. The helper applies the requested rule once and fits that action on the entire source bank.

| Rule | When it selects an action |
| --- | --- |
| `cv` | Select the lowest average out-of-fold Brier loss. |
| `paired` | Switch from native to the better arithmetic pool when its average gain reaches one paired standard error. |
| `sign` | Require the paired threshold and a strictly positive gain in every outer fold. |
| `stress` | Require the paired threshold and a nonnegative gain both below or at and above median provider disagreement. |
| `one_se` | Select the first of mean, Top-N and native within one paired standard error of the best action. |

Disagreement is the average squared class-probability distance between a provider and the arithmetic mean, averaged within each scene. Equal disagreement values stay together. If all units have the same value, the stress rule checks the pooled gain. The reported standard errors are selection heuristics computed from paired unit losses on shared folds. The returned evidence contains these losses, the fold and stratum of each unit and the selected simple pool's gain in each. With fewer than three units or fewer than two providers, the helper returns the arithmetic mean and records the fallback reason.

To fit a ranked prefix as a standalone OKAPI tree, use `okapi.ranked_prefix.fit_brier_ranked_prefix(providers, labels)`. It returns the native tree and the provider IDs in its selected prefix. Only these providers are required at inference. [Native API and example](https://github.com/damiankucharski/OKAPI/blob/25947c4a695c64c85922d18ac605b8eb2a61d4d7/docs/api/core.md)

## Screen providers across observed source environments

`EnvironmentScreenedOKAPI` looks for a provider whose Brier loss changes across source environments, such as sites or acquisition conditions. The adjusted mode compares sites at a common mix of observed contexts. For example, if difficult cases occur more often at one site, pass their recorded category as the context. The estimated discrepancy can also reflect genuine changes in expertise, so removing a provider can hurt in a target environment where it remains useful.

```python
from fusionbench.methods.environment_screen import EnvironmentScreenedOKAPI

model = EnvironmentScreenedOKAPI(mode="adjusted")
model.fit(
    source_predictions, source_labels, task="binary", n_classes=2,
    environments=source_sites, contexts=source_case_types,
)
probabilities = model.predict(test_predictions)
print(model.drop_, model.screen_info_["reason"])
```

Supply one site ID and, in adjusted mode, one context ID per source row. When rows repeat within scenes, `screen_indices` can supply a **prespecified, independent** representative per scene as integer row indices or as a Boolean mask. The screen uses only these diagnostic rows, then the native method fits on the full source bank and preserves its usual discovery and validation split. Environment IDs and contexts are needed only at fit time.

With the defaults, the screen requires eight observations in every site/context cell. For each provider it takes the largest absolute difference in adjusted site losses after subtracting 1.96 times its estimated standard error. It drops a provider only when its score is uniquely highest and exceeds 0.10 full-class Brier. Missing metadata, inadequate cell coverage, and ties leave the pool intact. Raw mode compares observed site losses directly and requires two observations per site. The standard errors assume independent diagnostic rows and the thresholds are selection heuristics.

`predict` accepts the original provider ordering. For compact inference, compute only the providers listed in `model.kept_indices_` in that order and pass their predictions to `model.predict_retained`. If only one provider remains, its normalized probabilities give the fused result. The site screen is opt-in.

## Limit reliance on one provider at prediction time

A fitted fusion model can use one provider as a gate for others. To measure that dependence, substitute a fixed set of **whole source reports** for this provider on every query row, run the fitted model once per substitution, and average its outputs. The other providers keep their observed values on that row. The procedure uses one fitted model.

```python
from fusionbench.methods.provider_reliance import (
    limit_reliance, marginal_prediction, source_donors,
)

donors = source_donors(source_predictions, provider=2, count=8)
original = model.predict(query_predictions)
marginal = marginal_prediction(model.predict, query_predictions, provider=2, donors=donors)
probabilities = limit_reliance(original, marginal, strength=0.5)
```

At strength zero the result equals the ordinary fusion forecast. At strength one it equals the average after substitution. Eight donor reports require eight fusion evaluations for each batch, plus the original evaluation when blending. Donors are selected at the middle rank of each of eight equal probability intervals after sorting source reports lexicographically by their class columns. Duplicate source reports retain their frequency. With one donor per source row the average uses the complete empirical marginal distribution.

Substitution breaks the flagged provider's correlations with the rest of the pool. This can produce report combinations absent from training and can remove information a genuine specialist or gate supplies. The chosen source sample, class ordering and donor count matter when using a finite set of donor reports. Evaluation on separate target data measures the consequences.

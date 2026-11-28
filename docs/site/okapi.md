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

Install the optional dependency with `pip install '.[okapi]'` from this FusionBench checkout. It pins the OKAPI revision that supplies pool diagnostics, the prediction-pull objective and the learned-policy reader. [Pinned OKAPI source](https://github.com/damiankucharski/OKAPI/tree/5707500fd5cf8b199ce3d1f201e63af8b1dec0f7)

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

# What can a fusion rule learn from the identity of the wrong class?

With three providers and ten classes, an own-class logistic stacker has 30 slopes and ten intercepts. For example, it scores class 3 using each provider's probability of 3. Unrestricted multinomial stacking uses 300 slopes and ten intercepts and can additionally use each provider's probability of 8 to score class 3. We compared both on identical fit and test predictions, with the same pooled scaling and regularization convention. Two simpler controls share provider slopes across classes, with or without class-specific intercepts. [Frozen design](protocol.md)

## Results on real prediction banks

Mean full-class test Brier over eight partitions, lower is better. The digits pools reuse underlying images; A uses logistic regression, nearest neighbors and a random forest, B uses Gaussian naive Bayes, distance-weighted neighbors and extra trees, and the bag uses bootstrap logistic regressions. B also has fresh partitions. Differences between A and B therefore cannot be attributed solely to the provider types. The six pools cover four underlying datasets.

| Pool | Shared slopes | Shared + intercepts | Own class | Full | Previous native OKAPI | New native full, standalone |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| digits_hetero_a | 0.03416 | 0.03319 | 0.03402 | 0.03190 | 0.03868 | 0.03597 |
| digits_lr_bag | 0.07174 | 0.07358 | 0.07047 | 0.06790 | 0.07232 | 0.07206 |
| digits_hetero_b | 0.03284 | 0.03043 | 0.03287 | 0.03616 | 0.04267 | 0.04031 |
| wine | 0.02462 | 0.02093 | 0.02054 | 0.02163 | 0.03004 | 0.03028 |
| iris | 0.08593 | 0.09470 | 0.09822 | 0.09846 | 0.08109 | 0.09824 |
| seeds | 0.12689 | 0.12893 | 0.12754 | 0.12726 | 0.13286 | 0.12624 |

Full stacking beat own-class stacking on seven of eight splits in digits A. Own-class stacking beat full stacking on seven of eight in digits B. Across the six pools the mean favored each in three. Stable class identities can carry useful information, but these real-data results alone do not identify which pairs carried it. All 11 methods, including uniform mean, validation-ranked Top-N, production probability and logit stacking, are archived. [All controls](results/panel-means.csv) · [Paired comparisons](results/paired-capacity.csv)

## Controlled intervention on classes 3 and 8

The generator assigns the largest *incorrect* probability to a destination sampled from a shared or provider-specific matrix. The shared matrix pairs 3 with 8 in both directions. Here the private matrices map those two labels to a distinct destination for each provider. Each paired intervention preserves every provider's true-class probability, sorted probabilities and individual Brier up to rounding. All eight untouched classes retain the exact original predictions. The matrix controls a destination preference, so its rows are not the classifier's observed confusion frequencies. [Scenario API and example](../../site/scenarios.md) · [Checks for each provider and seed](audit/invariants.csv)

On the affected classes alone, the change in Brier from own-class to full stacking averaged −0.00006 in the original panel, −0.00103 with the shared pair, −0.00152 in the equal mixture and −0.00439 with private destinations. In the last condition the gain occurred in all six draws. On the eight untouched classes the refitted full model lost 0.00070, giving a net gain of only 0.00032 overall. In the shared-pair condition the overall difference favored own-class stacking by 0.00077. This distinguishes the gain on the planted mechanism from the cost of fitting extra coefficients elsewhere. [Bank-level losses](results/scores.csv)

## How much of that rule can OKAPI learn?

We added two native probability-affine nodes with softmax output. Importing the production stacker's coefficients and absorbing its scaler into a full node reproduced held-out predictions to within 8.4 × 10⁻⁸ across all 72 banks. For the actual native fits a derivative-free coordinate search optimized penalized Brier skill, with up to 4096 objective calls. Full and diagonal modes can accept provider leaves or evolved subtrees. On digits A, standalone full OKAPI improved Brier from 0.03868 to 0.03597 while matched full LR reached 0.03190. Exact representability and the score reached by these two training procedures are separate observations. Different objectives, penalties and solvers prevent attributing their gap solely to the optimizer. [Coefficient replay](audit/full-lr-replay.csv) · [Native fitter](https://github.com/damiankucharski/OKAPI/blob/ce6ae177a4058612b7ffa33194a13e3fda86f5cb/okapi/multiclass_affine.py)

A secondary experiment allowed validation to choose the ordinary GP tree, a flat native head or a head over inputs augmented with evolved subtrees. On digits A, integrated full OKAPI reached 0.03790. On the shared 3/8 panel it reached 0.02408 versus 0.01839 for the earlier native procedure; five of six choices used augmented heads. Extra capacity therefore carries a measurable selection cost in this setting. These two cohorts were specified after inspecting standalone outcomes and integrated fitting uses a larger total budget. A next test is to evaluate a complexity-aware selection rule on fresh banks, keeping the fitted candidate set fixed. [Secondary design](integrated-protocol.md) · [Deployed scores, selections, fit times and call counts](integrated/summary.csv)

[Reproduction instructions, timings and independent verification](results/README.md) · [Standalone driver](run.py) · [Integrated driver](integrated.py) · [Summary computation](analyze.py) · [Replay check](certify.py) · [Intervention check](check_scenario.py)

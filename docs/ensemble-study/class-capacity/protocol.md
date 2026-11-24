# Shared class confusions and the missing stacking baseline

## Comparisons frozen before the runs

Use all 48 previously released real prediction banks (six pools, eight partitions per pool). Four pools come from the external-coverage and runner-up studies on digits and Wine, plus Iris and Seeds. The six pools reuse four datasets; digits pools can reuse images or identical partitions. Published control test results on these banks are known. No tuning on these evaluations.

Generate six independent matched draws at seeds 3500000 through 3500005 for each of four ten-class conditions, 449 fusion-fit rows and 2,048 test rows. Use the high-separation finite-vote panel B recipe. Compare original predictions, a shared wrong-class destination preference, a 50:50 shared/provider-specific preference and fully provider-specific preferences. The shared preference targets class 8 for true class 3 and class 3 for true class 8. Provider j's deviation targets class j for both affected true classes, for j = 0, 1, 2. Leave all other true classes unchanged. Use full intervention strength in the three altered conditions. An intervention only reorders wrong-class probabilities, keeping each provider's own row-wise true-class probability and Brier fixed. Refit the fusion rules independently in each condition on matched data.

Compare arithmetic mean, Brier-ranked best Top-N prefix, production regularized stacking on probabilities and logits, and the established regularized native tree procedure. Isolate capacity on the same provider-pooled probability scale, multinomial training loss and L2 convention (C=1), with four nested logit families:
1. Shared provider slopes.
2. Shared provider slopes with class-specific intercepts.
3. Own-class slopes for each provider and class with class-specific intercepts (the requested intermediate model).
4. Unrestricted provider × input class × output class map with intercepts.
The reduced slopes use the minimum squared norm of an equivalent centered full coefficient block on the probability simplex. Report all rungs separately from the production probability stacker with per-column scaling.

Add two gradient-free native OKAPI probability-affine trees with softmax output and class intercepts. The diagonal node learns own-class slopes; the full node may mix scores across classes. Both use Brier skill, l2_strength=0.25, and a maximum of 4,096 scalar objective calls per fit. Record actual calls and fit times. The native and logistic learners have different training objectives. Check that fitted production full-LR predictions can be replayed by loading its coefficients into a native node; this verifies expressivity separately from training. Check the optional use of the new heads on evolved subtree inputs via the normal native tree pipeline.

Use full-class Brier throughout (lower better); report per-bank scores, paired cohort means, and secondary synthetic Brier for labels 3/8 and the other labels. Primary scientific contrasts are own-class minus shared-plus-intercepts (class-dependent reliability) and full minus own-class (incremental cross-class signal). Report wins and losses and heterogeneity across pools. Treat real results descriptively; a native model becomes a measured candidate here, with a new default requiring independent confirmation. Design changes discovered during implementation or any missing controls must be disclosed.

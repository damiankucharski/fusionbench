# Can source evidence protect a useful provider interaction?

This is a bounded manual series of ten rounds. The prior series showed that screening a provider by variation in standalone Brier can remove a useful selector. We test whether a training-only measure of its ability to select other experts protects that role, and whether retaining some of its influence helps under uncertain transfer. The original regularized native-pattern model (GP and split seed zero, eight eligible GP finalists) is the reference. In every round include train-ranked best-prefix Top-N by Brier, arithmetic mean and probability logistic stacking (C=1). The metric is full-class Brier. All real-data work is deferred.

## Source information and decisions

Start with the prior adjusted environment screen (threshold .10, z=1.96, minimum eight rows per source cell). It can flag at most one provider. The same benchmark partition is used by every rule. Focal sources supply environment and contextual region from observations. On the 30 standard-v6 scenarios the stress test uses the original disagreement-based proxy partition and confidence strata. Three scene-grouped families use a single representative per training scene for screening and diagnostics. Every fusion model sees the full training bank. Source roles and target data are unavailable to the policies. All source diagnostics and method settings are fixed before scoring the bank.

The direct diagnostic D is cross-fitted source Brier improvement of the flagged provider over a constant source-prevalence prediction. The routing diagnostic G is the cross-fitted source Brier improvement from allowing one provider to choose between two other providers, relative to the best fixed other provider. Fit the median threshold and choices on the opposite half. Choose the positive-class coordinate in binary tasks or the best training coordinate in multiclass; require eight training rows per side. Create two halves by permuting distinct group IDs (seed, 817) and allocating alternating groups to each half. All given diagnostic rows receive one held-out assessment. A flagged provider is considered useful by either diagnostic when its gain exceeds .02. Also measure D and G for the known selector on the gate-control sample solely for auditing. In the shuffled-evidence control shuffle diagnostic labels by a stream keyed on (seed, 818) before computing the two utilities.

The hard rule retrains native OKAPI after excluding the flagged column (one remaining input is used directly). In the soft rules one native tree supplies the prediction F. To obtain M, replace only that provider's report by each of eight full probability vectors chosen at evenly spaced quantiles of its empirical *training* distribution, predict again and average. Deploy (1−s)F+sM. This intervenes on the model input with all other reports fixed. It breaks their natural correlations and can require repeated tree evaluations. A rule that finds no flag always returns F. No independent ensemble, test labels, test-frequency oracle, or extra fit split is used by the soft policies.

| Rule | Strength on a flagged provider |
|---|---|
| half | .5 |
| protect | 0 if D>.02 or G>.02, otherwise .5 |
| adaptive | 0 if G>.02; otherwise .25 if D>.02, else .75 |
| shuffled_evidence | adaptive with independently permuted diagnostic labels |

## Rounds

| Round | Falsifiable question and control |
|---:|---|
| 1 | Do direct and routing gains protect useful standard providers while still allowing intervention in the mixed source? Compare hard retraining, uniform half, protect and adaptive. |
| 2 | Do actual environment assignments support the decision? Compare adaptive to the same procedure with permuted source environment IDs, fixed source predictions and labels, and to shuffled diagnostic labels. Include varied shortcut activity at constant region prevalence. |
| 3 | Does gate utility disappear when only the selector is shuffled? Apply a paired intervention to gated_experts, independently permuting its training and test selector while preserving both experts, labels and selector marginals. |
| 4 | What is the dose response for the same fitted predictor? On one set of predictions compare strengths 0, .25, .5, .75 and 1 against hard removal and refitting. |
| 5 | Does numerical approximation dominate the marginalization? Compare two, eight and 32 donor quantiles at strength .5 on the same source. |
| 6 | Reduce the source contrast in shortcut activity from (0,1) to (.25,.75) at unchanged sample size and case mix. |
| 7 | Reduce the mixed training bank from 512 to 128 rows. The standard panel keeps 512 rows. |
| 8 | Is the interaction measure independently useful? Compare protective rules based only on D or only on G with the combined rule, including a homogeneous source. |
| 9 | Lock a rule before confirmation. Eligible rules have empirical standard macro Brier at most .005 higher than native. Among protect, adaptive, half choose lowest mixed mean per-source maximum endpoint regret. Exact tie order: protect, adaptive, half. Fall back to native if none qualify. Add reversed relation between source shortcut activity and specialist prevalence. |
| 10 | Confirm on fresh seeds, all 30 standard families, all seven focal designs and original/shuffled gate controls. Primary: upper paired 95% limit below zero for chosen minus native on mixed max regret. Safeguards: upper limits at most +.005 for standard and mixed/s100, +.010 for standard/gated_experts. Show all candidate results even when selection abstains. Stop after this round. |

Within each source bank, endpoint regret is expected Brier minus the full-information Bayes Brier of that test world. The primary is the larger regret across target shortcut activity 0% and 100%, averaged across sources **after** computing each maximum. Worst absolute Brier and intermediate-world risks are secondary. Focal tests integrate over all 16 report patterns, with extra s50 and specialist80 worlds. Standard and gate-control test 512 fresh rows from the simulator. Identical source data can precede different target laws.

For round r use seed base 940000+1000r, plus 400 for standard, 100 for focal, and 700 for the gate controls. Standard uses 2/4/8 repeats per family in R1–8/R9/R10, focus uses 6/12/24. Every round includes the mixed and case-mix-only profiles. The standard aggregate equally weights the 30 fixed families. Use 10,000 paired bootstrap resamples of seeds within each fixed family at 93000+r and paired resampling of source seeds in each focal design. Record per-family levels, per-source decisions and fit/forecast hashes, compute time, independent diagnostic checks, and all safeguards.

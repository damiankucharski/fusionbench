# Can training evidence identify when to standardize case mix?

In the previous series 20 pairs of mixed and case-mix-only source banks had identical environment and context counts and identical training weights. Weighting reduced worst-endpoint regret by 0.01928 on mixed but increased it by 0.01501 on case mix alone. This prospectively specified ten-round study makes the decision depend on whether provider loss changes between source environments after adjusting for context. The intervention acts on native tree fitness. Every prediction comes from a single fitted OKAPI tree in the OKAPI arms.

## Policies

Fit the same regularized native-pattern OKAPI as before (GP seed 0, split seed 0, eight eligible GP finalists, variance-scaled L2 regularization) on all source rows with its ordinary internal split. Compare native, arithmetic mean, train-Brier-ranked best-prefix Top-N, probability logistic stacking (C=1) and unconditional ESS-guarded weighting. The unconditional rule equalizes source context proportions across environments, using unrestricted weights if ESS/n ≥0.65, else weights clipped above 2 before normalization if they meet that floor, else ones. It returns ones if a source environment/context cell has fewer than four rows. All enabled weights act during discovery, validation and final refitting. Controls retain providers and the full source bank.

Use full-class Brier on source labels to screen providers. Adjust each environment to pooled context proportions. The original screen selects at most one provider whose unique largest cross-environment discrepancy after subtracting 1.96 standard errors exceeds 0.10 (eight diagnostic examples per environment/context cell). Grouped standard presets use one diagnostic row per known scene; fitting and weighting still use all rows. Standard environments are proxies from provider disagreement; context describes consensus confidence. Focal simulations supply true environment labels and the observed region. The rare case metadata are unavailable to any method at deployment.

| Method | Decision on these same source rows |
|---|---|
| single | Use the unconditional ESS-guarded weights only if the adjusted screen finds a provider. Otherwise use ordinary native. |
| half | Apply the single-screen decision with half-strength training weights, 1+0.5(w−1). |
| replicated | Require the full adjusted screen and two stratified, disjoint half-screens to identify the same provider with the same sign of discrepancy on the same environment pair. Half-screens require four diagnostic rows per cell, z=1 and effect threshold .05. Use full ESS-guarded weights only when all checks pass. |
| raw (R2) | Use the unadjusted full source discrepancy as the trigger. |
| gate_sham (R3) | Within each observed context stratum, independently shuffle environment labels for screening, preserving its contingency table. If active, apply the original source weights to their original rows. RNG keyed on (source seed, 1143). |
| random_gate (R8) | Apply the same unconditional weights only if a label-independent fair coin keyed on (source seed, 1148) is <.5. This is a chance trigger, not a trigger-frequency-matched test. |

All deterministic split and tie handling is recorded in the pinned OKAPI implementation. An abstaining rule reuses ordinary native exactly. Store source decisions, selected provider indices, row-weight hashes, effective counts, forecast hashes and Brier including the actual source role as an audit-only label. A detected provider's loss can have several causes. The screen supplies a decision statistic, not a calibrated significance test for a causal shortcut.

## Fixed reporting panels

Group standard-v6 by the public mechanism taxonomy **mechanisms-v1**, assigned using the generator laws before fitting this series. Each standard scenario belongs to exactly one of seven primary groups: evidence_calibration, pool_selection, error_dependence, structured_rules, context_routing, transfer_shift and grouped_sampling. Cross-cutting purpose tags can overlap. Report the historical uniform-over-30 macro for comparability, each mechanism mean with its membership and count, and the uniform-over-seven-groups macro. Two illustrative profiles use explicit question-specific weights. **multi_site** assigns 50% to transfer_shift, 30% to grouped_sampling and 20% to context_routing. **rule_design** assigns 75% to structured_rules and 25% to context_routing. Within each group, weight its fixed scenario means equally. These choices express examples of research priorities; neither estimates the prevalence of these situations in actual deployments. The focal source profiles are separate from the standard macro. None of the cluster rankings or illustrative profiles tunes the lock.

## Ten comparisons

| Round | Source intervention and question |
|---:|---|
| 1 | Mixed vs case-mix-only, same counts and labels. Can source losses determine whether to weight? Compare against both ordinary native and unconditional guarded. |
| 2 | Apply the raw screen that ignores case proportions; add varied activity at constant case mix. Are false activations attributable to unbalanced relevant cases? |
| 3 | Shuffle environment IDs **within** contexts for the trigger, keeping real weight assignments; add reversed association between case mix and shortcut. |
| 4 | Compare 128 versus 512 training examples. Does requesting replicated evidence destroy useful sensitivity at small n? |
| 5 | Reduce overlap to region rates .035/.965, keeping 512 training rows. Separate insufficient diagnostic support from high training-weight concentration. |
| 6 | Reduce the source activity contrast from (0,1) to (.25,.75) while preserving regional prevalence. Test the accuracy cost of imposing a minimum effect. |
| 7 | Flip 25% of observed source context metadata only. All provider reports and labels stay paired with mixed at the same seed. |
| 8 | Homogeneous null plus randomized activation. Separate selective triggering from occasional use of weights. |
| 9 | Lock a gate on independent development seeds. Eligible methods have empirical mean differences vs native ≤+.005 on standard, ≤+.010 on case_mix/specialist80 and ≤+.010 on mixed/s0. Among them take lowest mean mixed max regret. Exact tie order: half, replicated, single. Fall back to native if none qualify. Commit raw-score hash and rule before final confirmation. |
| 10 | Fresh confirmation on 30 standard families and nine focal profiles. Require upper 95% paired limit below zero on mixed/max_regret and upper limits at most +.005 standard, +.010 case_mix/specialist80 and +.010 mixed/s0. Report each separately even if the conjunction fails. Evaluate mechanisms and illustrative panels as secondary, irrespective of lock. Stop after ten. |

Round r uses base seed 1140000+1000r plus 400 for standard and 100 for focal. Use 2/4/8 fresh draws per standard family and 6/12/24 per focal profile in R1–8/R9/R10. Mixed and case_mix appear every round. Each focal bank produces four exact test risks with 16 shared input patterns (s0, s50, s100 and specialist80). Compute per-source regret after subtracting each endpoint's own Bayes risk, then take the maximum across s0 and s100 before averaging. Standard draws use 512 source and 512 test observations per family. Use 10,000 paired seed bootstrap replicates at 113000+r, stratified within fixed families for a weighted multi-family panel. Compare predictions on the same banks. Record Brier, macro-F1, source and test identities, risk intervals, gate activations, resource use and reproducible code. Synthetic scope and a single GP/split seed limit the inference.

## Secondary dose comparison added before confirmation

During the exploratory rounds, before locking R9 or inspecting R10, we specified a comparison applying half-strength ESS-guarded weights to every eligible focal source. This separates the size of the intervention from the decision to apply it. It uses the same R10 source banks, prediction code and target laws. Its result has no role in rule selection or the primary pass criteria. See the [dose-control protocol](dose_control.md).

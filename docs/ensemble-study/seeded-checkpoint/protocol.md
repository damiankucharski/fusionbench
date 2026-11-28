# Can OKAPI keep a useful arithmetic mean and leave it for a better provider?

Manual bounded series, frozen before computing the study outcomes. The periodic executor stays paused. The previous checkpoint-reuse study found that even the best programs visited by ordinary GP scored worse than averaging on its equal-competence law. Here we change what is present during search and compare it to what is merely available at the final decision.

## Two matched laws on fresh scenes

Use 32 fresh banks with seeds 285100 through 285131, each under two paired laws. Reuse the preceding generator's eight providers, eight correlated checkpoint choices, two exchangeable development blocks A/B (24 scenes of eight observations per block) and 256 audit scenes I from the same law. Under **equal**, retain the original generator unchanged. The arithmetic mean is a useful baseline with no assumed optimality. Under **signal**, replace all eight checkpoints of one anonymized provider by the exact positive-class probability q = sigmoid(1.8 z). Choose its index from an independent seed stream. The remaining providers, labels, latent probabilities and group IDs are bit-identical between laws. This is a strong positive control with a perfect probabilistic provider. All training methods receive only the selected binary provider columns and fit labels.

Select checkpoints by pooled positive-class F1, strict p > .5, lowest slot on ties. Pick on A, fit on A and separately B; repeat picking on B, fitting on B and A. Thus every frozen bank supports a matched comparison. Reuse consumes 24 distinct labeled development scenes and disjoint fitting consumes 48. One GP seed (data seed + 1000000) per bank is shared across arms, splits and laws. Sham provider index is drawn independently of data and the signal slot using PCG64(SeedSequence([data seed, 0, 2])). Retain occasional chance matches with the signal.

## Four live searches and one terminal policy

Use native OKAPI at 1d7ad2d, population target 8, 16 generations, multiplier 2, tournament size 2, ordinary variation, three-decimal selection, size as an additional objective and max 31 nodes / depth eight. All eight provider singletons start in every arm, in identical native order. Deploy the best unrounded fit Brier among final survivors, tie-breaking by fewer nodes and then final order.

| Policy | Founders and survival | Deployment |
| --- | --- | --- |
| stock | Eight singletons | Ordinary final selection |
| sham | Add a ninth founder with the same shape and arity as the true mean, repeating one provider eight times | Ordinary final selection |
| seeded | Add the native arithmetic mean of all eight providers as ninth founder | Ordinary final selection |
| protected | Use the seeded search path unchanged | Compare the seeded winner with the original mean on fit Brier, prefer the mean on a tie |
| carry | Start with the seeded founders. After each ordinary survivor decision, if the exact founder genotype is absent, substitute a copy for the last returned survivor using its cached rounded fitness. Preserve current capacity. | Ordinary final selection, free to deploy another tree |

Stock isolates the cost of the extra founder from the sham-versus-seeded comparison. Both actual proposal counts and evaluation counts can vary as populations diverge. We match the generation and variation budgets and report their realized costs. Protected adds one comparison outside evolution. Direct references are simple mean, development-Brier-ranked Top-N prefix, standardized probability logistic stacking (C=1) and the frozen regularized native-pattern OKAPI with its usual row-based inner split.

## Endpoints and decision rules

The sole primary contrast is **seeded minus sham conditional excess full-class Brier on fresh scenes, under equal competence and reused development scenes**. Smaller is better. Conditional excess is twice the mean squared difference between predicted positive probability and q on I. This integrates out audit-label noise; it still samples scenes. Under the equal law q uses the latent signal, while under the signal law one observable provider attains it exactly. Contrast two orientations within each bank, then resample the 32 bank means with 10000 paired draws (PCG64 seed 97822), using a two-sided 95% percentile interval. An interval strictly below zero supports a benefit under this protocol.

The corresponding signal comparison tests whether the easier access to averaging is compatible with leaving it for a genuine specialist. Prespecified secondary comparisons include protected versus seeded (terminal availability) and carry versus protected (in-search protection with the same final opportunity); report both laws, and disjoint fits separately. Baselines, sampled-label Brier, pooled positive F1, distinct required providers, the fraction selecting a function equal to the mean or specialist, and retention/reappearance over generations are descriptive. Functional equivalence means absolute difference at most 1e-12 on every fit and audit probability, using no audit labels. Track structural founder identity separately. Score all candidates on audit only after fit selection; use conditional best-in-final and best-ever as diagnostic bounds. The signal should have zero conditional excess when the exact provider is used. Its identical slots are concealed by the staged input contract; an algorithm with direct access to raw checkpoint histories solves another problem.

## Broader benchmark

Run all 24 standard-v3 scenarios on two fresh seeds (285201 and 285202), 512 fit and 512 test rows, search seed zero, on each scenario's usual suite test distribution. Compare stock/sham/seeded/protected/carry and the same four references on identical predictions. Rank prefixes by fit Brier, normalize multiclass probabilities in native evaluation and select on unrounded fit Brier. Use up to eight native singleton founders and preserve the complete provider palette in the extra mean, even if M exceeds eight. Sham repeats one provider, drawn independently using SeedSequence([285000, scenario index, data seed]). Set both arms' size cap to max(31,M+1) so the 100-provider needle and its matched sham are legal. Depth stays eight. Report sampled full-class Brier with equal weight across the 24 fixed scenarios, paired bootstrap resampling the two banks within each scenario (10000 draws, seed 97822); also show the 23-scenario contrast excluding the enlarged cap and per-scenario outcomes. These suite intervals describe a thin regression panel with two draws per scenario and no model tuning.

Keep the periodic switch and default OKAPI configuration unchanged. Run generator tests, independent outcome/trajectory and oracle-leakage checks, and retain raw bank-level outputs and runnable commands.

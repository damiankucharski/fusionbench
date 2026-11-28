# Scene count, row count and a compound shift

Manual series. The previous study found that native OKAPI could replace a still-available exact provider with a tree having lower empirical Brier. We freeze the design before scoring these seeds. This series uses synthetic data throughout.

## Independent units at fixed fit size

Use 32 new banks, seeds 286100 through 286131. For each, draw three disjoint blocks of eight-row scenes from the previous checkpoint simulator. A has 24 scenes and selects one of eight checkpoints per provider by pooled positive-class F1 with strict p > 0.5 and earliest-slot ties. F is a common fit pool of 1,536 scenes; I contains 512 fresh audit scenes (4,096 rows). Freeze the same selected columns across all four fit designs within a law. Within F select the first r rows of the first g scenes. The generators assign no special meaning to the order. The first row of a scene is therefore a randomly drawn observation, paired across designs. Sources of variation match the previous generator. Under the equal law all eight providers have identical population competence, with independent provider errors conditional on latent signal. Under the signal law one hidden provider emits the true event probability q. Labels, other providers and scene identities are held fixed between laws. Selection and fitting receive provider predictions and sampled labels only.

| Fit design | Independent scenes | Rows per scene | Rows |
| --- | ---: | ---: | ---: |
| Small, clustered | 24 | 8 | 192 |
| Small, independent | 192 | 1 | 192 |
| Large, clustered | 192 | 8 | 1,536 |
| Large, independent | 1,536 | 1 | 1,536 |

At equal total rows we vary the scene composition. Along the 192-independent versus 192-clustered comparison we add seven observations per scene while retaining the very same 192 scenes. Within each sampling layout, increasing the size preserves the smaller subset. All methods use the same frozen provider columns and common audit. Checkpoint selection always uses separate scenes, a deliberate change from the prior study's reuse arm.

Run the previous native seeded GP unchanged at OKAPI 1d7ad2d: eight singleton founders and a ninth all-provider mean, target population eight, 16 generations, size as additional objective, Brier fitness and ordinary variation. The search seed is the bank seed plus 1,000,000. Deployment minimizes unrounded fit Brier among final survivors. Run simple mean, Brier-ranked best-prefix Top-N, standardized probability logistic stacking and the existing L2-regularized native-pattern OKAPI. Its usual inner split operates on rows. Report fit/audit full-class Brier and decision/availability paths, and audit excess Brier = 2 × mean((p−q)²). This oracle diagnostic averages over possible outcomes while retaining audit feature sampling. Equivalent means maximum absolute prediction difference of 1e−12 on both fit and audit probabilities.

**Sole primary contrast:** on the signal law, final seeded GP audit excess for 24×8 minus 192×1. Positive values mean that clustering hurts. Form within-bank contrasts and use a two-sided percentile interval with 10,000 bank bootstrap draws (NumPy default_rng(97823)). Report estimate and interval even if it crosses zero. Secondary comparisons are the equal law, clustering at 1,536 rows, larger sample sizes within each layout, additional observations at fixed 192 scenes and changes from generation zero. Count exact-provider availability, exact predictions chosen and cases in which the winner bypassed an available exact prediction. All repeated designs and generations share the same bank. Do not treat individual rows as independent experimental replicates.

## Crossed failure modes and suite regression

Add a ready-to-run binary scenario, cloned_shortcut, to standard-v4. Preserve v1–v3 and all historical seed slots. It has eight providers, two stable votes with 0.75 accuracy and six shortcut votes with 0.80 source accuracy. Independently toggle copying of one shortcut vote across its six columns and reversal of the shortcut direction at deployment. When copying is off the six shortcut votes have independent errors conditional on label and reversal state. Fit is aligned in every case. The clone gate and reversal gate are independent row-level events. The default applies both at probability one. Use the same random draws across the four conditions and identical fit banks across reversal settings. Fit methods see provider scores only. No target labels enter deployment decisions.

* Across 24 paired new banks (286300–286323), use 1,024 fit rows and 4,096 test rows per cell. Score all five methods at clone and reversal probabilities independently set to zero or one. Report the four test Briers, the marginal effects, and (both − clones only) − (reversal only − neither), with bank-paired percentile intervals. Positive interaction means a larger reversal penalty with copying on this loss scale. Treat this as a secondary diagnostic without directional prediction.
* Run all 25 standard-v4 scenarios and all five methods on shared fit/test draws of 512 rows each, seeds 286201 and 286202. Use the prior standard-panel search procedure with search seed zero and max_nodes=max(31, provider_count) plus space for the root whenever required, i.e. max(31, provider_count+1). Weight scenario means equally. This provides regression coverage with two data draws per fixed scenario. Report the 24 historical families separately to show the composition effect of the added family. Neither panel tunes the method.

Keep the oracle, subgroup indicators and provider identity out of fitting. Record actual score and trajectory grids, timings, environment and independent numerical or data-identity checks. No change to OKAPI search defaults or periodic research execution.

# Where should the standard regression suite expand?

Design review of standard-v2, 24 November 2026. The 20 current scenarios remain the regression suite. The proposals below are hypotheses. They have no benchmark scores yet.

Implementation follow-up: the four candidates entered [standard-v3](standard-v3/README.md). The regression suite now has 24 scenarios. The historical design and observations below explain their selection. [Subgroup and control results](v3-slices/README.md)

## Evidence from existing evaluations

The [independent confirmation](../pattern-iterations/round-3/README.md) has ten primary draws on each of the original 19 scenarios. Restrict its results to clean test rows and the primary search seed (`gp_seed == data_seed`). Compare the regularized native procedure (`challenger`) with probability stacking within each draw. [All 19 paired summaries, including simple averaging and Top-N](clean-paired.csv). The [20-scenario integration check](../class-capacity/standard-v2-smoke.csv) uses one draw and a different sample size. It cannot supply ten observations on the twentieth scenario.

* Weakest link shows the clearest clean shortfall against LR: 0.07216 vs 0.06164 Brier, eight losses in ten. At four independent fair conditions the positive rate is 1/16. Changing their count changes both rarity and rule depth. Vary one while holding the other fixed before attributing the gap.
* On polarity fault a single small bank gave 0.21092 vs 0.11688. The larger historical comparison gave 0.10404 vs 0.10421. Search variation, development size and bank composition are competing explanations.
* The [digits and controlled class-pair study](../class-capacity/README.md) shows a reversal between real provider pools. On synthetic affected labels full LR gained more over diagonal LR with provider-private wrong destinations than with shared destinations. This motivates a shared-to-private axis and class-conditional evaluation.

## Candidate additions

| Problem | Paired generator and negative control | Signature and external check |
| --- | --- | --- |
| Rare conditional expertise | Draw the same latent cases for a reliable generalist and a specialist with a controlled gain in a rare region. Cross region prevalence with the reliability of an observable routing cue. Include zero specialty gain and an uninformative cue. Keep class prevalence fixed across arms. | Compare excess Brier against the posterior conditional on observable inputs, both inside and outside the region. Record realized fit and validation counts. Look for the pattern on out-of-fold predictions using externally defined groups. |
| Apparent extra evidence in a large pool | Draw a calibrated anchor p and labels from Bernoulli(p). Generate alternative scores from p and independent noise, then vary the number of candidate models at fixed labels. Add a positive control in which one anonymous candidate observes extra label information. Permute column roles across banks. | In the null condition p is exactly optimal. Measure test excess Brier vs anchor and the fit/test gap as candidates per fit row grow. One historical failure to transfer motivates the test, but does not establish this mechanism. |
| Unstable predictive shortcuts across environments | Hold a stable provider's class-conditional law fixed. For a second provider make its association with the label positive, zero or reversed depending on environment. Cross the environment proportions independently at fit and test. Include a stationary mixture and a stable-only control. | Report performance within each environment and for the pooled deployment mixture. Within-environment harm and changing mixture weights have different interpretations. Expose environment IDs to the evaluator alone in the base task. |
| Loss of resolution on difficult examples | Let one provider return a valid fallback probability on unresolved cases. Compare random and difficulty-dependent fallback at equal frequency. Offer the availability flag only in a separate explicit treatment, with an always-available reference. | Compare errors on fallback and ordinary rows. This tests whether the fusion rule can detect when an otherwise good source has lost information. Check discretization and repeated probability vectors on real banks before promoting this design. |

Also evaluate prior shift as a protocol on an existing class-conditional generator: fix the distribution of scores given each class and vary the class prior. This isolates changing prevalence from a changing provider or routing rule.

## Candidate severity axes for the current 20

Each row has a mechanism-specific axis. Some require new generator controls. Proposed profiles must be calibrated on diagnostics before release. An endpoint may become easier, or make the signal entirely unobservable.

| Scenarios | Paired intervention |
| --- | --- |
| committee, multiclass_committee | Move provider competence toward chance; at fixed competence vary the number of independent sources separately. |
| deadweight, needle | Add uninformative providers while retaining the same informative columns and labels. Measure candidate count relative to fit rows. |
| correlated, saturated | Increase shared noise at fixed marginal strength. Include exact duplicates as a separate invariance control. |
| calibrated_suppressor | Cross the correlation at which the weak provider's optimal conditional coefficient changes sign. |
| miscalibration | Scale the range of temperature distortions while preserving each provider's decisions. |
| polarity_fault | Vary sign-flip frequency and match the baseline competence of stable providers; report outcomes on flips separately. |
| hardveto, hardrescue, weakest_link | Vary the frequency of the decisive veto, sole rescue or single unsatisfied condition; hold detector quality fixed. Include a separate matched-class-prevalence comparison. |
| two_pathways, compound | Vary the frequency of cases solvable only by the second path or rescue branch. Track branch-specific loss at fixed global prevalence. |
| parity | Vary the accuracy of one latent-bit detector group with the other fixed; chance performance defines a useful information limit. |
| gated_experts | Vary how reliably the selector reveals the active region; vary the specialist advantage separately. |
| native_switch, native_close_switch | At a fixed gate and known threshold vary the mass on examples where choosing the wrong operand materially changes the posterior. Record gate disagreement and oracle margin. |
| compositional_switch | Vary support on corners vs interior at fixed rule; track how often shallow competitors disagree with the exact rule. |
| similar_classes_38 | Mix shared and provider-private wrong destinations at fixed per-provider true-class probability and Brier. Specify nonidentical private matrices in the private arm. Report losses on 3/8 and the other eight classes. |

## Evaluation and promotion

Retain a fixed full-suite reference run with mean, Top-N and LR. In stress runs average seeds within setting, settings within scenario, then scenarios equally. Publish the individual cells. Reuse underlying draws across settings where the generator supports pairing, use separate data and search seeds, and estimate paired uncertainty over independent banks. At rare settings provide enough evaluation examples for a conditional estimate and separately preserve deployment prevalence for the aggregate.

Use failures to propose mechanisms, verify the mechanism against a known posterior or negative control, then test prospective predictions on fresh banks or external out-of-fold pools. Promote a reproducible mechanism with interpretable controls even if several fusion methods fail or a simple baseline wins. Maintain a held-out confirmation suite for method development.

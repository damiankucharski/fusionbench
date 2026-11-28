# Open research log

FusionBench is a general benchmark for prediction fusion with controlled generators, reference datasets and a public scenario interface. OKAPI provides a case study in diagnosing representation, search and generalization. Earlier ensemble-compression studies below are prior results. The active objective and pause switch are in [the research settings](research-state.json).

The [literature index for the paper](literature/README.md) connects primary sources to benchmark assumptions, dated proposals and measured results. It includes the [21 November review](literature/2026-11-21.md) and a [BibTeX bibliography](literature/references.bib).

## Validation optimism and loss of useful candidates · 28 November 2026

Two iterations measured finite-menu selection and tracked complete native populations, then tested paired uncertainty penalties throughout training. With a known optimal mean injected at initialization, standard native GP lost it in 12/12 clustered banks; scoring true conditional risk on the same training features retained it in 12/12. The new penalty helped the clustered null but worsened the 24-scenario suite test by 0.00198 Brier on 144 fresh banks, with no resolved reduction in relative validation optimism. The second round kept the default algorithm. Candidate retention and final deployable selection remain separate targets. [Report, research protocols, code and scores](validation-curse/README.md)

## Standard-v3 and mechanism stress profiles · 24 November 2026

The regression suite now contains 24 scenarios. The four additions cover rare conditional expertise, spurious improvements over a sufficient anchor, environment-dependent shortcuts and difficulty-dependent fallback. On 96 held-out banks, regularized OKAPI scored 0.21656 macro Brier and probability LR scored 0.24662. On six separately seeded banks, shifting the environment mixture moved OKAPI from 0.2852 to 0.7157 and left the stable provider at 0.3772. Next question: can source-only diagnostics identify unreliable provider reliance without knowing which provider will change? [Full-suite scores and protocol](scenario-coverage/standard-v3/README.md) · [Stratified results and paired controls](scenario-coverage/v3-slices/README.md)

## Coverage review of the 20-scenario suite · 24 November 2026

On ten archived clean draws, regularized OKAPI lost to probability stacking in eight for weakest link. The sign-flipping provider showed a large loss in one small-bank run and a near tie in the larger historical comparison. The [coverage review](scenario-coverage/README.md) separates those observations from four proposed mechanisms and proposes severity axes for every standard scenario.

## Standard suite for OKAPI regression evaluations · 24 November 2026

Evaluate each OKAPI change across the entire current standard suite and report per-scenario scores against mean, ranked-prefix Top-N and logistic stacking. Include the suite version and seeds. The current standard-v2 suite contains all 19 standard-v1 scenarios plus `similar_classes_38`. The ready-made ten-class problem pairs labels 3 and 8 for all three providers in both splits. Run `fusionbench run --methods okapi_native_pattern_l2` to evaluate the full suite. The [single-seed integration run](class-capacity/standard-v2-smoke.csv) used 256 rows per split and seed 1729. Targeted diagnostics supplement the full evaluation.

## Shared versus provider-specific class confusions · 24 November 2026

Implemented the missing own-class logistic control, two native gradient-free OKAPI softmax nodes, and a seeded wrong-destination intervention with configurable shared and per-provider matrices. On original digits the full stacker beat the own-class stacker 0.03190 to 0.03402; the ordering reversed on a second provider pool and partition set. On the targeted synthetic classes the full model gained 0.00103 Brier under shared confusions and 0.00439 with provider-specific destinations. The new native full node could replay the production stacker within 8.4 × 10⁻⁸, yet its independent fit on original digits reached 0.03597. Validation selection over evolved subtrees regressed on the shared-pair control. Next question: can a complexity penalty improve held-out tree selection on fresh banks? [Study, code and measured comparisons](class-capacity/README.md)

## What capacity did the digits comparison give each fusion rule? · 24 November 2026

The probability stacker on three ten-class providers fits multinomial logistic regression (L2, C=1) on 30 standardized features. Each of the ten output logits has its own 30 slopes and intercept (300 stored slopes and ten intercepts; redundant parameterizations are possible). The diagnostic symmetric stacker shares three slopes across all classes and uses pooled provider scaling. On original historical digits its mean Brier was 0.03416 versus 0.03190 for unrestricted LR with identical scaling and 0.03193 for the production stacker. The multiclass native logit head shares one weight per input across class slots and combines log probabilities; an input may be a learned subtree. Its training and input transform differ from symmetric probability LR. [Implementation](../../fusionbench/methods/stacking.py) · [Matched-control results](runner-up-transfer/permutation/means.csv)

Persistent class identity gave the stacker useful predictive information under the original supervised fusion task. The per-row permutation removes class prevalence, class-dependent calibration and stable confusions together. These observations do not identify a particular pair of visually similar digits, establish that a confusion law is shared by providers, or indicate leakage. Panel E plants a provider-specific wrong destination on hard examples; its shared difficulty and noise do not supply a configurable common class-pair confusion matrix. A controlled extension would mix shared and provider-specific destination preferences at fixed true-class probabilities and per-provider losses. To attribute the stacker's gain, compare shared slopes, then class intercepts, then class-specific own-class slopes, then a full cross-class map on the same probabilities, scaling and regularization protocol. A native derivative-free node on probability features could express the same affine scores and softmax. Matching the function family leaves optimization and generalization to be tested. [Generator](../../fusionbench/scenarios/multiclass_provider_panel.py) · [Matched intervention](runner-up-transfer/permutation/README.md)

## Runner-up information and class-identity transfer · 24 November 2026

Appending probability margin and repeatability of the strongest incorrect class left all three prospective nearest-scenario choices unchanged. Mean absolute error of the OKAPI-minus-probability-stacking gap was 0.01360 for both baseline and augmentation. The fit profiles of new digits, Iris and Seeds pools exceeded the matched synthetic range on repeatability. A separate intervention on historical digits preserved true-class probabilities while randomly reassigning class slots per row in both fusion-fit and test. OKAPI stayed at 0.03868 Brier and probability stacking moved from 0.03193 to 0.03919. A directed synthetic mechanism showed a much larger loss of stacking's advantage. New external scoring took 19.8 s, with 3.6 s to produce base predictions and profiles in that run. A next controlled study could manipulate persistent wrong-class destinations on easy as well as hard examples while holding true-class probabilities fixed. [Study, controls, forecasts and data](runner-up-transfer/README.md)

## Can FusionBench forecast real fusion performance? · 24 November 2026

Added a controllable multiclass generator with heterogeneous provider quality, shared hard examples, continuous or finite-vote probabilities, and provider-dependent wrong-class evidence. Forecasts made from fusion-fit profiles before opening test outcomes were evaluated on two digits provider banks sharing images and one Wine bank. The expanded library reduced absolute error for the native-OKAPI-minus-probability-stacking gap on all three (heterogeneous digits 0.02961→0.01288, bootstrap-LR digits 0.02700→0.00984, Wine 0.01992→0.00086). It predicted the winner on Wine alone. In the controlled ten-class panel, finite votes moved the native-versus-stacker gap by −0.00043; adding a fixed wrong-class pattern moved it by +0.03422, reversing the ordering in every paired draw. A next test is whether fit-side runner-up and class-confusion profiles improve blind forecasts on *new* provider pools and datasets. Four workers took 240 s on the synthetic panel; external bank builds and scoring took about 40 s on one numerical thread. [External coverage investigation](external-coverage/README.md)

## Three native OKAPI iterations, 24 November 2026

Explicit pairwise MIN features missed the promotion criterion. A variance-scaled penalty inside the native derivative-free logit weight search passed in exploration and on independent confirmation (19 families, ten replicates). It lowered the primary Brier from 0.21524 to 0.20673, paired difference −0.00851 [−0.01079, −0.00624]. In a held-out correlated provider fault it improved from 0.31975 to 0.29914. Probability stacking remained competitive on nine canonical shifts and won on the eight overlapping digits splits. Search-seed sensitivity persisted. A next test could separate class-specific weight capacity from candidate selection on fresh multiclass banks. End-to-end synthetic compute on four workers was at most 53 s in round 1 (timestamp bound), 54.05 s in round 2 and 118.92 s in round 3. The secondary digits run took 39.08 s alongside round 3. From frozen protocol to published result, the first two rounds took about five minutes and 89 seconds respectively; the final confirmation was replayed within five minutes of its freeze. [Protocol, reports, costs and replay](pattern-iterations/README.md)

## FusionBench library and reference datasets, 24 November 2026

The publication scope is the controlled dataset collection and the API for reproducing, modifying and adding scenarios. All 19 default generators have a small, versioned reference dataset. A separate wheel installation reproduced their labels, provider predictions and available posterior arrays at their stored precision. The OKAPI case study illustrates how a known target can distinguish an expressivity limit from a failure of search or selection. Transfer under a distribution change is one of several experiments enabled by the same library. [Library guide](../site/index.md) · [Scenario catalog](../site/scenarios.md) · [OKAPI case study](../site/okapi.md)

## Graded validation-to-test transfer on 24 November 2026

FusionBench now ships configurable smoke and full transfer suites with fixed seed slots, matched control predictions and example extension instructions. A comparison across 19 scenarios, 152 banks and 3,840 paired evaluation conditions found that mild validation stress and selective subtree shrinkage did not improve native OKAPI. The primary difference in Brier was +.00112 (95% CI −.00026 to +.00249). With clean source labels, ranking the same four discovery trees transferred to clean test data in 65.1% of banks and to the strongest logit noise in 27.6%. Training and evaluation took 129.1 seconds on four workers, with about ten minutes from the frozen protocol to the independently verified report. An open question is whether limiting how strongly one provider can change a tree's prediction improves unseen transfer, including a fault mechanism withheld during development. [Results and diagnostics](transfer-grid/README.md) · [Public benchmark guide](../transfer-benchmark.md)

## One native OKAPI tree on 24 November 2026

With score-only logit weight fitting and optional evolved subtrees, one native tree reached .18388 macro-average IID Brier over all 19 scenarios. Probability LR scored .21698, logit LR .22017, the flat native node .21891 and native GP alone .26004. Across nine shifts the native tree trailed probability LR by .01518 and logit LR by .00503. The independent replay verified 2,688 test scores and prediction hashes. The run took 59.4 seconds across three workers and about nine minutes from frozen protocol to verification and report. The next question is whether shrinking the weights on evolved subtrees can preserve the IID improvement under shift. [Report, tree archives and paired comparisons](native-pattern/README.md)

## One round on 23 November 2026: quality against logistic stacking

The cross-fitted native GP and logistic hybrid reached .20219 IID Brier averaged equally over the fixed 19 families, versus .21747 probability LR, .22039 logit LR and .21421 for the same selector restricted to the two logistic models. GP alone scored .26187. The incremental reduction from adding GP was .01202 (95% interval .01012–.01392). On nine shifts probability LR retained a small average lead. Training and evaluation took 156.9 seconds, and protocol, implementation, audit and report took about five minutes. A future comparison could use tuned LR and an external OOF bank. Scheduled work remains paused. [Full results and source](oof-gp-stack/README.md)

## Follow-up on 23 November 2026: pruning the frozen mixed-start winners

The focal balanced-start trees shrank by 7.63 nodes at +.00097 fresh IID Brier, below the prespecified quality cost in the paired upper confidence limit. Two of 16 trees dropped a provider. The primary criterion requiring reliable base-model savings was unmet. Across the 19 families, pruned balanced-start winners used 4.09 providers versus 4.73 intact, with Brier .26114 versus .25861. A future study could test moves that remove all occurrences of one provider, including the root. Automated work remains paused. [Protocol, times, verification and scores](prune-frozen-mix/README.md)

## One authorized round on 23 November 2026

Six balanced 15-node starters and 24 ordinary shallow starters reduced IID full-class Brier on CompositionalSwitch by 0.07073 relative to the same six left-fold starters (95% paired interval 0.02906–0.11240). Across all 19 families the selected balanced arm averaged .25811, ordinary OKAPI .27526 and same-total-label logistic stacking .21995. The balanced winners referenced more providers. A possible follow-up measures which providers can be pruned from those frozen trees at an independently verified quality cost. The scheduled program remains paused. [Results, timings and reproduction](mixed-deep-init/README.md)

## Continuing between rounds

While the research switch is enabled and the user's instructions allow work, the FusionBench and OKAPI leads send the coordinator each published result and start the next falsifiable experiment in the same active run. Record the next question with the result and coordinate directly on code revisions and controls. If a run reaches a limit or a blocker, record the last verified result, the remaining question and the reason work stopped so the next run can resume. A completed round alone does not retire either research lead.

### A fitted gate and its neutral duplicate both changed FAR search

Using the same admission, slot and other parents, the fitted operand improved final FAR excess Brier by 0.00288 (0.00267 versus 0.00555 for the duplicate). A neutral duplicate also improved over matched reordering by 0.00334. The operand comparison was unresolved on the CLOSE and logistic laws. The next registered test bypasses genuinely inherited occurrences on fresh rows. [Report and reproducible evidence](search-regret/matched-graft/README.md) · [Frozen follow-up](search-regret/gate-necessity-protocol.md)

### Bypassing genuinely inherited operations in frozen switch programs

Across all 32 FAR fit-selected winners, bypassing their traced operations increased conditional-excess Brier from 0.00267 to 0.02437 on fresh rows; resetting just their numeric cutoffs increased excess by 0.01115. The corresponding CLOSE losses from bypass and cutoff reset were 0.03368 and 0.01581. The next controlled search test compares operator versus operand point mutations on a target with a population lower bound for any single operation. [Functional necessity, matched baselines and shifts](search-regret/gate-necessity/README.md)

### Organic operator versus input mutation on a compositional target

At 1800 native evaluations, changing one operator family gave 0.05957 excess Brier versus 0.06259 for changing one direct provider on the composition law. The paired interval crossed zero. Generating distinct candidates was easier than surviving ordinary selection. The next prospective experiment separates structural information in features from labels and tests exact compilation of a posterior over possible hidden pathways. [Organic search cases](search-regret/organic-composition/README.md) · [Matching study protocol](search-regret/matching-compression-protocol.md)

## Regression testing across the default suite

Each change motivated by an individual mechanism also faces the remaining default FusionBench families as regression tests. Freeze one configuration before seeing the panel. Publish results and their spread for every scenario, and distinguish mechanism-specific findings from broad improvements. If a table suggests a code change, verify it on fresh data. Use a previously omitted family or an external task before making a transfer claim. [First full-suite protocol](full-suite-point/protocol.md)

### A single shallow network for an average over 10,395 latent matching trees

Pair-marginal networks reproduce the fixed model average through three active inputs; adding quadruple units recovers the average through five. On 12 anonymous providers and 16 labels, extra input-frequency information improved balanced matching-law excess Brier by 0.02559; a uniform design removed the gain exactly. The next run checks the deployable operator-type mutation on the full default suite before changing the search further. [Study, proofs, controls and two PyTorch models](search-regret/matching-compression/README.md) · [Full-suite freeze](full-suite-point/protocol.md)

### Regression results for operator-type mutation across the full suite

Mean IID Brier was 0.25947 with point mutation and 0.26839 without across all 19 canonical scenarios; 15 point estimates favored the extension. Five unadjusted intervals favored enabling it and none favored disabling it. Logistic stacking was strongest on average at 0.22874. Both multiclass cases were included. The largest adverse point estimate was on the 100-provider Needle and remained uncertain. The next question separates actual relabeling from changes in native mutation frequencies on fresh data, with an omitted or external transfer task. [All cases, spreads, control methods and tested shifts](full-suite-point/README.md)

### What adding an operator-type mutation changed

Across all 19 canonical families, native retype lowered IID full-class Brier by 0.01202 versus a genotype-preserving fourth opportunity. Same-family refresh produced little change; in 134/228 synthetic searches the previous family at every chosen site lacked parameters. The same direction held on external Wine at equal fitness calls, although three simple baselines had lower losses there. The next question concerns whether rounding native fitness to three decimal places suppresses useful variation, and whether any fit improvement survives independent testing. [Paired regression and external test](mutation-opportunity/README.md) · [Native selection-precision witness](https://github.com/damiankucharski/OKAPI/blob/630fd1f/experiments/selection_precision/README.md)

### Corrected references for multiclass Gaussian providers

The four- and five-class scenario posteriors lacked the proper signal and covariance factors. Recomputing every affected probability distance changed the first study’s Saturated secondary effect from +0.00273 to −0.00754. All 2,460 matched-model predictions and their true-label Brier values were verified. [Exact weights, reconstruction and corrected tables](multiclass-oracle/README.md)

### Precision changed survivor choices while the finite grammar remained underexplored

Removing native three-decimal rounding changed 134/191 one-step survivor sets, while the 19-case IID Brier difference was +0.00127 [−0.00452, +0.00705]. On the four-provider matched law the exact fit winner was the truth on 12/12 banks and neither precision found it in any of 24 starts. The next question is whether intermediate programs needed for two complementary paths ever coexist in native populations. [Complete suite, exact populations and replay](selection-precision/README.md)

### Availability of complementary subtrees in actual native populations

Exhausting all legal next crossovers on the 32 matched generation-15 populations found no way to produce the true two-pathway rule. All 25,600 scored programs across both laws were equivalent to raw-input chains. A single MIN or MAX joining two possible chains covers all 166 four-input rules, yet only four of 32 matched archives included appropriate whole-root ingredients even by 400 calls. The next intervention tests two branching pairs versus a size-, operand- and type-matched chain at initialization, and how far any gain transfers. [Measured opportunities, grammar certificate and limits](semantic-donors/README.md)

### Matched topology at initialization

On eight anonymous inputs where four implement two hidden pathways, branching initialization reached the true rule in 21/24 GP runs against 11/24 for chains. The estimates after validation differed by −0.00944 IID Brier across twelve banks. A post-hoc bound placed even the best possible mixture of empirically optimal lattice programs above the validation-selected branched GP in ten of twelve banks. The 19-family regression suite gave +0.00044 [−0.00286, +0.00374] in full Brier. The next test connects these results to the earlier research branch's deployment-mixture selector on a fixed pool of trees and measures the loss of identification when the label law changes silently. [Matched initialization, exact reference and full suite](seed-geometry/README.md) · [Original selector](https://github.com/damiankucharski/fusionbench/blob/1dfb93e/fusionbench/eval/observer.py) · [Earlier exact lookup](https://github.com/damiankucharski/OKAPI/blob/c614670/okapi/lattice_search.py)

### Ranking frozen trees under observable and silent changes

At 50% changed-context prevalence, the one-bit deployment-mixture selector improved Brier by 0.01479 versus ordinary validation when context-specific labels were stable, then worsened it by 0.02754 when all changed-context labels reversed. Every unlabeled target observation stayed identical. The paired change across 16 banks was +0.04233 [0.01879, 0.06586]. With 32 target draws, gap-directed auditing cut pair-selection regret from 0.00113 to 0.00053 in the stable condition. An independent native follow-up to the preceding shape study found an immediately productive crossover in 12 of 20 later branched-start discoveries. The next prospective search test studies a minority of deeper starters, pooled versus isolated search and post-selection simplification with independent labels. [Paired drift and paid-label measurements](frozen-pair-ranking/README.md) · [Initial-crossover audit](https://github.com/damiankucharski/OKAPI/blob/47029ef/experiments/initial_crossover/README.md) · [Earlier island experiment](https://github.com/damiankucharski/OKAPI/blob/3eae0e5/experiments/islands/README.md)

## Current research questions

- **Search.** Within a stated tree family, how close does an optimizer get to the best attainable fit score for its budget? In small families, enumerate every admissible tree. In larger ones, record a feasible reference tree and the budget needed to match it. Measure variation across optimizer seeds on identical examples.
- **Generalization.** For each selected tree, evaluate the same predictor on fit, validation, fresh IID examples and examples under controlled distribution shifts. Compare a proposed change with the existing OKAPI search and the three baselines below. Test candidate changes on new data seeds and include settings in which the proposed improvement may hurt.

### Baselines for every OKAPI experiment

1. **Arithmetic mean.** Average every provider's class probabilities with equal weights.
2. **Ranked-prefix Top-N mean.** Fix the scoring metric and its direction before the run. Score providers individually on labeled development predictions and rank them once, best first. Score the arithmetic mean for every prefix from a single provider through all providers in that order. Keep the best-scoring prefix, including when the score falls and rises again as N increases. Break individual-score ties by the fixed input order and equal prefix scores by the smallest N. Save the chosen N and the scoring metric for each fit. The full prefix reproduces the arithmetic mean on the same inputs.
3. **Logistic stacking.** Fit a regularized logistic regression on the provider predictions. State whether the inputs are probabilities or logits, and the regularization used.

Use the same held-out examples and provider columns for all methods. Record which development rows were used to rank, select or fit, and count any separate validation labels in the budget. For learned base models, use out-of-fold or independent development predictions for fusion. Freeze provider order, N and stacking parameters for both IID and shifted tests. For earlier experiments, label any added comparator as a retrospective replay.

The first paired study found that on TwoPathways, with 384 labeled fit examples, 48-generation OKAPI had IID Brier 0.16071, the stacker 0.13275 and a manually constructed feasible tree 0.06744. The tree used the known grouping of detectors. The first milestone is to measure the search gap in a small exactly enumerable problem, and to test whether OKAPI can discover useful compositions from anonymized expert outputs in the original setting at a matched budget. [Matched-baseline study](https://github.com/damiankucharski/OKAPI/blob/639c5ff/experiments/fit_gap/README.md)

## How the two projects evaluate an OKAPI change

The FusionBench and OKAPI leads exchange failure cases, exact revisions and run configurations directly. For a joint comparison, FusionBench owns the scenario, held-out seeds, scoring and reference methods. OKAPI Lab owns the implementation, its regression test and the before/after revisions. Both teams use one protocol and one set of results. The published report links the algorithm change to its measured effect, including null results. A baseline implementation published in FusionBench can be imported by OKAPI experiments on the same data splits, as in the [ranked-prefix replay](https://github.com/damiankucharski/OKAPI/blob/ec1ba02/experiments/fit_gap/README.md).

The next joint comparison concerns two different trees with identical printable representations. The OKAPI regression test asks whether both candidates survive duplicate filtering. The FusionBench experiment asks whether that correction also reduces the gap to the minimum achievable training Brier at fixed search cost, and whether held-out risk changes. The exact reference applies to trees using only MIN and MAX. [OKAPI collision study](https://github.com/damiankucharski/OKAPI/blob/7676602/experiments/structural_key/protocol.md) · [FusionBench exact reference](https://github.com/damiankucharski/fusionbench/blob/1f12dbf/docs/ensemble-study/search-regret/continuous.md)

## Starting evidence

- An ICA projection and a compact logical rule recovered a single predictor in a four-factor bimodal simulator. Isotropic Gaussian sources removed the information needed to recover those axes. A task outside the rule library required a more flexible classifier.
- Under a two-context prevalence shift, an expert-pair disagreement rate helped select one of twelve GP trees. The individual expert's marginal scores were uninformative by construction. The selector used labeled and context-annotated source validation rows and unlabeled target predictions.

[Previous study and raw results](README.md)

## Open experiments

### A calibrated provider can require a negative weight

At high class-conditional correlation, signed logistic fusion of two calibrated predictors reached .08481 IID Brier versus .11304 for the same fit with nonnegative slopes (24 independent banks). Changing correlation alone made the signed fit worse, .20536 versus .11327. For this law we also derive the population Brier minimum over predictors nondecreasing in the weaker provider. Started 2026-11-21 10:07:34 UTC. Compute took 1.71 seconds and time through validated report was 271 seconds. Can a signed OKAPI head match the linear reference and can an unlabeled target bank tell us when to stop subtracting? [Experiment, derivation and replication](calibrated-suppressor/README.md)


1. Train one raw-input network for prediction and monitoring. Compare it with a static predictor, a context-aware reference and a condition where the shift is observationally invisible. Track the monitoring supervision and target batch size alongside predictive risk.
2. Change the law inside a context. Test whether the unlabeled target statistics reveal a violation of the two-context mixture assumption and whether declining to adapt improves held-out risk. Include a failure designed to leave observed statistics unchanged.

The results of each experiment determine which comparisons to repeat on independent seeds, which assumptions to falsify next, and which changes to retain in the libraries.

### Joint predictor checkpoint

The two-seed pilot completed. With paired observations and unit sensor gain, the joint raw-input model achieved 80.80% accuracy, the static predictor 74.26%, and the context-aware reference 82.74%. Breaking the pairing in the unlabeled support data reduced joint accuracy to 74.21%, with the static predictor at 74.26%. A separately trained observer also reached 80.73% with pairing. These preliminary results motivate an independent confirmation and a check on whether joint training offers any benefit over the separate observer.

### Independent confirmation and stress tests

Across eight fresh seeds, the raw-input network with 32 paired, unlabeled support observations reached 80.68% accuracy against 74.21% for the static predictor. Training the observer separately reached the same accuracy. Breaking the informative pairing removed the advantage. The next comparison will test whether the model can identify combinations of changes held out during training and how it compares with an ensemble on the same examples. [Protocol and results](context-student/README.md)

Across eight banks of OKAPI trees and twelve target seeds, a diagnostic detected a strong new expert fault in 96 of 96 trials and a weaker fault in 43 of 96. When only the label mechanism changed, identical target predictions produced identical diagnostic decisions while Brier loss rose from 0.0718 to 0.3267. An open experiment is to measure how many randomly audited labels are needed to detect this invisible shift, including periods without an alarm. [Protocol and results](context-stress/README.md)

### Randomized label audits for invisible changes

The preregistered audit used eight independent banks of twelve GP searches. With sixteen randomly audited deployment labels, a likelihood-ratio alarm detected 74.0% of batches with 15% random label reversal, compared with 61.4% for Brier at a common nominal 5% clean alarm rate. Ordinary log loss reached 72.8%. With reversals confined to the least certain 15% of records, even sixty-four labels gave the likelihood-ratio test 6.2% power, compared with 38.8% for the error count. The next comparison is confidence-stratified versus random label sampling at equal total budget, including shifts near the decision boundary. [Protocol and independent results](sentinel-audit/README.md)

### Equal-budget confidence sampling for invisible changes

In eight independent banks, fixed allocation of a 16-label budget across confidence strata detected 29.7% of batches with silent reversals in the least confident 15% of records. Simple random sampling detected 12.1%, for a paired difference of 17.6 percentage points [0.4, 34.8]. The ranking reversed in a middle-confidence shift (15.1% versus 51.8%). With fixed allocation, conditional detection ranged from 5.9% to 69.1% across banks. The next study will examine whether alternative trees or the original input experts retain label information where the selected tree is uncertain. Computation took 11.77 s and the research cycle through report preparation took 225 s. [Protocol, timing, and full results](stratified-audit/README.md)

### Residual information from providers and alternate trees

On the least confident 15% of the winning tree's examples, a linear readout of its score plus seven provider logits reached 76.6% accuracy versus 67.1% from its score alone (paired difference 9.54 points [6.76, 12.32] across eight new banks). Other GP scores gave 70.6%. Permuting provider vectors within winner-score bins removed the gain. For the same 16 audited labels, detection of localized reversals rose from 58.7% to 92.2%. Next, decompose the Gaussian generator's optimal rule into bounded unreliable-expert evidence and correlated stable evidence, and test learned compact alternatives. Computation took 15.49 s; the cycle through report preparation took 167 s. [Protocol, results, and timing](audit-witness/README.md)

### A bounded-evidence single network from raw sensors

In the registered polarity-fault generator, an unreliable sensor's optimum evidence saturates at log(9) when 10% of observations reverse sign. A label-only model with one learned input cap (two tied ReLU units and a linear skip) reached 93.36% versus 93.59% Bayes, 92.90% raw logistic, 92.89% MLP and 91.12% for a short OKAPI search. It chose the true channel in 8/8 new banks under correlated noise but 3/8 under independence. Turning off the fault made plain logistic preferable. Next, test whether covariance estimated from unlabeled deployment data can transport the signed stable-sensor evidence, including changes in class prevalence. Experiment computation took 56.27 s and independent export verification 4.61 s. The cycle through the report took 294 s. [Derivation, transferable weights and data](bounded-evidence/README.md)

### Unlabeled covariance transport and a common-offset counterexample

Four contrasts orthogonal to the class-strength and common-mode directions isolate conditional sensor correlation. In eight confirmation banks, after correlation fell from 0.65 to zero and class prevalence rose to 0.8, adapting from 1,024 unlabeled examples reduced excess Brier by 0.05258 [0.04929, 0.05588]. GP seed-ranking Spearman rose from 0.613 on source validation to 0.957, with wide paired regret uncertainty. Moving all instrument baselines made the fitted prevalence 0.81 despite the true 0.50 and made adaptation worse than freezing. Next test whether the shape of a sensor distribution can distinguish the two shifts. CPU study time was 11.89 s, elapsed time including preparation through report was 214 s. [Protocol, method and bank results](covariance-transport/README.md)

### A translated score distribution distinguishes prior from offset under clean assumptions

A one-channel empirical class-conditional CDF fit exposed the common sensor offset that fooled means. On 12 new synthetic banks it recovered prevalence 0.501 versus 0.806, reducing excess Brier by 0.11176 [0.10278, 0.12074]. On nine new UCI HAR participants it improved Brier 0.2246 to 0.1099 under an injected shift of 0.6 SD, but worsened 0.0457 to 0.1899 under pure prior shift to 0.8. Low univariate residual did not identify dangerous extrapolation across people. Next test whether three conditionally independent ensemble views can recover class prevalence from centered joint moments despite different offsets, and what breaks under conditional correlation. Validation compute was 2.93 s and elapsed time through report preparation was 236 s. [Method, HAR results and complete data](translated-priors/README.md)

### Recover a single raw sigmoid from unlabeled moments of six complementary sensors

With six positively oriented Gaussian sensors and one unknown conditionally dependent pair, medians across triples identify class contrasts despite different instrument gains and offsets. On 16 fresh banks with 1024 targets, excess Brier was 0.00180 versus 0.09630 for a deliberately contaminated first triple, 0.00646 freezing source, and an even better 0.00077 for a tied-covariance GMM. Dense dependence broke the moments (0.10068) and weak separation hurt the GMM (0.04093 versus 0.01418). With only 128 targets both adaptations lost to freezing under pair corruption. Invisible label drift gives a two-world minimax excess lower bound of 0.01416. Next separate likelihood optimization failures from target identifiability under weak signal. CPU time was 2.94 s and elapsed preparation through report was 214 s. [Derivation, exported models and comparisons](joint-moments/README.md)

### Weak mixture likelihood and single-network posterior need different checks

Across 12 fresh weak-signal banks, an independent unlabeled likelihood validation over four starts and six EM checkpoints reduced excess Brier 0.07120→0.01731 versus the ordinary three-start thresholded search, a paired decrease of 0.05389 [0.01695, 0.09083]. The fixed moments initializer had 0.01447. Even a true-parameter start reached excess Brier 0.03448 after 192 updates as finite-sample training likelihood increased. Reducing class separation all the way to zero makes input density identical for all prevalence choices. The next round tests explicitly budgeted labels for choosing among the same candidates and prevalence-only rules. Final compute took 25.94 s, 106.44 s including reruns and numeric controls, and elapsed preparation through the report was 336 s. [Trajectories, figure and numerical test](weak-mixtures/README.md)

### Sixteen random target labels resolve some latent posterior ambiguities

Updating a frozen pool of six-dimensional classifiers on 16 random labels reduced expected excess Brier from 0.04596 to 0.01554 in the weak-signal condition, a paired reduction of 0.03043 [0.01486, 0.04600] across 24 confirmation banks. Hard selection on four labels made risk worse, while strong-signal mixtures gained approximately zero from the audit. In two worlds with exactly identical input matrices and class priors 0.25 or 0.75, simply counting labels outperformed the pool. The winning arithmetic mixture still requires several sigmoids; whether logit averaging can retain the gain in one exported model remains open. CPU study time was 13.77 s, and elapsed preparation through report was 213 s. [Preregistration, methods and per-bank data](mixture-sentinels/README.md)

### Compress the sentinel-weighted mixture pool algebraically

On 16 fresh weak-signal banks with 16 labels, a single six-weight raw sigmoid matching the ensemble value and derivative at the source of expansion had excess Brier 0.01313 versus 0.01255 for 43 candidates. The paired difference +0.00058 [−0.00002,+0.00118] met the fixed 0.002 margin. Averaging parameters was even simpler (0.01266); actual distillation gave the lowest fidelity RMSE. In a two-sigmoid plateau counterexample, tangent excess Brier was 0.05747 and secant 0.00808, and unequal center/tail slopes prove exact collapse impossible. One batch benchmark gave 0.0104 ms single versus 0.1859 ms for the exactly constant-grouped pool. Next test whether real ensembles offer deployment utility over the strongest label-only logistic baseline before measuring compression. Final CPU time was 14.31 s, elapsed preparation through report 303 s. [Derivation, parameters and measured controls](sigmoid-pool/README.md)

### Verify ensemble utility before claiming real-data compression

In a locked UCI Spambase split grouping identical input rows, six disjoint logistic views achieved 0.10304 Brier versus 0.10287 for the validation-selected member, 0.05804 for full raw logistic and 0.04358 for label-budget-matched logistic using log1p features. On nine held-out UCI HAR subjects, disjoint pooling improved its best member from 0.04781 to 0.02683 but lost to full logistic at 0.01903. In secondary overlapping Spambase views, averaging parameters preserved pool Brier within 0.002 while tangent matching failed; log1p remained better. A pointwise logit-variance certificate at absolute probability error 0.05 covered 41.15% of overlapping Spambase and 100% of overlapping HAR, with zero violations. CPU time was 9.24 s and elapsed preparation through report 236 s. [Utility tests, certificate and exported rules](real-sigmoid-pool/README.md)

### Choose the width of one sigmoid layer with a pointwise error certificate

A Ward merge of logit parameters controls the probability approximation by within-cluster logit variance divided by 12√3. On 16 weak-signal banks four merged units had RMSE 0.00044 versus 0.01605 keeping the four largest terms. Every target record passed the pointwise 0.05 bound. In an eight-axis control, four units had RMSE 0.104 and only 0.59% of targets passed; a fixed training certificate threshold retained all eight axes versus 1.52 mean units on the weak-signal banks. No target predictions from the pool enter compression. Next attempt a multiclass bound invariant to arbitrary shifts of each provider’s class logits. Final compute 13.05 s, 26.09 s including a numerical repeat, elapsed through report 267 s. [Proof, OKAPI interface and bank measurements](width-certificate/README.md)

### A sharp class-count-independent TV certificate for arithmetic softmax pools

The curvature bound for the probability of every subset of classes, sharp at 1/(6√3) times squared directional logit oscillation, yields a simultaneous total-variation bound with coefficient 1/(12√3). Centering logits gives a quadratic raw-input certificate invariant to per-expert affine gauge shifts. On near-identical ten-class digit views, the oscillation bound certified TV≤.05 on 99.17% of points; on overlapping HAR views actual TV≤.05 on 99.37% but the proof certified it on only 0.16%, while certifying the same decision on 77.36%. Neither dataset established general utility over both strong single-model controls. Next examine low-rank certificate execution and direct agreement of component margins. Compute took 238.99 s on one CPU core plus full OKAPI regression 15.37 s; elapsed through report 428 s. [Proof, implementation and held-out results](softmax-certificate/README.md)

### Exact winner routing has limited measured speedup

The centered multiclass certificate factors to at most (K−1)(C−1) projections, and a second logit-gap condition certifies component unanimity. All exact routes reproduced the arithmetic pool class on 12 real source/view conditions and tie controls. Fused-logit unanimity followed by computing probabilities only where needed beat the full pool on 31/36 benchmark medians; full-rank proof routing did so on 11/36, and rank 2 on 5/36. No route beat a full pool for single-record batches. The nonduplicate full-batch HAR near-duplicate case was favorable (1.502 µs/row router vs 1.625 µs agreement check and 2.057 µs full). Next investigate exact two-provider binary decision identities and certificates from paired affine logits. Compute 68.92 s, elapsed through report 367 s. [Method, timings and frozen source experts](softmax-routing/README.md)

### Exact binary pair decisions and local certificates for six experts

Two equally weighted binary logistic experts have the same top class as one raw affine threshold at every input. All 54,360 pair–row evaluations on Spambase and held-out HAR participants agreed after compilation, while probabilities differed by RMSE up to 0.09067 for the selected pairs. On disjoint Spambase the chosen pair raised accuracy from 83.93% for the best member to 89.58%; a full-feature log1p logistic model reached 93.92%. For six experts, three validation-selected midpoint signs certified 77.85% of decisions versus 53.53% using individual unanimity, a 24.32 percentage-point gain [19.92, 28.59]. Sparse MAC cost increased on disjoint views and decreased on overlapping views. Sorting extreme logits exactly characterizes existence of a successful pairing; unequal weights or three equal experts can produce three crossings and defeat a single affine decision rule. Next ask whether monitoring an ensemble at several decision thresholds can preserve ranking or calibration without evaluating every member. Compute 6.54 s, byte-identical numerical rerun 5.55 s, elapsed setup through report 370 s. [Derivation, raw models and measurements](binary-pair-decisions/README.md)

### Compile the standard averaging methods from fitted sklearn classifiers

SimpleAverage and the fitted ValWeighted now expose compile and compile_estimators. Standalone logistic models and StandardScaler/logistic pipelines can map from different feature views into a shared input space while preserving original class labels. Nonsoftmax OvR fits are rejected. The compilers return predictions using the existing FusionBench floor and normalization, a TV contraction, so the source-pool error bounds remain valid. The optional [okapi] dependency now pins a published revision containing the interfaces; wheel metadata and a combined regression run (495 passed, 2 skipped in 51.54 s) were verified. [Public entry points and example](softmax-certificate/README.md)

### Pausing or redirecting future experiments

The scheduled worker reads [the research settings](research-state.json) on this branch before starting an experiment, before expensive computation and before publishing. Set `enabled` to `false` to skip new experiments. Change `objective` to redirect subsequent work. Changes take effect when a worker next reads the file. A process already computing can continue until that check.

### When a reference copies a weak expert

With 32 unlabeled observations and 12 independent source/calibration/target draws, an OKAPI selector switched from Brier 0.0948 to 0.2541 when the weak expert and the reference acquired shared errors. The eight possible observable sign triples exactly match those of the opposite clean regime. A second independent, stable reference restored 0.0943. Falling back to the equal mixture on reference disagreement achieved 0.1453. The next study tests whether the identity of the faulty reference can be identified if either could be copying the weak expert. The round started at 2026-11-20 23:19:51 UTC. The 24 GP fits and the evaluation took 7.52 seconds. The elapsed time through validated code, results and report was 209 seconds. [Study and frozen protocol](reference-copy/README.md)

### Two unknown witnesses and one contaminated reference among three

All 16 observable sign patterns can have identical probabilities when the more reliable primary sensor and the faulty reference are both exchanged. With three witnesses and one strong copier among two weak healthy references, median prevalence gave Brier 0.1375 versus 0.2242 for pooled features at 32 support observations. The median failed when two channel identifiers duplicated one copier (0.2550). The next round tests whether an active, label-free input intervention can reveal a direct observer dependency and when shared upstream noise defeats it. The round started at 2026-11-20 23:23:29 UTC. GP and evaluation took 2.36 seconds, and the time to validated report was 220 seconds. [Study, exact law and frozen protocol](unknown-references/README.md)

### Changing a reported model input to diagnose a witness

Four paired interventions identified a directly dependent OKAPI threshold witness in 98.46% of deterministic batches and reduced binary Brier from 0.1446 to 0.0953. Observationally identical shared upstream noise removed the response. With 10% stochastic readouts, the reference-index heuristic declared an effect in 64.15% of four-record batches even when neither witness could respond. The next comparison asks if randomized paired probes can control false declarations while preserving useful deployment decisions. The round started at 2026-11-20 23:27:18 UTC. GP and evaluation took 1.27 seconds; time through validated report was 221 seconds. [Study and frozen protocol](intervention-witness/README.md)

### Randomized paired probes with controlled false declarations

A two-witness e-process detected 91.48% of direct dependencies and used 30.99 of 64 available paired records under order-dependent noise. Brier was 0.1045 versus 0.1446 for staying with the candidate mixture. Under no intervention effect the method declared a dependency in 1.94% of randomized batches versus 84.29% with every intervention scheduled second. The next step is to distinguish functional input dependence from unique predictive utility in a learned fusion model. Started 2026-11-20 23:31:09 UTC. GP and scoring took 2.74 seconds and the elapsed time through the validated report was 204 seconds. [Study and formal assumptions](randomized-probes/README.md)

### Exact linear compilation and duplicate information on handwritten digits

On five held-out folds a centered-log stack of two view-specific linear experts compiled to one 650-parameter raw softmax within 1.23e−15. The compiled Brier was 0.07846. A single raw softmax trained on base labels achieved 0.06731. Permuting one exact clone moved Brier by 0.1642; exact twin substitution changed zero outputs. Preserving ridge multiplicity after deduplication restored all probabilities to within 8e−10. Next test is exact compilation and simplification for binary evolved min/max OKAPI programs. Started 2026-11-20 23:34:54 UTC. Training and scoring took 6.18 seconds and time through validated report was 286 seconds. [Study and frozen protocol](compiled-views/README.md)

### Compiling evolved min/max programs into raw residual ReLU circuits

All eight native trees compiled within 5.56e−16 on independent raw queries. Their total natural gate count fell from 17 to 16, while all artificial absorption shells disappeared. Only 2/8 GP programs recovered every majority corner. Reinitializing the selected topology gave Brier 0.1063 versus 0.1280 for a dense MLP and 0.0952 for the known median topology. Next test is whether exact signature enumeration solves search and how much view membership contributes to label-only training. Started 2026-11-20 23:39:56 UTC. Compute took 13.74 seconds and time through validated report was 333 seconds. [Study and frozen protocol](lattice-compile/README.md)

### Exhaustive small-lattice search and the price of a wrong view partition

Exact Brier search across 18 distinct rules recovered majority in 12/12 new problems, versus 0/12 for the previous GP budget. A single median network trained from raw inputs and final labels reached 97.96% with known matching blocks, but 77.20% with equally sparse crossed blocks. After rotation the old masks scored Brier 0.15987 versus 0.08403 with dense leaves. Next test asks whether smooth gradient routing helps a dense student recover independently useful latent directions. Started 2026-11-20 23:46:10 UTC. Compute took 56.18 seconds and time through validated report was 181 seconds. [Study and frozen protocol](lattice-views/README.md)

### Discovering three latent directions with a smooth majority training loss

With the same median decoder on both 30-parameter networks, smooth training on 512 final labels improved matched axis cosine from 0.798 to 0.952 and sign-preserving gain-shift Brier from 0.1560 to 0.1030 on 12 independently rotated problems. Starting from three exact clones left their learned directions collinear. Next test estimates these axes directly from the first and third supervised Gaussian moments. Started 2026-11-20 23:49:24 UTC. Compute took 43.81 seconds and time through validated report was 161 seconds. [Study and frozen protocol](majority-routing/README.md)

### Recovering three hidden majority axes from labeled first and third moments

For three orthogonal signs on Gaussian inputs, the labeled mean identifies their sum and the third Hermite tensor identifies their plane and orientation up to permutation. At 2048 labels direct estimation gave mean axis cosine 0.989 and gain-shift accuracy 90.17%, versus 77.05% with randomized phase in the same plane. At 128 labels the first-moment linear classifier still won IID. Next test checks predictive transfer on genuine datasets without assuming majority labels. Started 2026-11-20 23:52:20 UTC. Compute took 37.82 seconds and time through validated report was 185 seconds. [Study, derivation and frozen protocol](majority-moments/README.md)

### Genuine-label falsification of the three-leaf majority prior

The moment-initialized student lost to equal averaging of full-feature logistic, RBF and forest models on each of five Wisconsin folds, five digits-parity folds and nine held-out HAR stair-direction subjects. Brier was 0.03184 vs 0.02131, 0.08040 vs 0.02408 and 0.07105 vs 0.02198. A full-feature single logistic or RBF model beat that ensemble on each dataset; family choice was retrospective. Next test selects a single family without test labels on new tasks. Started 2026-11-20 23:55:47 UTC. Compute took 29.42 seconds and time through validated report was 187 seconds. [Study and frozen protocol](majority-transfer/README.md)

### Choosing a single 64-landmark radial network on three real datasets

A nested inner-Brier rule selected and deployed a single supervised radial network on all Banknote and Ionosphere folds (Brier 0.00045 vs 0.00340 and 0.03313 vs 0.04659 for equal ensemble averaging). On Spambase it kept the ensemble in all five folds; mandatory compact selection scored 0.06790 versus 0.04764. All 15 decisions agreed with the held-out 0.005 tolerance, across only three tasks. Next controlled question is how much of the hidden-axis recovery theorem survives non-Gaussian examples. Started 2026-11-20 23:59:13 UTC. Successful compute took 46.90 seconds; a first numerical-guard abort cost 42.06 seconds. Time through validated report was 276 seconds. [Study and frozen protocol](compact-selection/README.md)

### Hidden majority axes from uniform directions with arbitrary training radii

Row normalization and a d/(d+1) correction recover the Gaussian population tensor up to scale and make the fitted model invariant to positive per-row gains. Under 2% training contamination by a factor of 40, 16 independent problems gave axis cosine 0.9939 vs 0.5650 and gain-shift accuracy 91.46% vs 74.47%. A directionally tilted control gave a sum cosine of 0.8558 at n=8192 versus an analytic limit 0.8581. Next test asks whether a revealed covariance direction and contradiction with the final label can recover the other two axes in the tilted distribution. Started 2026-11-21 00:04:05 UTC. Compute took 4.13 seconds and time through validated report was 151 seconds. [Study and frozen protocol](angular-majority/README.md)

### Recovering two unobserved votes from contradictions to a third

At n=2048 a true covariance spike plus final-label contradictions recovered mean axis cosine 0.9967 and gain-shift accuracy 94.36% versus 73.91% for angular recovery alone. Dropping a pure nuisance spike reached 92.22% versus 71.98% when falsely treated as a vote. A 45° mixture still agreed with labels 74.24% but gave only 82.66% shifted accuracy, below the smooth network's 85.90%. Next test computes how often two candidate vote rules disagree under biased training versus target inputs. Started 2026-11-21 00:07:12 UTC. Compute took 63.33 seconds and time through validated report was 207 seconds. [Study, population derivation and frozen protocol](majority-anchor/README.md)

### Exact label coverage limit for two candidate majority directions

Two known candidates can disagree on 12.5% of targets yet only 0.03754% of training observations. Even with 512 clean labels their Bayes-optimal target Brier is 0.02578; 7978 labels give 95% chance to distinguish them. A shared four-threshold, two-ReLU network represented the exact posterior, and 24,576 independent confirmation trials measured Brier 0.02598 vs 0.05240 for hard selection. Both simulations' uncertainty-frequency intervals narrowly missed the analytic value in opposite directions. The next question is continuous angle uncertainty with many compatible candidates. Started 2026-11-21 00:10:56 UTC. Initial and confirmation compute took 1.38 and 1.95 seconds. Time through validated report was 206 seconds. [Study, formulas and registered confirmation](majority-coverage/README.md)

### A continuum of possible votes collapses to two minimax endpoints

For a consistent vote-angle interval the midpoint minimizes worst classification error and the two-endpoint average minimizes worst binary Brier. At k=15 and n=512 the exact expected target ambiguity was essentially the same for true 0° and 30° (13.761% vs 13.759%), despite different asymptotic rates. At one million labels it was 7.716% vs 1.982%. The next question is which general classes of symbolic ensembles admit two bounding representatives, and how architectural restrictions break that result. Started 2026-11-21 00:14:42 UTC. Compute took 10.60 seconds and time through validated report was 229 seconds. [Study and derivation](majority-interval/README.md)

### Exact minimax envelopes for OKAPI's monotone Boolean programs

For every nonconstant monotone rule on four bits, both data-derived extremal functions were feasible and their average gave exact worst Brier D/4 for disagreement mass D. At 32 source labels the endpoint worst Brier was 0.05759, versus 0.06999 for uniform averaging and 0.21533 for selecting the simplest consistent program. Restricting the class to four single-coordinate projections broke closure, and the best two-member mixture had Brier 0.375 versus 0.1875 for all four. With 16 inputs the true three-vote task still left 48.36% of query predictions ambiguous after 512 labels. An open question is which constraints survive label errors and retain a usable calibrated prediction. Started 2026-11-21 00:19:12 UTC. Compute took 5.56 seconds and time through validated report was 153 seconds. [Study and frozen protocol](monotone-envelope/README.md)

### Exact empirical minima stay closed under label noise, error budgets can break closure

A single cut recovered both extreme empirical Hamming-loss minimizers in all 144 exhaustive comparisons. At 12 labels the true rule belonged to that set in only 51/72 tasks. The correct one-error budget always covered truth, but its extremal functions were infeasible in 14/72 tasks. There the certified Brier center reduced worst risk by 0.01974 on average. The next research question is whether a small active collection of programs can certify this center without enumerating the full noisy candidate class. Started 2026-11-21 00:22:06 UTC. Compute took 0.54 seconds and time through validated report was 235 seconds. [Study and derivation](noisy-monotone/README.md)

# Open research log

The program has two objectives. Improve the reliability and diagnostic value of OKAPI and FusionBench. Find conditions under which one predictor can learn the useful behavior of an ensemble. Each entry records the question, comparison, evidence and next experiment. Completed experiments are checkpoints for continued work.

## Starting evidence

- An ICA projection and a compact logical rule recovered a single predictor in a four-factor bimodal simulator. Isotropic Gaussian sources removed the information needed to recover those axes. A task outside the rule library required a more flexible classifier.
- Under a two-context prevalence shift, an expert-pair disagreement rate helped select one of twelve GP trees. The individual expert's marginal scores were uninformative by construction. The selector used labeled and context-annotated source validation rows and unlabeled target predictions.

[Previous study and raw results](README.md)

## Open experiments

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

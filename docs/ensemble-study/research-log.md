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

### Pausing or redirecting future experiments

The scheduled worker reads [the research settings](research-state.json) on this branch before starting an experiment, before expensive computation and before publishing. Set `enabled` to `false` to skip new experiments. Change `objective` to redirect subsequent work. Changes take effect when a worker next reads the file. A process already computing can continue until that check.

### When a reference copies a weak expert

With 32 unlabeled observations and 12 independent source/calibration/target draws, an OKAPI selector switched from Brier 0.0948 to 0.2541 when the weak expert and the reference acquired shared errors. The eight possible observable sign triples exactly match those of the opposite clean regime. A second independent, stable reference restored 0.0943. Falling back to the equal mixture on reference disagreement achieved 0.1453. The next study tests whether the identity of the faulty reference can be identified if either could be copying the weak expert. The round started at 2026-11-20 23:19:51 UTC. The 24 GP fits and the evaluation took 7.52 seconds. The elapsed time through validated code, results and report was 209 seconds. [Study and frozen protocol](reference-copy/README.md)

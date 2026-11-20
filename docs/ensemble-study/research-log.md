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

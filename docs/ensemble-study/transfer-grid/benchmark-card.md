# FusionBench transfer grid

**Purpose.** How well does a fusion rule selected on labeled development predictions transfer when providers change? Each row contains a probability vector per provider and one integer class label. The grid measures both the deployed rule's full-class Brier and the validation selector's ability to rank its candidates. [Frozen protocol](protocol.md) · [Generator API](../../../fusionbench/eval/transfer_grid.py)

## Source and labeling

The 19 scenarios are deliberately chosen synthetic mechanisms. Seventeen are binary. Saturated has four classes and MulticlassCommittee has five. [Registered constructors](../../../fusionbench/scenarios/registry.py)

| Scenarios | How predictions and labels arise |
| --- | --- |
| Committee, DeadWeight, Needle, Saturated, CorrelatedRedundancy, CalibratedSuppressor, PolarityFault, Miscalibration, MulticlassCommittee | A sampled class or signed class signal conditions the provider scores. Constructors vary informative sources, shared noise, polarity faults or confidence distortion. |
| HardVeto, HardRescue, WeakestLink, TwoPathways, Parity, Compound | A Boolean function of latent binary conditions determines the label. Providers are noisy detectors of individual conditions. |
| GatedExperts | Sample a class and region. The region selects the informative specialist. A separate provider gives a noisy indication of region. |
| NativeSwitch, NativeCloseSwitch, CompositionalSwitch | Sample six discrete scores, then a Bernoulli label. The default conditional laws use FAR, CLOSE or alternating MIN/MAX. These cases deliberately align with particular tree functions. |

**Provenance and rights.** The grid generates all rows numerically without human records, private prediction logs or external datasets. The separate pulsar study documents HTRU2 under CC BY 4.0. The pinned OKAPI dependency has an MIT license. At the protocol revision FusionBench had no top-level license file or project license declaration. The owner still needs to specify reuse terms for benchmark code and generated release artifacts. [Manifest](../../../pyproject.toml) · [Pinned OKAPI license](https://github.com/damiankucharski/OKAPI/blob/0e88dbe762f5fbdf6b8c043aa1aa71908fd1dea8/LICENSE) · [HTRU2 provenance](../pulsar-affine/README.md)

## Units and transfer conditions

Eight seeded replicates per scenario give 152 banks. Each has 640 development rows (448 discovery, 192 validation) and 2,048 independent clean test rows. At training noise levels 0, 0.15 and 0.30, nested masks replace both discovery and validation labels by a uniformly chosen wrong class with the stated probability. The test evaluator retains the generator labels.

All scenarios have a clean test and two common stresses at 0.25, 0.50 and 1.00. In a provider swap, severity is the probability that one fixed provider's vector comes from a different row. In log jitter, severity scales Gaussian noise added to class log probabilities before renormalization. Each stress shares its random draws across severity levels.

Only nine scenarios have an additional native shift. Three weaken detectors (TwoPathways, Parity, WeakestLink), three change correlations (CorrelatedRedundancy, CalibratedSuppressor, PolarityFault), and two replace a gate label law with a logistic law (NativeSwitch, NativeCloseSwitch). CompositionalSwitch removes its corner-sampling component while preserving its label law. Intermediate severity mixes **complete prediction–label pairs** from the endpoint populations. It controls the fraction of alternative-domain examples, leaving each endpoint's parameters fixed. [Endpoint definitions](../../../fusionbench/eval/full_suite_point.py)

**Oracle boundary.** A post-corruption Bayes predictor would condition on the scores the method actually observes. Swaps or jitter can hide information used by the clean posterior. With hidden domain membership, combining endpoint posteriors generally needs membership probabilities conditional on the observed scores. The grid scores realized labels. Its retrospective selection regret compares the clean-validation winner with the lowest test Brier among the same four frozen, pre-refit candidates. [Candidate diagnostics](run.py)

## Independent validation before publication

Full-class Brier sums squared probability errors across classes and averages across rows. Its range is 0–2, and its binary value is twice positive-class Brier. The 3,840 noise-by-test cells reuse rows, models and perturbations. The primary interval uses **eight paired replicate macro-averages**, weighting the 19 scenarios equally. It measures seed variation on the fixed suite. Repeated design against published suites can favor methods adapted to their mechanisms even with fresh test rows. [Primary estimand](protocol.md)

1. **Replay.** Pin both repositories and the numerical environment. Have an independent evaluator regenerate banks and verify per-bank scores, choices, costs and paired endpoints. Cross-check Brier and one corrupted bank against an independent implementation. Record any discrepancies with revisions, seeds and tolerances. [Archive verifier](verify.py)
2. **Hold out a mechanism.** Before exposing results, document the law, label source, observable information and a negative control. A candidate addition is a persistent provider outage shared within a site, using held-out sites as independent units. Freeze tuning budgets and endpoints. Give every method identical rows, labels and access to target information. Compare with the mean, all-prefix Top-N and both regularized logistic stackers.
3. **Check an observed deployment.** Evaluate independently obtained provider scores and native labels. Use disjoint or out-of-fold development predictions and hold out whole sites, subjects or time periods as appropriate. Report absolute risks, paired differences and costs by population, including reversals.

In 2026 NeurIPS named the former Datasets & Benchmarks track **Evaluations & Datasets**. Its call explicitly included data generators, benchmark suites and analyses of how evaluation changes conclusions. The 2026 call required code at submission for a reusable executable benchmark. This project's scientific claim concerns which changes in development and deployment cause a validation ranking to stop predicting test performance, and how much that choice costs. Check the call for the intended submission year before preparing the release. [NeurIPS 2026 Evaluations & Datasets call](https://neurips.cc/Conferences/2026/CallForEvaluationsDatasets)

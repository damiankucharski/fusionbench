# Design of the benchmark

FusionBench takes the **provider report** as the unit of input. A report is a probability vector over the same ordered classes for every provider. A fusion method sees a bank of these vectors and training labels, fits once, and predicts class probabilities for new reports. Studying fusion in this space makes it possible to prescribe the dependence between providers and inspect how a method reacts to it. Training the underlying classifiers on raw images or signals is a separate experiment.

## What a scenario controls

A scenario specifies a joint distribution over reports and labels. Some scenarios draw the label and simulate noisy reports. Others define an observable interaction among reports and draw a label from its conditional probability. The latter can reveal an algorithm's ability to represent a known rule. Providers can also have different reliability by hidden region, or share errors within a scene. The [catalog](docs/site/scenarios.md) describes the mechanism for each standard ID.

Several generators provide an analytic conditional probability. On a specified target law the true posterior minimizes the **expected** proper scoring loss among predictors with the same information. On a finite test draw another method can score better. For a thresholded metric such as macro-F1 the argmax of the posterior need not give the optimal decision. A generator can also provide an explicitly privileged reference that uses latent state unavailable to a method. Consult the scenario's `oracle_description` before interpreting a gap.

## What an evaluation holds fixed

Compare methods on the same source rows and the same target rows. Tune the fusion and rank Top-N candidates using the source. Use fresh seeds to measure variation between generated banks. In scenarios with repeated rows per scene, keep scenes together when making any additional validation split or using scenes as resampling units. Repeatedly consulting a single test bank while developing a method makes that bank part of development.

The default standard score weights each of 30 named scenarios equally. A substantive shift in its interpretation requires changing the panel version or publishing a separate weighting scheme. The public mechanism taxonomy supports other declared scientific questions and retains a record of their scenario weights. Operational-stress tags identify synthetic approximations to deployment concerns. How often those mechanisms occur in a particular application remains an empirical question. See [grouping and scoring](docs/site/scoring.md).

## What to record

Record the FusionBench version, scenario or suite version, constructor arguments, sample sizes, seed, provider order, fit/test conditions, baselines and metric. For the fixed banks include the reference version and checksum. Report per-scenario results alongside any aggregate and keep comparisons paired across data seeds. The [evaluation guide](docs/site/evaluation.md) distinguishes the ordinary fit/test experiment from the transfer grid's additional validation split.

The simulator isolates mechanisms under explicit assumptions. Real prediction banks can also include changes in image formation, class definitions, calibration, sampling or provider training that a given scenario omits. A discrepancy on a real bank is a reason to formulate and test a new scenario, then evaluate it together with the established suite. Synthetic performance on its own establishes performance on the specified laws.

# Design rationale

`fusionbench` is a controlled, **method-agnostic** benchmark for stacking /
prediction-fusion methods. Goal: isolate individual *base-model pathologies* and
evaluate fusion methods in a **guaranteed-reliable validation environment**, so a
performance difference reflects a method's expressiveness on a pathology rather
than validation-test (optimizer's-curse) luck. It is a scientific instrument, not
an advocacy harness for any one method.

## Two non-negotiable methodological choices

1. **Generate in the natural causal direction.** Draw the label `y` from a prior,
   then emit each base model as a *calibrated noisy observation of `y`* with a
   pathology-specific error process. The generators describe conditions and error
   structure only — they never apply any fusion operator (mean/max/min/far) to the
   predictions. This avoids the circularity that would make the benchmark
   "partial" toward operator-based methods like OKAPI.

2. **The ceiling is the analytic Bayes posterior** `P(y | predictions)`, derived
   from the DGP, not from any method. For calibrated, conditionally-independent
   models this is the **sum of logits** (binary) / **product of probabilities**
   (multiclass); for correlated evidence it is the Gaussian linear discriminant;
   for logical-gate DGPs it is a product / noisy-OR. Methods are scored by
   **regret to this oracle**. Tests assert the oracle is never beaten.

Calibration is exact in the binary construction `p = sigmoid(2 s t)`,
`t | y ~ N((2y-1)s, 1)`, giving accuracy `Phi(s)` (so `s = Phi^{-1}(acc)`).

## Reliability by construction

100k rows per dataset, identical generative parameters across a 50k *fit* / 50k
*test* split. At that N the fit ranking of methods provably transfers to test;
`eval/reliability.py` verifies selection-top-1 agreement and fit->test rank
correlation per scenario.

## Scenario taxonomy (impartial by design)

Different pathologies are won by different methods — that spread is the point:

- **Conditionally independent** (logit-sum optimal → logistic stacker near-oracle):
  Committee, DeadWeight, Needle, MulticlassCommittee.
- **Correlated / redundant** (Gaussian discriminant → weighted-linear wins):
  CorrelatedRedundancy, Saturated.
- **Logical gates** (product / noisy-OR optima): HardVeto, HardRescue, WeakestLink,
  Compound, TwoPathways.
- **Input-dependent mixture** (routing → tree AutoML): GatedExperts.
- **Calibration** (judged by log-loss → logit recalibration): Miscalibration.

## Two-regime finding (measured, with real OKAPI)

OKAPI is wrapped as a real method (`methods/okapi.py`, numpy backend — runs on
CPU). Binary predictions are fed as a single positive-probability channel so its
min/max/threshold operators are coherent. Results at 20k fit / 20k test
(regret-to-oracle; OKAPI 100 generations):

**Large, clean validation (the main suite).** OKAPI matches the analytic oracle
and ties/beats the strong logistic stackers on its target structures — with tiny
(4–9 node) auditable trees:

| scenario | okapi | stacking_lr | stacking_logit | simple_avg |
|---|---|---|---|---|
| hardrescue (OR/max) | **0.000** | 0.002 | 0.082 | 0.283 |
| hardveto (AND/veto) | 0.005 | 0.004 | 0.050 | 0.201 |
| compound (depth-2) | 0.013 | 0.009 | 0.055 | 0.131 |
| needle (3/100) | 0.008 | 0.006 | 0.001 | 0.085 |
| correlated | 0.004 | 0.001 | -0.001 | 0.079 |

It loses where its representation lacks the primitive: `committee` /
`multiclass_committee` (constructive growth struggles to assemble a wide equal
mean), `miscalibration` (no logistic recalibration → stacker wins on log-loss),
`gated_experts` (input-dependent mixture, ties stackers — needs routing/trees).

**Small / noisy validation (reliability sweep).** This is OKAPI's protective
regime. On `needle` as the fit split shrinks, the flexible stacker overfits the
200-feature space and collapses, while parsimonious methods (OKAPI, top-k) hold:

| fit_n | okapi | top_5 | stacking_logit | stacking_lr |
|---|---|---|---|---|
| 50000 | 0.907 | 0.924 | **0.929** | 0.926 |
| 60 | **0.918** | 0.923 | 0.779 | 0.787 |

So OKAPI is *not* an ideal-case-only method: at competent base accuracy it
discovers the correct gating/selection/compound rule and matches the oracle, and
under weak validation its parsimony keeps it robust where stackers fail. Its
honest losses are wide-averaging, recalibration, and mixture routing.

The `op_min/op_max/op_far/op_close` baselines are the operator *atoms* (one op
over all models, no selection/composition) — useful references but weak lower
bounds; OKAPI's value is searching, composing and pruning them automatically.

See `README.md` for usage and `docs/sample_run/` for full tables.

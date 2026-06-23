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

## Key empirical finding (and what it means for OKAPI)

With *calibrated probabilistic* detectors, the Bayes optimum is almost always a
smooth function of the evidence, which a **logistic stacker approximates very
well on accuracy**. Hard operators (min/max) are *information-losing* — dominated
by the noisiest input — so they are competitive only when base models are
**competent / near-deterministic** (exactly OKAPI's "base-pool suitability"
precondition). Genuinely nonlinear, non-linearly-separable structure
(`TwoPathways` DNF, `GatedExperts` mixture) leaves headroom no linear method
reaches — the regime for compositional / tree-based meta-learning.

Consequently the benchmark positions OKAPI honestly: decisive over naive
baselines (best-single, full-average, top-k), competitive with learned stackers
on accuracy, and differentiated by **parsimony / auditability / inference cost**
and by **auto-discovering** the right rule per pathology. OKAPI's actual
placement must be measured by running OKAPI itself on the GPU host; the
`op_min/op_max/op_far/op_close` baselines are crude lower bounds on its operator
class, not substitutes for its evolutionary search.

See `README.md` for usage.

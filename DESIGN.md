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

## OKAPI integration and search

OKAPI is wrapped as a real method (`methods/okapi.py`, numpy backend — runs on
CPU). Three things were needed to make it competitive, all in the adapter:
- **binary as a single positive-probability channel** `[N,1]` so min/max/threshold
  operators are coherent (per-cell ops on `[N,2]` mix `min(p)` with `1-max(p)`);
- **population seeding** with diverse depth-1 compositions (min/max/mean over
  random model subsets) so crossover has the building blocks to assemble deep
  rules — standard GP diverse-init, not the target rule;
- **multi-restart** + configurable parsimony to reduce the GP's variance.

Before these, OKAPI collapsed to mean-level and lost everywhere; after, it
realises its representation.

## Where OKAPI wins, ties, and loses (measured)

A logistic stacker is a very strong baseline: for *calibrated, stationary*
probabilities the Bayes optimum is a smooth monotone boundary that L2-logistic
fits with little data. So OKAPI does **not** beat it by a wide margin on the
simple monotone structures — it **ties** them (committee, correlated, deadweight,
needle, saturated, hardrescue, hardveto, weakest_link, compound: regret within
~0.01) while being far more parsimonious/auditable, and it **dominates** the naive
consensus baselines (avg/top-k/best-single) everywhere.

OKAPI's *clear* wins come in two regimes:

**1. Genuinely non-linearly-separable structure (even at perfect reliability).**
Linear methods — including the logit stacker — *provably cannot* represent
mixture-routing or XOR; OKAPI's per-instance threshold operators and composition
can. Test macro-F1 (mean ± std over 3 seeds, 10k/10k):

| scenario | **okapi** | stacking_lr | stacking_logit | simple_avg | oracle |
|---|---|---|---|---|---|
| gated_experts (mixture) | **0.786 ±0.006** | 0.730 | 0.734 | 0.701 | 0.869 |
| parity (XOR) | **0.692 ±0.004** | 0.470 | 0.498 | 0.496 | 0.951 |

These are robust, clear wins over every linear/consensus method. (A tree-based
AutoML such as TPOT can also represent these and is untested here; OKAPI's edge
over it would be parsimony/auditability, not necessarily accuracy.)

**2. Validation optimism — the verification gap (OKAPI's design thesis).** When
base models look better on the fit split than they generalise (realistic), the
flexible stackers degrade as they overfit the spurious fit signal. On `hardveto`
(a gating structure), test macro-F1 vs fit-optimism `delta` (mean of 2 seeds):

| delta | okapi | stacking_lr | stacking_logit | avg / top-5 |
|---|---|---|---|---|
| 0.0 | 0.931 | 0.932 | 0.912 | 0.73 |
| 0.2 | 0.894 | 0.920 | 0.863 | 0.73 |
| 0.4 | 0.861 | 0.861 | 0.825 | 0.73 |

Here the honest reading is narrower than a clean win: OKAPI and the *L2-logistic*
stacker degrade together and stay close, both clearly above the *logit* stacker
(which overfits hardest) and far above the naive consensus methods (which cannot
gate at all). So optimism widens OKAPI's margin over `stacking_logit` and the
naive baselines, but `stacking_lr` (with L2) remains a strong, robust competitor.

**Honest losses:** `miscalibration` (needs a logistic recalibration link OKAPI
lacks → stacker wins on log-loss) and `two_pathways` (a depth-3 DNF its GP search
still under-realises — the analytic oracle shows the headroom is reachable, so
this is a concrete OKAPI search-improvement target, not a representation limit).

The `op_min/op_max/op_far/op_close` baselines are the operator *atoms* (one op
over all models, no selection/composition) — useful references but weak lower
bounds; OKAPI's value is searching, composing and pruning them automatically.

See `README.md` for usage and `docs/sample_run/` for full tables.

# fusionbench

A controlled benchmark for **stacking / prediction-fusion methods**. It isolates
specific *base-model pathologies* (the issues fusion is meant to solve) and
evaluates methods in a **guaranteed-reliable validation environment**, so an
observed difference reflects a method's expressiveness on a pathology — not
optimizer's-curse / validation-test luck.

## What makes it different

- **Reliability by construction.** Each dataset is one i.i.d. generative process,
  split 50k *fit* / 50k *test* from identical parameters. At that size the fit
  ranking of methods provably transfers to test. A reliability report verifies
  selection-top-1 agreement and fit→test rank correlation per scenario.
- **Not circular / not partial.** Data is generated in the natural causal
  direction (label → calibrated noisy base predictions with a pathology-specific
  error process). The ceiling is the **analytic Bayes posterior** `P(y | preds)`,
  derived from the DGP — *never* by applying any method's operators to the
  predictions. Methods are scored by **regret to that oracle**.
- **Impartial by design.** Different pathologies are won by different methods:
  logit-space stacking on conditionally-independent / calibration scenarios,
  weighted-linear on correlated ones, gating (min/max) on logical-gate scenarios,
  tree AutoML on input-dependent mixtures.

## Scenarios (base-model pathologies)

Committee, DeadWeight, Needle, MulticlassCommittee (conditionally independent →
logit-sum optimal); CorrelatedRedundancy, Saturated (correlated/redundant →
weighted-linear); HardVeto, HardRescue, Compound (logical gates → nonlinear
min/max optimal); GatedExperts (input-dependent mixture → tree AutoML);
Miscalibration (judged by log-loss → logit recalibration).

## Methods under test

best_single, simple_average, top_k, val_weighted, stacking_lr, **stacking_logit**,
nsga3 *(extra: nsga)*, tpot *(extra: tpot)*, okapi *(extra: okapi)*, plus the
analytic **oracle** ceiling.

## Usage

```bash
uv sync                       # core deps; add --extra nsga/tpot/okapi as needed
uv run fusionbench list
uv run fusionbench run --n 50000 --out results
uv run fusionbench generate --n 50000 --out data   # write datasets to disk
```

Heavy methods (nsga3 / tpot / okapi) are optional and skipped when their
dependencies are absent; the full comparison is intended to run on the GPU host.

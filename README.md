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

best_single, simple_average, top_k, val_weighted, **op_min/op_max/op_far/op_close**
(fixed operator atoms), stacking_lr, **stacking_logit**, **okapi** *(extra: okapi —
runs on the numpy/CPU backend)*, nsga3 *(extra: nsga)*, tpot *(extra: tpot)*, plus
the analytic **oracle** ceiling.

## Results

A regularized logistic stacker is a very strong baseline on *calibrated,
stationary* probabilities, so OKAPI **ties** it on the monotone structured
pathologies (committee, correlated, deadweight, needle, hardveto, hardrescue,
weakest_link, compound — regret within ~0.01) while being far more parsimonious,
and it **dominates** the naive consensus baselines (avg / top-k / best-single)
everywhere.

OKAPI's *clear* wins are on **genuinely non-linearly-separable** structures that
linear methods provably cannot represent: `gated_experts` (mixture routing, OKAPI
0.786 vs stacking 0.73) and `parity` (XOR, 0.692 vs 0.49) — robust across seeds.
Its honest losses are `miscalibration` (needs a logistic recalibration link) and
`two_pathways` (a depth-3 DNF its GP search still under-realises). Making OKAPI
competitive required adapter-level search fixes (coherent single-channel binary,
population seeding, multi-restart). See `DESIGN.md` and `docs/sample_run/`.

## Usage

```bash
uv sync                       # core deps; add --extra nsga/tpot/okapi as needed
uv run fusionbench list
uv run fusionbench run --n 20000 --out results
uv run fusionbench sweep --kind reliability --scenario needle --out results/sweep
uv run fusionbench generate --n 50000 --out data   # write datasets to disk
```

OKAPI runs on the numpy/CPU backend (no GPU needed); these datasets are small
enough to evaluate every method locally. Optional methods (okapi / nsga3 / tpot)
are skipped when their dependencies are absent.

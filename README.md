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

For binary datasets with 1–12 providers, the opt-in method `lattice_brier_exact` computes the least empirical-Brier minimizer over unrestricted MIN/MAX trees. Install the `lattice` extra for the exact integer-capacity minimum cut. This comparator measures how far a search finished above the best attainable fit loss. Evaluate its generalization on separate rows as for any other fitted method.

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

## Evaluate predictions from your own models

Export probabilities on two disjoint sets. Each base model should have been trained on other examples, or its fit-set probabilities should be out-of-fold predictions. Keep all arrays in identical sample order within each split and identical class order across models. The CLI fits fusion methods on the fit split and scores them on the test split.

```text
my_data/
  meta.json
  gt/fit.npy       gt/test.npy
  fit/model1.npy   fit/model2.npy
  test/model1.npy  test/model2.npy
```

Each prediction file has shape `[N, C]`, finite probabilities and rows summing to one. Each ground truth file contains `N` integer labels from 0 to `C-1`. Example metadata follows.

```json
{"name":"my_data","task":"binary","n_classes":2,"model_ids":["model1","model2"],"knobs":{"primary_metric":"log_loss"}}
```

```sh
fusionbench evaluate --dataset my_data --methods simple_average stacking_lr --out results/my_data
```

The report ranks methods on the held-out primary metric. It includes regret when `oracle/test.npy` contains an analytic posterior for the test set. The default primary metric is macro-F1. For a reproducible multiclass example using the bundled scikit-learn digits data, run the following from the repository root.

```sh
python examples/digits_predictions.py --out data
fusionbench evaluate --dataset data/digits --methods best_single simple_average stacking_lr okapi --out results/digits
```

The example needs the OKAPI extra for its last method. The 898 base-training, 449 fusion-fit and 450 test images come from a single stratified split.

## Research branch

Crossed-seed experiments measure when a reliable validation ranking breaks under a controlled distribution shift. Single-model experiments test which ensemble mechanisms can be recovered from raw inputs and labels. [Results, controls, data and reproduction commands](docs/ensemble-study/README.md)

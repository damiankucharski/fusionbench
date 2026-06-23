# Manufactured Fusion Benchmark — Design

Purpose: the manufactured-dataset benchmark requested in the IEEE TEVC R2 revision.
Goal: **remove the validation-test reliability confound by construction**, so the benchmark
isolates "can OKAPI recover the correct recombination when the validation signal is
trustworthy?" — the clean complement to the STARCOP/MedMNIST fragility sweeps and the
PMLB regime map.

Features are calibrated probability vectors `[N, C]` only (no logits, no hard decisions),
matching OKAPI's `ValueNode.Value`. OKAPI operators available: Mean, Max, Min,
WeightedMean, CloseThreshold(τ), FarThreshold(τ).

## Core principle: reliability by construction

1. **One i.i.d. generative process, large N.** 100,000 rows from a fixed stochastic
   process, stratified split into **Train = 50k** (OKAPI fitness/selection set) and
   **Test = 50k**. No distribution shift — identical parameters across splits.
2. **Tight estimates.** ~50k selection points ⇒ SE(acc/F1) ≈ √(0.25/50000) ≈ 0.0022.
   Engineer every optimal-vs-suboptimal gap to exceed ~0.005 so the validation *ranking*
   of fusion rules provably equals the test ranking. Nothing to overfit.
3. **Known oracle.** Each dataset ships (a) the Bayes/oracle fusion rule, (b) the explicit
   OKAPI tree achieving it, (c) the predicted baseline ranking. Report **regret =
   oracle − method**, not a bare horse-race.

## Generative skeleton

Per row: draw latent label Y. Each base model m emits a calibrated proba
`p_m = σ(α_m · s(Y) + ε_m)` where s(Y) is true signal, α_m competence, ε_m model noise.
The engineered structure lives in error masks, competence regions, and calibration —
that is the "fault" each dataset injects against naive aggregation.

## Datasets

| # | Task | Engineered structure | Oracle OKAPI tree | Baselines that fail | Proves |
|---|------|----------------------|-------------------|---------------------|--------|
| D1 Committee | Binary | 10 calibrated models, equal ~85% acc, independent error masks | Mean(all) | none — all tie | OKAPI must not lose trivial case; abstains to mean-like tree |
| D2 Dead weight | Multiclass(5) | 3 strong + 4 mediocre + 3 ~random | WeightedMean(strong)/pruned subset | Simple Average | WeightedMean + implicit selection/pruning |
| D3 Confidence routing | Binary | Specialists: confident⇒correct, unsure⇒~0.5 | Far(τ=0.5) | Mean, Top-K | FarThreshold per-instance routing |
| D4 FP veto (STARCOP) | Binary, imbalanced | Trigger-happy detector + precise negative-veto | Min(detector, veto) | Mean, Top-K, Stacking-LR | Min as logical veto/AND gate |
| D5 Detection rescue | Binary, imbalanced | Each model detects a different positive subset | Max(all) | Mean | Max as OR/union gate |
| D6 Compound ★ | Binary | Consensus core + FP-veto + missed-detection rescue | Max(Min(Mean(A,B,C), veto), rescue) — depth≥2 | everything except OKAPI (maybe TPOT) | Stateful sequential alternating-tree composition |
| D7 Per-label experts | Multilabel(6) | Each model expert on a different label, ~0.5 elsewhere | Far(τ=0.5) per-cell | Mean, Top-K | Element-wise threshold routing across label columns |
| D0 Saturated control | Multiclass | All models near-identical ~98%, redundant | 1-node tree (abstain) | over-fusers add gap | Parsimony / constructive growth declines to fuse |

★ D6 is the centerpiece: only a compositional method solves it; instantiates the STARCOP
"trust A unless B vetoes" narrative with a guaranteed-reliable signal, so the win is
attributable to expressiveness, not luck.

## Method comparison set (per dataset)

best-single-by-val · Simple Average · Top-K average · Stacking-LR · NSGA-III · OKAPI ·
TPOT · Oracle (upper bound). Report test metric and regret-to-oracle.

Claim: when verification reliability holds, OKAPI recovers the oracle recombination on
D2–D7 where consensus methods structurally cannot, ties on D1, abstains on D0.

## Why each fault defeats the right baseline

- Linear methods (Avg, Stacking-LR) cannot express Min/Max/routing ⇒ D3–D7 nonlinear.
- Mean dilutes minorities ⇒ D5 (lone correct high-proba), D3 (confident specialist).
- Top-K can't gate per-instance ⇒ loses D3/D4/D7.
- TPOT can learn meta-representations ⇒ where it ties OKAPI, that is the honest
  learned-meta vs symbolic-fusion taxonomy point; OKAPI wins on parsimony/auditability.

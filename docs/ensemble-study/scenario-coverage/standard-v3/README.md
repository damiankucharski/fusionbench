# Standard-v3 regression evaluation

The four new mechanisms enter the same suite as the original 20 scenarios. Each scenario contributes equally. Use four paired data seeds (1729–1732), 512 development and 512 held-out rows per seed, the existing regularized native OKAPI procedure with search seed 0, simple averaging, ranked-prefix Top-N and production probability stacking. Top-N uses the scenario's primary development metric in the ordinary runner. The common endpoint here is mean row-wise full-class test Brier. These runs measure the existing fusion methods and serve as a regression reference for future algorithm changes.

From the FusionBench checkout with its OKAPI extra installed:

```sh
export PYTHONPATH=. BACKEND=numpy OKAPI_LOG_LEVEL=ERROR
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python docs/ensemble-study/scenario-coverage/standard-v3/evaluate.py
```

The [per-bank scores](scores.csv) include fitting times and the [per-scenario means](means.csv) give equal weight to each draw. A single shared search seed holds that source of variation fixed. Sample sizes and budgets here are suitable for finding regressions; small differences warrant more independent draws.

Across 24 scenarios and four draws, macro-average Brier was 0.21656 for OKAPI, 0.37811 for simple averaging, 0.36482 for Top-N and 0.24662 for probability LR. OKAPI had lower mean Brier than LR in 17 scenarios. On the rare-specialist case Top-N scored 0.37939 versus 0.39470 for OKAPI. A separate six-bank evaluation breaks out deployment subgroups and tests mechanism-specific controls. [Slice analysis](../v3-slices/README.md)

The severity smoke check uses one data seed, 512 fit and 1,024 test rows, Brier-ranked Top-N, and independent RNG streams per split. Its reference cells use a different protocol from the standard evaluation above. Reproduce all settings with:

```sh
fusionbench profiles --out results/v3-stress --methods okapi_native_pattern_l2 --seeds 23309 --fit-rows 512 --test-rows 1024
```

All 72 profile cells completed with all four methods (288 scores). Across the 48 nonreference settings, OKAPI scored 0.22671 and LR 0.23557, giving equal weight to each scenario. This single-seed check exercises the controls; it cannot establish a population ranking. In the rarer-class variant of weakest link absolute Brier falls even for a constant low-prevalence predictor. The held-out class counts and oracle in the cell file help separate that mechanical change from an improvement in conditional modeling. [Per-cell scores](stress/cells.csv) · [Subgroup scores](stress/groups.csv) · [Macro means](stress/macro.csv) · [Configuration](stress/manifest.json)

Running the ordinary CLI at seed 1729 with 512 rows per split reproduced all 96 method/scenario Brier scores for that seed to within 10⁻¹².

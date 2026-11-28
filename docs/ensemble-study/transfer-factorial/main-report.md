# Directional transfer on 16 observable report patterns

Training with a temporary shortcut produced the largest penalty of the three individual source changes. On a clean target where the shortcut was independent of the outcome, full-class Brier for regularized native-pattern OKAPI rose from **0.25193 to 0.46570**, a paired increase of **0.21377** (95% interval **0.18146 to 0.25509**). Combining rare source examples, an active source shortcut and noise introduced at deployment raised Brier on the noisy target from **0.40980 to 0.69863**. Both comparisons hold the target population fixed. [Preregistered design](protocol.md) · [Paired estimates](results/main-contrasts.csv)

Each independent bank supplied 512 source observations for each training law. The source region frequency was 5% or 50%, the shortcut was active or inactive, and generalist and specialist reports were independently flipped with probability 0 or 20%. We evaluated each frozen prediction rule on two equally mixed target populations with an inactive shortcut, one clean and one with 20% report flips. Label noise was zero throughout. The arithmetic is exact over the 16 observable report patterns. Uncertainty intervals describe variation across 24 paired training banks at fixed search seed zero.

## Penalty relative to training matched to the same target

The penalty is the target Brier of the mismatched fit minus the target Brier of a fit drawn from the target law. An increase in absolute Brier equals the increase in excess Brier because the comparison shares one target Bayes floor. The last column applies the *exact source posterior* directly to the target, so its excess measures how much the population forecasting rule itself changed.

| Source-to-target change | Target | Regularized target Brier | Paired penalty (95% interval) | Source Bayes excess on target |
| --- | --- | ---: | ---: | ---: |
| A. Rare region from 5% to 50% | Clean | 0.35686 | +0.10493 (0.08377, 0.12717) | 0 |
| B. Shortcut disappears | Clean | 0.46570 | +0.21377 (0.18146, 0.25509) | 0.18306 |
| C. Provider noise in training only | Clean | 0.29602 | +0.04409 (0.02359, 0.06308) | 0.03403 |
| D. Provider noise at deployment only | Noisy | 0.43523 | +0.02543 (0.01591, 0.03439) | 0.03891 |
| E. Rare source, shortcut disappears, noise introduced | Noisy | 0.69863 | +0.28883 (0.26459, 0.31250) | 0.17006 |

The routing cue reveals the latent region exactly. Changing only the region's frequency leaves the true probability of either outcome at each observed pattern unchanged. All 16 patterns have positive source mass. For A, the sample contained 26.8 region-specific examples on average, against 256.6 under balanced training. Target Brier conditional on that region rose by 0.22219 relative to balanced training, while common-region Brier fell by 0.01233. The balanced target weights these changes equally. The native selector chose an augmented tree in 5/24 rare-source fits and 22/24 balanced-source fits. These frequencies describe the fitted procedures. [Conditional risks and counts](results/main-scores.csv) · [Selected families](results/main-family-counts.csv) · [Population maps](results/main-law-maps.csv)

## A signed generalization gap can hide impaired transfer

Write the target-minus-fit gap as **(source population minus empirical fit) + (target population minus source population)**, holding the fitted prediction rule fixed. Under A, the two terms were +0.00674 and −0.02368, giving a signed gap of **−0.01694**. Upweighting the region where this particular rule had lower absolute loss improved its population Brier, even though training on a balanced source would have helped much more. Under E, the two terms were +0.00507 and +0.53215, giving **+0.53721**. [All 16 cells](results/main-cells.csv)

A second comparison changes the target law while keeping both the training data and the predictor fixed. For matched, clean, balanced training without a shortcut, deployment noise raised absolute Brier by **0.18330** (0.17227, 0.19174). The exact Bayes floor rose by 0.16083, from 0.23217 to 0.39300, leaving an excess increase of **0.02246** (0.01144, 0.03090). For clean training with both a rare source and an active shortcut, the same target perturbation raised absolute Brier by 0.05147 while excess *fell* by 0.10936 as the floor rose. [Frozen clean-to-noisy contrasts](results/main-frozen-target-noise.csv)

## Reference methods and interactions

| Method | A. Rare source | B. Shortcut | C. Training-only noise | D. Target-only noise | E. All three |
| --- | ---: | ---: | ---: | ---: | ---: |
| Regularized native-pattern OKAPI | +0.10493 | +0.21377 | +0.04409 | +0.02543 | +0.28883 |
| Brier-ranked Top-N | +0.03594 | +0.45365 | 0.00000 | 0.00000 | +0.38311 |
| Probability logistic stacking | +0.05126 | +0.23678 | +0.01906 | +0.01750 | +0.28019 |
| Arithmetic mean | 0 | 0 | 0 | 0 | 0 |

The arithmetic mean has the same predictions across training conditions. Its absolute Brier was 0.42820 on the clean target and 0.47345 on the noisy target. It trailed regularized OKAPI under matched training and led all four methods under the joint mismatch. Across the full factorial, the disappearing shortcut's target-Brier penalty for regularized OKAPI grew by 0.10288 (0.08006, 0.12488) with noisy training reports, averaged equally over region prevalence and target noise. The four-way difference was +0.00479 (−0.01361, +0.02398). All 15 main effects and interactions are available as signed differences of differences, along with fit-target gap components and subgroup risks. [Factorial estimates](results/main-factorial-effects.csv) · [Bank-level contrasts](results/main-bank-contrasts.csv)

## Reproduction

The single-threaded run fit 768 models and evaluated 1,536 target conditions. The 95% intervals come from 10,000 resamples of the 24 entire banks with seed 97941, and apply marginally to each contrast. Pattern-level training counts, observed positive counts and frozen predictions reproduce all empirical fit scores. The model-free source and target maps give each report pattern's mass, posterior and region-conditional contributions. Replaying the archived predictions against these maps recovered all main Brier scores, subgroup scores and gap identities within 4.5 × 10⁻¹⁶. [Runner](study.py) · [Predictions and counts](results/main-support-predictions.csv.gz) · [Fit bank identities](results/main-fit-banks.csv) · [Run manifest](results/main-run.json)

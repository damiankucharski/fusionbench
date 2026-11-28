# More rare examples or more weight on the rare region?

Increasing the source sample tenfold at 5% specialist-region prevalence gave no clear improvement on the balanced target. Regularized OKAPI’s full-class Brier rose from 0.35613 to 0.36072, a paired change of +0.00459 (95% bootstrap interval −0.00980 to +0.01826). The mean number of specialist-region examples rose from 26.4 to 260.4. There was a small improvement in the common region (−0.00464, −0.00803 to −0.00180). The specialist-region change was uncertain (+0.01382, −0.01368 to +0.03970).

## Target risk

All models were evaluated on the exact population with 50% specialist-region prevalence. Lower full-class Brier is better. The Bayes Brier on this target is 0.23217.

| Source prevalence | Source rows | Mean rare rows | Arithmetic mean | Top-N | Logistic stacking | Regularized OKAPI | OKAPI common | OKAPI rare |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5% | 512 | 26.4 | 0.42820 | 0.37500 | 0.38059 | 0.35613 | 0.38041 | 0.33185 |
| 5% | 5,120 | 260.4 | 0.42820 | 0.37500 | 0.36427 | 0.36072 | 0.37577 | 0.34567 |
| 50% | 512 | 256.1 | 0.42820 | 0.33906 | 0.31861 | 0.24698 | 0.39212 | 0.10184 |
| 50% | 5,120 | 2,566.3 | 0.42820 | 0.33906 | 0.31528 | 0.23689 | 0.38092 | 0.09286 |

The 5% source with 5,120 examples and the balanced source with 512 each supply 256 specialist examples in expectation. The former increased target Brier by **0.11374** (0.10360 to 0.12280). Rare-region Brier increased by 0.24383 (0.22554 to 0.26018), and common-region Brier decreased by 0.01635 (−0.02183 to −0.01095). Thus the count of rare training rows alone does not explain the gap. These designs also differ in objective weights and the amount of common-region evidence. The comparison measures their combined effect.

## Where the difference appeared

On 5%/5,120 the augmented and flat discovery heads made identical predictions within 10⁻¹² on all 16 possible inputs in **16/16 banks**. On 50%/512 the two heads differed in **16/16 banks**, and source validation selected the augmented family every time. The pattern persisted after refitting both heads on all source observations. With 5%/5,120, source validation picked flat on eight banks and GP on eight. With 5%/512 it picked augmented twice, flat twice and GP twelve times.

| Source | Flat after refit | Augmented after refit | GP | Gain available from hindsight choice among the three |
|---|---:|---:|---:|---:|
| 5% / 5,120 | 0.36416 | 0.36416 | 0.36462 | 0.00218 (0.00053 to 0.00416) |
| 50% / 512 | 0.31885 | 0.24698 | 0.33321 | 0.00000 |

The first three columns are average target Brier. The last column compares the deployed choice to the best family on each bank using exact target risk. Even with that hindsight advantage in both cells, the difference was 0.11156 (0.10217 to 0.11981). For the source-selected family, refit changed target Brier by −0.00048 (−0.00117 to +0.00017) on 5%/5,120 and −0.00155 (−0.00296 to −0.00027) on 50%/512. Most of the gap was already present in the candidate predictions available to the chooser. This audit cannot distinguish failure to discover a GP basis, failure to retain it during screening, or failure to assign it a useful coefficient. The internal GP archive itself was not scored on the target.

## Design and replay

Sixteen fresh banks (252100–252115) shared random draws across source prevalences. Within each prevalence, 512 was a prefix of 5,120. The cue identifies the region, the shortcut is an independent fair vote, and both provider and label noise are zero. All four methods see identical source rows. Regularized OKAPI uses the production defaults with GP seed and split seed zero, 70% discovery and 30% validation. Validation is used both to choose the GP finalist and to choose among flat, augmented and GP. Heads are then refit on the full source bank; the GP finalist is unchanged. Both conditions averaged about 180 rare discovery rows and 76–79 rare validation rows in the equal-expected-count comparison.

The scorer integrated squared loss against the exact joint distribution of the observed label and all 16 input states. Fitting and selection receive source data only. Paired percentile intervals resample all 16 banks with replacement 10,000 times at seed 97942. They describe sampling uncertainty for this generator and fixed optimizer seed. There were 64 regularized fits, 600 GP discovery evaluations per fit and 210,867 total reported native objective calls including validation, screening and refitting. Measured regularized fit time was 44.3 seconds on one CPU thread across all fits.

Run `rare_evidence.py` with the FusionBench checkout and native OKAPI on `PYTHONPATH`, and single-threaded BLAS. The `results/rare-*.csv*` files include fitted choices, observed rare counts, validation scores, every state prediction, exact risks, paired contrasts, and both discovery and refit family comparisons. A separate check reconstructed all 1,920 full Brier and excess-risk pairs from 640 saved probability vectors using direct joint-label enumeration. [Protocol](protocol.md) · [Runner](rare_evidence.py) · [Paired contrasts](results/rare-contrasts.csv) · [Candidate equivalence](results/rare-family-equivalence.csv) · [Family diagnostics](results/rare-family-diagnostics.csv)

# Does the new breeding score reduce sensitivity to the search seed?

We ran three paired search seeds for each of 24 independently generated banks in each law. The inputs, empirical anchor, labels, four folds and test cases stayed fixed within bank. Each seed was used once on each of the pooled, fold and joint breeding paths. All comparisons below keep the same joint final choice. A range is the highest minus the lowest fresh-scene excess loss among the three runs; the worst is their maximum. We summarize these values across banks. No audit score chose the seed used for deployment.

| Law and breeding rule | Mean excess Brier | Mean three-seed range | Mean worst of three |
| --- | ---: | ---: | ---: |
| Perfect provider, fold | 0.00371 | 0.00713 | 0.00783 |
| Perfect provider, joint | 0.00385 | 0.00807 | 0.00850 |
| True two-provider minimum, fold | 0.07026 | 0.06599 | 0.10018 |
| True two-provider minimum, joint | 0.06275 | 0.06218 | 0.09267 |

On the MIN law joint breeding reduced mean excess relative to fold breeding by **0.00751**, with a paired 95% interval for the reduction of **0.00103–0.01593**. It visited an exact function on the fit and audit samples in 19/72 runs versus 13/72 and deployed it in 14 versus eight. However, ordinary pooled breeding and pooled deployment had mean excess **0.05321**. The two research heuristics both lagged this unmodified policy on these banks. Logistic stacking reached 0.03709, regularized native-pattern OKAPI 0.03546, Top-N 0.08494 and the mean 0.13499.

On the perfect-provider law the change in mean excess was +0.00014 (−0.00151 to +0.00199). The changes in the three-seed **range** were +0.00094 (−0.00191 to +0.00424) with a perfect provider and −0.00381 (−0.01914 to +0.00815) with MIN. Both range intervals include changes in either direction. Three seeds expose realized differences on the same data but provide limited information about the tails of the search distribution. The four references were fitted once per bank with their prescribed seed and repeated in the raw table only for paired comparisons. Their zero observed range follows from that design.

The 432 native searches took 77.76 s on two single-threaded CPU workers. [Bank-level means, ranges and worst cases](results/r3-controlled-banks.csv) · [Contrasts](results/r3-controlled-contrasts.csv) · [Input fingerprints](results/r3-controlled-run.json)

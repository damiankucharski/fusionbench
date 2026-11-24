# Held-out results on three real provider banks

The best of the five primary methods by mean test Brier changed across tasks. Probability stacking led on digits, logit stacking narrowly led the arithmetic mean on Iris (0.07581 versus 0.07613), and the arithmetic mean led on Seeds. Native OKAPI beat probability stacking on Iris by 0.01848 and lost to it on digits by 0.00667 and Seeds by 0.00525. The corresponding counts of splits favoring native OKAPI were six of eight on Iris, one of eight on digits and two of eight on Seeds. The external forecasts were frozen before scoring any target. [Forecast freeze](https://github.com/damiankucharski/fusionbench/commit/d4d84f3) · [Per-split scores, selections and accuracies](scores.csv)

| Method | Digits | Iris | Seeds |
|---|---:|---:|---:|
| Arithmetic mean | 0.072130 | 0.076134 | **0.122249** |
| Learned Top-N | 0.046463 | 0.088840 | 0.126598 |
| Probability logistic stacking | **0.036009** | 0.099568 | 0.127616 |
| Logit logistic stacking | 0.065220 | **0.075809** | 0.133831 |
| Native OKAPI, penalty 0.25 | 0.042675 | 0.081089 | 0.132861 |
| Fit-selected temperature | 0.063844 | 0.082861 | 0.122934 |
| Fixed T=0.5 | 0.065629 | 0.075411 | 0.125708 |
| Class-symmetric probability logistic stacking | 0.032842 | 0.085933 | 0.126889 |
| Unrestricted probability logistic stacking, pooled scale | 0.036157 | 0.098456 | 0.127261 |

Scores are means across eight test partitions of full-class Brier, the summed squared probability error per example. Bold marks the lowest mean among the five primary methods. The symmetric and unrestricted logistic diagnostics use identical per-provider pooled scales. Constraining coefficients across classes lowered their mean Brier on each task. Among all nine methods, symmetric stacking led on digits, the fixed sharpened mean led on Iris, and the unadjusted mean led on Seeds. [Mean accuracy, Brier and timing](means.csv)

Digits uses a fresh provider pool on a previously examined image collection. Gaussian naive Bayes, distance-weighted 11-nearest neighbors and 100 extremely randomized trees see only the 898 base-training images in each partition. Iris and Seeds each have three classes and 44 fusion-fit observations. Base-standardized logistic regression, uniform five-nearest neighbors and a 100-tree random forest supply their prediction banks. Each held-out split contains 450, 32 and 62 observations on digits, Iris and Seeds, respectively. Across eight repetitions these amount to 1,601, 129 and 198 distinct test observations. Splits overlap, and these comparisons are descriptive of the selected datasets and provider families. [Provider settings, split roles and Seeds attribution](../external-fit/README.md) · [Timings and within-task overlap](run.json)

A separate reconstruction trained all 72 source classifiers and matched the archived fit probabilities bit for bit. A second scoring implementation then refit all nine fusion rules on the saved banks and reproduced 216 test Brier scores within 5.6 × 10⁻¹⁷ and all 216 accuracies exactly. It checked the 27 method means, test row IDs and labels, and the native, Top-N and temperature selections. All four fit artifacts matched the forecast freeze byte for byte.

From the repository root, first reproduce the fit artifacts using the fit-stage command, and run the following after fixing any forecasts. It rebuilds the base learners, checks all frozen fit artifacts, and writes held-out probabilities, labels, row IDs and scores. The scoring run took 3.64 seconds to build and profile the 24 banks and 19.80 seconds to fit, score and archive all methods, with one numerical thread on an Intel Xeon Platinum 8573C. [Driver](../external.py) · [Test bank](test_banks.npz) · [Frozen fit bank](../external-fit/fit_banks.npz)

```sh
export BACKEND=numpy OKAPI_LOG_LEVEL=ERROR LOGURU_LEVEL=ERROR
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1
python docs/ensemble-study/runner-up-transfer/external.py --score --out results/runner-up-transfer
```

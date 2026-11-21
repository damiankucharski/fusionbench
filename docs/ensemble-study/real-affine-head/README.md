# Real out-of-fold predictions with one nonlinear input to a signed OKAPI head

On the locked test split, the primary signed logit head with an added maximum reached binary Brier **0.023091**, compared with **0.023133** for its flat logistic stack and **0.023191** for its equal-size duplicate-input control. Neither improvement met the prespecified threshold for evidence. The largest absolute difference between a native OKAPI prediction and the corresponding sklearn prediction was **8.36 × 10⁻¹³** over both transforms, six head variants, two splits and three banks. The native head reproduced the fitted mapping on these float64 probabilities.

## Primary and control results

Lower binary Brier is better. It measures the mean squared error of the positive-class probability. The OOF-fit scores below evaluate a fusion on the same OOF rows used to fit it. The outer test comprised 228 separate cases.

| Model, primary OOF bank | OOF-fit | Outer test |
|---|---:|---:|
| Arithmetic mean | .023080 | .028258 |
| Learned Top-N (N = 1) | .017981 | .027005 |
| Probability stacking / native flat head | .014615 | .021657 |
| Native probability head, duplicate | .014528 | .021497 |
| Native probability head, maximum | .014611 | .021628 |
| Logit stacking / native flat head | .013839 | .023133 |
| Native logit head, duplicate | .013555 | .023191 |
| Native logit head, maximum | .013873 | .023091 |
| Separately budgeted ordinary OKAPI GP | .013329 | .022151 |

The primary risk reduction versus flat logit was .00004221, with a corrected one-sided lower bound of −.00003925. The reduction versus the duplicate-input control was .00010016, with a lower bound of −.00001330. The preregistered claim required both bounds to exceed zero. The compatible effects include small degradations; this experiment did not test equivalence.

## How the extra input was chosen

Six ordinary classifiers supplied probabilities: logistic regression, k-nearest neighbors, Gaussian naive Bayes, random forest, gradient boosting and linear discriminant analysis. Each of the 341 development cases received predictions from classifiers trained without that case. Before test predictions were computed, the pair with the largest mean absolute OOF disagreement was selected, and every head and GP tree was frozen. All three OOF fold assignments picked forest and LDA. The new feature was the pointwise maximum of their positive probabilities. The width-matched control used the maximum of forest with itself. The duplicated feature can change effective L2 regularization even though it adds no information.

Both augmented variants had ten native nodes, seven folded input slots and fourteen fitted coefficients before folding the class pairs, plus an intercept. Each required one fit of the same standardized, regularized two-channel head. The flat model had seven nodes, six slots, and twelve fitted coefficients plus intercept. The fit solved logistic cross-entropy, and the head was then transferred into a real OKAPI tree with a maximum subtree. This procedure tests a specified outer-node optimizer and an OOF-guided choice of inputs. It does not establish that GP mutation discovers those coefficients.

On the two sensitivity banks, outer-test Brier for flat logit was .022782 and .023897; adding the maximum yielded .022477 and .023827. In the probability transform, the maximum slightly worsened risk on each sensitivity bank. All banks used **the same 228 test cases**. They assess sensitivity to OOF partitioning, not three independent replications. Wisconsin had previously been used elsewhere in this research program. This prospectively frozen partition is a replication on familiar data without an independent clinical site or group labels.

## Reproduction and cost

Use FusionBench with OKAPI research revision `d464dae3ad85041b426ea6155fcd91a5ebbf0b8a`. Fold-wise base preprocessing used only training rows; base predictions for the outer holdout used six final fits on all development rows. The primary bank required thirty OOF base fits (1.93 seconds), one bounded ordinary GP search (0.41 seconds), and six outer-head fits/transfers (0.0035–0.0044 seconds each). The primary logit flat and max took 11 optimizer iterations each; the duplicate took 10. All three banks and the six shared final fits together took 6.82 seconds wall time on CPU. Per-model times and optimizer counts are available below.

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OKAPI_LOG_LEVEL=ERROR \
python examples/real_affine_head.py --out results/real-affine-head
```

[Protocol frozen at fec8d77](https://github.com/damiankucharski/fusionbench/blob/fec8d77/docs/ensemble-study/real-affine-head/protocol.md) · [Scores and native transfer discrepancies](results/scores.csv) · [Paired decision criteria](results/contrasts.csv) · [Fit costs](results/costs.csv) · [Configuration and split fingerprint](results/config.json) · [Wisconsin dataset](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)

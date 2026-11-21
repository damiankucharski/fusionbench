# Does a maximum improve a raw-probability stack on radio pulsar candidates?

On 5,966 held-out HTRU2 candidates, duplicating one expert produced the same predictions as reducing its effective L2 penalty. The largest difference in predicted probability was 3.33 × 10⁻¹⁶. The point estimate for the nonlinear component of a maximum was about 0.00024 in positive-class Brier, compared with 0.000003 for the duplicate. Both preregistered nonlinear gains had confidence bounds extending below zero. This survey therefore leaves the predictive benefit of the nonlinear component unresolved.

We trained six heterogeneous classifiers using five-fold out-of-fold (OOF) predictions on 11,932 development candidates, including 1,093 positives. The frozen pair was logistic regression and Gaussian naive Bayes. Only 0.64% of the standardized maximum's variance remained after affine projection on development predictions, and 0.69% on held-out predictions. The test set contained 546 pulsars. These are candidate-level splits within one survey. The public file does not identify the underlying astrophysical sources or survey pointings, so dependence among observations could make candidate-level intervals optimistic. All comparisons share one training bank and one test set.

## Preregistered contrasts

A positive gain is a reduction in mean squared error of the positive-class probability. The percentile range uses 10,000 paired test-candidate resamples conditional on the fitted bank. For each co-primary comparison, the lower endpoint is its Bonferroni-adjusted one-sided lower bound. The decision required both bounds to exceed zero. The duplicate comparison is descriptive.

| Control | Alternative | Brier gain | Paired 2.5%, 97.5% quantiles |
|---|---|---:|---:|
| Linear projection of maximum | Full maximum | +0.000239 | −0.000140, +0.000617 |
| Flat head, zero padded | Residual of maximum | +0.000237 | −0.000169, +0.000640 |
| Flat head, zero padded | Duplicate logistic provider | +0.0000029 | −0.0000032, +0.0000094 |

## Benchmarks on the same test candidates

| Model | Positive-class Brier ↓ | Average precision ↑ |
|---|---:|---:|
| Boosting, chosen among singles on OOF Brier | 0.015737 | 0.93563 |
| Arithmetic mean of all six experts | 0.017357 | 0.92422 |
| Learned Top-N, N=2 | 0.015904 | 0.93371 |
| Flat stack, raw probabilities | 0.016503 | 0.92425 |
| Flat stack, logits | **0.015539** | **0.93644** |
| Native raw-probability head with duplicate | 0.016493 | 0.92432 |
| Native raw-probability head with maximum | 0.016261 | 0.92282 |

Both native augmented heads had ten nodes and seven input slots, with fourteen coefficients and an intercept before folding complementary class channels. The flat native head had seven nodes and six slots. Each native variant was fitted independently using OKAPI's fit_stacking_tree. Its test probabilities differed by at most 6.0 × 10⁻¹³ from the corresponding public raw-probability stacker. The flat logit stack had the lowest observed Brier and highest average precision; these comparisons were descriptive. The maximum helped raw-head Brier in point estimate while slightly lowering average precision.

The geometry controls used a tighter solver tolerance than the public and native heads. In the tight fits, flat and duplicate scored 0.016501 and 0.016498. The corresponding public/default native fits gave a larger apparent duplicate gain (0.0000107) than the tight fits (0.0000029). For the maximum, the largest test-probability discrepancy between solver tolerances was 0.00279, with Brier differing by less than 0.000001. The tight-fit projected and residual-only controls scored 0.016501 and 0.016264, respectively.

The run took 14.9 seconds on one CPU thread. Thirty fold-local base fits took 9.75 seconds, and six full-development fits took 2.44 seconds. Each native fit took 0.016–0.021 seconds. Each numerical control used one supervised fit with 28–49 solver iterations. Pair selection and the sole unsupervised projection together took 0.011 seconds. A second execution reproduced scores, confidence bounds and projection/parity diagnostics.
## Why the controls differ

The procedure picked the two providers with the greatest mean absolute probability disagreement on five-fold OOF predictions. The nonlinear candidate was their maximum on each observation. It is a deterministic function of existing predictions. The question was whether its shape helps a regularized affine head predict new labels.

Adding an exact copy of provider A permits the head to distribute any coefficient on A across two identical coordinates. Splitting equally makes the sum of squared coefficients half as large. The same penalty geometry is available without adding a column by multiplying that already standardized coordinate by √2. This identity continues to hold for the two complementary probability channels. A duplicate can therefore change the fitted solution despite adding no new function.

On development rows, we decomposed the standardized maximum into its least-squares projection onto the existing probability features and a residual orthogonal to that affine span. The projection alone changes the penalty geometry along a linear combination. The full maximum additionally contains the residual. A second contrast adds just the residual to a flat head, preserving the original L2 penalty on the original coefficients. The projection and residual carry their original scales in all comparisons. These controls fit the same number of coefficients and see the same OOF labels. The projection is unsupervised. Only the flat, duplicate and complete maximum were fitted directly from native OKAPI probability roots; the signed residual and projected coordinate are numerical diagnostics.

## Reproduce

Use FusionBench from this revision and OKAPI at `15fbac628c77838ce2df47084bac18a258faf985`, with NumPy and scikit-learn versions in the configuration. The archived outputs contain aggregate prediction metrics and split fingerprints; obtain the original observations from UCI.

```sh
curl -fsSLo htru2.zip https://archive.ics.uci.edu/static/public/372/htru2.zip
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OKAPI_LOG_LEVEL=ERROR \
python examples/pulsar_affine.py --archive htru2.zip --out results/pulsar-affine
```

The preregistration referred to 2017 in the citation. The UCI citation lists Lyon, R. (2015), *HTRU2* [Dataset], DOI 10.24432/C5DK6R, under CC BY 4.0. The original source, DOI, archive and CSV hashes were fixed correctly.

[Preregistration](https://github.com/damiankucharski/fusionbench/blob/d4204610d925839bee31e6b0f099af567c372275/docs/ensemble-study/pulsar-affine/protocol.md) · [Dataset and license](https://archive.ics.uci.edu/dataset/372/htru2) · [All scores](results/scores.csv) · [Contrasts](results/contrasts.csv) · [Parity and projection](results/diagnostics.csv) · [Fit costs](results/costs.csv) · [Configuration](results/config.json)

# Removing persistent class identity from two historical prediction banks

On heterogeneous digits, regularized probability stacking's 0.00675 Brier advantage over native OKAPI fell to a near tie after class IDs were randomized afresh on every example. On directed synthetic panel E, the ordering reversed in all eight banks. The probability stacker's loss increased by 0.00726 on digits and 0.06906 on panel E. Native OKAPI's mean loss changed by less than 0.00001 in either cohort. Each row retained its original provider probabilities and their agreement about which class was correct. [Registered protocol](https://github.com/damiankucharski/fusionbench/blob/f84564c/docs/ensemble-study/runner-up-transfer/permutation/protocol.md) · [Historical banks and scores](../../external-coverage/README.md)

## Paired differences in full-class Brier

Negative gaps favor the first method. The change is the permuted gap minus the original gap within the same seed. Intervals are two-sided 95% paired t intervals over eight splits or synthetic banks. The digits intervals describe variation across overlapping partitions of the same images. Panel E uses independent draws from one generator. [Per-seed contrasts](pairs.csv) · [Intervals for both gaps and their changes](contrasts.csv)

| Cohort | First minus second | Original gap | Permuted gap | Paired change [95% interval] |
|---|---|---:|---:|---:|
| Digits | Native OKAPI − probability LR | +0.00675 | −0.00051 | −0.00726 [−0.01301, −0.00152] |
| Panel E | Native OKAPI − probability LR | +0.02810 | −0.04095 | −0.06905 [−0.07415, −0.06394] |
| Digits | Pooled-scale full LR − symmetric LR | −0.00227 | +0.00487 | +0.00714 [+0.00122, +0.01307] |
| Panel E | Pooled-scale full LR − symmetric LR | −0.05344 | +0.01541 | +0.06885 [+0.06380, +0.07390] |

The two pooled-scale stackers share the same input scaling and multinomial logistic objective. The full head can fit different coefficients for each input–output class pair and separate class intercepts. The symmetric head learns one weight per provider. After relabelling, the symmetric head had lower Brier on seven of eight digits splits and all eight synthetic banks. The digits native-minus-LR gap after relabelling spanned zero [−0.00599, +0.00498]. The treatment removes stable class priors, per-class differences in provider quality and systematic confusion together. These banks establish sensitivity to persistent class identity. They cannot determine which of those sources explains the digits effect or predict transfer to a new dataset.

## Losses by method

Means over the eight held-out test splits. The historical digits bank uses logistic regression, five-nearest-neighbors and random forest providers. Both cohorts have ten classes, three providers and 449 fit rows per bank. Each test contains 450 rows on digits and 2,048 on panel E. We refitted the original nine-method evaluator twice per bank using the historical optimization seed. Learning, validation and method choices only used fusion-fit rows. Base estimators used separate rows on digits. [All method scores, choices and fit times](methods.csv) · [Unrounded means](means.csv)

| Fusion method | Digits original | Digits permuted | Panel E original | Panel E permuted |
|---|---:|---:|---:|---:|
| Native OKAPI (L2 = 0.25) | 0.03868 | 0.03868 | 0.03615 | 0.03616 |
| Arithmetic mean | 0.06408 | 0.06408 | 0.10384 | 0.10384 |
| Ranked-prefix Top-N average | 0.03958 | 0.03958 | 0.10384 | 0.10384 |
| Probability stacking | 0.03193 | 0.03919 | 0.00805 | 0.07710 |
| Logit stacking | 0.04084 | 0.04896 | 0.02166 | 0.07254 |
| Pooled-scale full LR | 0.03190 | 0.03904 | 0.00803 | 0.07689 |
| Symmetric LR | 0.03416 | 0.03416 | 0.06147 | 0.06147 |
| Mean with fit-selected temperature | 0.03591 | 0.03591 | 0.07474 | 0.07474 |
| Mean with fixed temperature 0.5 | 0.03545 | 0.03545 | 0.07474 | 0.07474 |

## Checks and reproduction

One permutation per example moves the complete three-provider vector for each class and moves the correct label to its corresponding coordinate. Fit and test permutations use separate streams. Sorting each provider's class values, gathering its score for the true class, and undoing the permutation on the mean recovered the original values exactly. The largest change in a row's mean-pool Brier was 4.45 × 10⁻¹⁶ or less. Across the 16 banks the largest change in aggregate Brier was 1.39 × 10⁻¹⁷ for the public arithmetic mean and 4.71 × 10⁻¹¹ for independently refitted symmetric LR. An independent replay regenerated all 27,168 row permutations, recovered all 81,504 provider vectors through the inverse label mapping, and matched 160 fingerprints. All 255 changes in provider argmax decisions involved tied class scores. All 144 Brier scores on the original banks exactly reproduced the published values. The checker independently reconstructed 96 individual gaps or changes and the 12 interval summaries to within 1.4 × 10⁻¹⁷. [Input and permutation fingerprints](checks.csv) · [Historical replay](archive.csv)

Run from the FusionBench repository root with the study's native OKAPI dependency and CPU environment.

```sh
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python docs/ensemble-study/runner-up-transfer/permutation.py --workers 2
PYTHONPATH=. python docs/ensemble-study/runner-up-transfer/permutation/verify.py
```

# Secondary check on handwritten digits

Adding the frozen native-head penalty (`l2_strength=0.25`) produced almost the same average held-out Brier as the previous OKAPI procedure on digits: **0.027183810** versus **0.027184628**. The challenger won one split, tied four, and lost three. Its improvement on the first split (0.004545) nearly canceled the losses on the other three (0.000226, 0.001763 and 0.002548). These results leave a benefit on this real dataset unresolved.

The challenger selected the ordinary genetic-programming (GP) tree on all eight splits. The previous procedure selected that same GP candidate four times, a flat signed head twice and a head with discovered features twice. The four shared GP selections gave identical test predictions. In the four differing selections the contrast concerns which candidate was deployed: the regularized heads were never the challenger's final predictor. Across all 3,600 test appearances the challenger repaired three previous classification errors and introduced six.

| Method | Mean test Brier ↓ | Top-1 accuracy ↑ |
|---|---:|---:|
| Logistic stack, probability features | **0.022847** | **98.47%** |
| OKAPI, variance-scaled penalty 0.25 | 0.027184 | 98.17% |
| Previous native OKAPI | 0.027185 | 98.25% |
| Learned Top-N | 0.028689 | 98.28% |
| Logistic stack, logit features | 0.030978 | 97.83% |
| Arithmetic mean | 0.054562 | 98.19% |

The probability-feature stacker achieved lower Brier than the challenger on seven of eight splits, by 0.004337 on average. The challenger beat arithmetic averaging on eight splits, Top-N on seven and the logit-feature stacker on six. Top-N selected the nearest-neighbor provider alone on every split.

The eight test partitions contained 1,613 distinct images across 3,600 appearances. A pair of test partitions shared 98–125 of its 450 images (mean 112.89). We report descriptive paired differences on these partitions. Uncertainty across independent datasets, writers or collection sites remains unmeasured.

The run took 39.08 seconds wall time with one numerical thread on an Intel Xeon Platinum 8573C CPU. The eight shared three-provider builds took 4.53 seconds in total. Per split, the two native fits averaged 1.95 seconds for the previous model and 1.86 seconds for the challenger. Their reported objective-call totals across eight fits were 37,607 and 35,005, respectively. Python 3.12.14, NumPy 2.3.5 and scikit-learn 1.8.0 were used.

An independent artifact check reconstructed all eight development, base-training and test partitions and their index hashes, checked every saved reference label, recomputed all 48 test accuracies and Brier averages from 21,600 per-image records, and reproduced every paired win/tie/loss count. The case archive stores the predicted class and full-class squared loss for each method and image. Recomputing complete probability-vector digests additionally requires rerunning the models.
## Data and controls

The sklearn digits sample contains 1,797 images of ten handwritten digits, each on an 8 × 8 grid. The existing FusionBench provider builder scales pixels by 1/16 and makes two stratified splits. In each of eight repetitions, logistic regression (2,000-iteration cap), unweighted 5-nearest neighbors and a 100-tree random forest train on 898 images. Another 449 images supply labeled predictions for fitting all fusion methods. The final 450 images are reserved for evaluation. Generator seeds are 2,810,000–2,810,007, with seed + 1 for the second stratified split and the bank seed for the forest. The three-way split avoids having a base model predict its own training examples for either fusion split. All six fusion methods receive the same float32 stored probabilities, in the same provider order.

The previous and challenger arms use the same native OKAPI search seed and default setup. The single change is a variance-scaled quadratic penalty on native head coefficients at strength 0.25. Both use the same internal 314 discovery / 135 selection rows drawn from the 449 fusion-fit cases. Head comparison uses the unpenalized predictive objective on internal selection rows. Each returns one fusion tree over the three provider outputs. The plain controls are arithmetic averaging, development-ranked Top-N (every prefix evaluated using Brier), and C=1 standardized logistic stacking on either probabilities or clipped log odds. Additional per-provider measurements in providers.csv are descriptive.

The primary outcome in this secondary experiment is the full ten-class Brier score, the mean sum of squared probability errors per image; lower is better. Accuracy is reported separately. Positive gain means the challenger had lower test Brier. The comparison uses a 10⁻¹² tolerance for ties. The eight seeded partitions are repeated measurements on one dataset. They do not supply eight independent populations or new writers, and this clean test does not measure sensor faults or distribution shift. Native fusion still requires all three providers at inference.

## Reproduction

Use FusionBench `862f8609c9d0525b6ce714ccce84b80610464e5b` and OKAPI `5af0db2bfaef981fcef4dd4aff183e69c2e165f7` with the CPU NumPy backend. From the FusionBench checkout, with the matching versions installed:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1 \
BACKEND=numpy OKAPI_LOG_LEVEL=ERROR LOGURU_LEVEL=ERROR \
python docs/ensemble-study/pattern-iterations/digits.py --challenger okapi_native_pattern \
  --challenger-kwargs '{"l2_strength":0.25}' --out results/pattern-digits
```

[Frozen decision and population](../round-3-protocol.md) · [Driver](../digits.py) · [Per-split scores](scores.csv) · [Model means](means.csv) · [Paired comparisons](contrasts.csv) · [Individual provider scores](providers.csv) · [Per-image decisions and losses](cases.csv.gz) · [Split hashes, overlap and environment](manifest.json)

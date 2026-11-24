# Held-out fusion on handwritten digits and Wine

Probability-feature logistic stacking had lower average Brier than native OKAPI on all three prediction banks. The mean paired advantages were 0.00675 on heterogeneous digits, 0.00414 on the logistic-regression digits bank, and 0.00825 on Wine. It beat OKAPI on seven, six and six of the respective eight splits. The fixed sharpened mean (T=0.5) also improved on OKAPI's average Brier on all three banks. Sharpening preserved the arithmetic mean's top-1 accuracy on every split. [Frozen synthetic forecasts](../predictions/forecasts.csv) · [Per-split scores and accuracies](scores.csv)

| Fusion method | Digits, heterogeneous | Digits, logistic bag | Wine |
|---|---:|---:|---:|
| Arithmetic mean | 0.064083 | 0.086204 | 0.044811 |
| Learned Top-N prefix | 0.039578 | 0.086039 | 0.036048 |
| Probability logistic stacking | **0.031930** | 0.068179 | **0.021783** |
| Logit logistic stacking | 0.040836 | **0.066812** | 0.039429 |
| Native OKAPI, penalty 0.25 | 0.038685 | 0.072319 | 0.030037 |
| Mean, fixed T=0.5 | 0.035447 | 0.071809 | 0.026384 |
| Mean, fit-selected T | 0.035912 | 0.072112 | 0.025575 |
| Class-symmetric logistic stacking, pooled scale | 0.034164 | 0.071737 | 0.024625 |
| Unrestricted logistic stacking, pooled scale | 0.031896 | 0.067897 | 0.021627 |

Values are means across eight test partitions. Bold marks the lowest Brier among the five primary methods. Each row's Brier is the sum of squared probability errors across all classes, averaged over examples. [Mean results including accuracy](means.csv)

On the same digit images, changing from heterogeneous providers to three bootstrapped logistic regressions changed the best primary method from probability stacking to logit stacking. Native OKAPI selected a genetic-programming tree on all eight heterogeneous digits splits and all eight Wine splits. On the logistic digits bank it selected a signed head with discovered features six times and a flat head twice. A single fusion tree still consumes predictions from trained providers. An earlier heterogeneous digits experiment also favored probability stacking over native OKAPI on seven of eight splits. [Earlier digits results](../../pattern-iterations/digits/README.md)

The unrestricted and class-symmetric logistic diagnostics share their input scaling and training loss. Allowing class-specific coefficients and intercepts lowered mean Brier by 0.00227, 0.00384 and 0.00300 across the three banks, respectively. The unrestricted model won six of eight splits on each digits bank and three of eight on Wine. With 45 test observations per Wine split, a few large probability errors can change the aggregate. [Definition of both controls](../symmetric.py)

## Bank construction and limitations

Each stratified split separates base-model training, labeled fusion fitting and held-out evaluation. Each digits bank uses 898/449/450 images from ten classes, with seeds 3140000–3140007. Wine uses 89/44/45 observations from three cultivars, with seeds 3150000–3150007. The heterogeneous digits builder divides pixel values by 16 and fits a multinomial logistic regression (C=1, up to 2,000 iterations), uniform five-nearest-neighbors and a 100-tree random forest. Wine uses these same learners after standardizing its thirteen measurements on base-training rows. The three logistic providers in the paired digits bank each resample the same 898-image base set with replacement, preserving its class counts. Switching banks jointly changes the provider family and sampling scheme. [Provider recipe](../../../../examples/digits_predictions.py) · [Bootstrap and split definitions](../external.py)

The base logistic regression learns from image pixels or measurements. The logistic stackers learn from the resulting provider probabilities on fusion-fit rows. Top-N evaluates each equal-weight prefix ranked by fit Brier. Native OKAPI uses a 70/30 discovery/selection split within the same fit-label budget. The temperature selector chooses among 0.25, 0.5, 0.75, 1, 1.5 and 2 using fusion-fit Brier. The two-stage driver freezes fit profiles and checks the same split, bootstrap and fit-input fingerprints on the scoring run. [Fit profiles](fit_profiles.csv) · [Per-provider profiles](fit_providers.csv) · [Fingerprints](fit_manifest.json) · [Fit tensors and row IDs](fit_banks.npz) · [Test tensors and row IDs](test_banks.npz)

These are three specified provider banks on two previously used datasets. Within a dataset the eight test partitions overlap. Across 3,600 digit test appearances there were 1,620 distinct images, and across 360 Wine appearances there were 159 distinct observations. The two digits banks share all their test images. The results describe these tasks and provider pools under clean, stratified splits. Site shift and performance on new tasks remain open. [Test overlap and timings](run.json)

An independent replay checked every original test row ID and label, refit all nine methods on the archived inputs, and reproduced all 216 test Brier scores to a maximum absolute difference of 2.8 × 10⁻¹⁷. Every top-1 accuracy, native selection, prefix size and temperature matched. The four archived fit artifacts are byte-identical to the forecast freeze.

## Reproduction

The outcome run used the code at the forecast freeze and the matching OKAPI checkout. Python 3.12.14, scikit-learn 1.8.0 and NumPy 2.3.5 ran with one numerical thread on an Intel Xeon Platinum 8573C. Rebuilding all 24 shared provider banks took 10.20 seconds and fitting and scoring the nine fusion rules took 29.41 seconds. [FusionBench freeze](https://github.com/damiankucharski/fusionbench/commit/e11f462) · [OKAPI revision](https://github.com/damiankucharski/OKAPI/commit/82ddc6f0f71af9b491052a8772a603083316e6e6)

```sh
export BACKEND=numpy OKAPI_LOG_LEVEL=ERROR LOGURU_LEVEL=ERROR
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1
python docs/ensemble-study/external-coverage/external.py --profiles-only --out results/external-coverage
python docs/ensemble-study/external-coverage/external.py --out results/external-coverage
```

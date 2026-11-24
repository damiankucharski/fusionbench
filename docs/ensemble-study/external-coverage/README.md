# Does the choice of synthetic provider bank transfer to real data?

The new generator narrowed the error of synthetic-to-real forecasts on all three tested provider banks. It predicted which of OKAPI or probability stacking would win on Wine. Both synthetic libraries predicted the wrong winner on both digits pools. This comparison covers two real datasets and uses repeated, overlapping splits. It establishes that matching the six fit-side summaries can leave practically important differences between fusion tasks.

## Forecasts made before external scoring

The signed gap is the regularized native OKAPI Brier minus probability-stacking Brier. Positive values favor stacking. Each forecast used the closest synthetic configuration as judged from the real bank's *fusion-fit* rows. All configurations matched the real number of providers, classes and fusion-fit rows. Forecasts came from eight synthetic draws per configuration and were [published before scoring](https://github.com/damiankucharski/fusionbench/blob/e11f462/docs/ensemble-study/external-coverage/predictions/forecasts.csv). The external assessment used eight splits per pool.

| Real provider bank | Existing forecast | Expanded forecast | Observed gap | Error, existing → expanded | Five-method rank agreement, existing → expanded |
| --- | ---: | ---: | ---: | ---: | ---: |
| Digits, LR + 5-NN + forest | −0.02286 | −0.00613 | +0.00675 | 0.02961 → 0.01288 | 0.40 → 0.60 |
| Same digits, three bootstrapped LRs | −0.02286 | −0.00570 | +0.00414 | 0.02700 → 0.00984 | 0.20 → 0.40 |
| Wine, LR + 5-NN + forest | −0.01167 | +0.00739 | +0.00825 | 0.01992 → 0.00086 | 0.40 → 0.80 |

Rank agreement is Kendall tau-b for OKAPI, mean, ranked-prefix Top-N, probability stacking and logit stacking. A tau of 1 means their order was reproduced exactly. The existing library consisted of four dimension- and budget-matched settings of the previous two multiclass generators. It chose the same high-correlation setting for every target. The expanded library chose high-signal finite votes for heterogeneous digits (B), their continuous counterpart for the three digit LRs (C), and directed confusion for Wine (E). Capturing a sparse probability profile helped matching but did not suffice for predicting the native-versus-stacking ordering. The settings are hypotheses about data-generating mechanisms; selecting one does not identify the true mechanism. [All profile distances](predictions/distances.csv) · [Forecast errors and ranks](forecast-evaluation.csv)

## What changes the competitive ordering?

- **Provider output resolution.** On heterogeneous digits 39.97% of fit probability coordinates were exactly zero, versus 0% for the bootstrap LR bank on the same images. Both old multiclass generators use continuous Gaussian logit scores. In the controlled high-separation, ten-class generator, turning on finite votes for two providers changed the OKAPI-minus-stacking gap by only −0.00043. It moved from −0.00570 to −0.00613.
- **Which wrong class receives evidence.** On hard synthetic rows we increased the logit of a predetermined wrong class, with its identity depending on the true class and provider. The ten-class comparison kept the labels, difficulty flags, Gaussian errors, provider strengths and voting scheme paired. The gap moved from −0.00613 to +0.02810, reversing sign in all eight draws. OKAPI lost 0.02039 Brier while probability stacking gained 0.01383. This intervention also changes the providers' predictive quality and output distributions. Other error structures can produce the same observed real ranking. [Paired interventions](paired-interventions.csv)
- **Calibration and class-specific information.** On heterogeneous digits, the mean scored 0.06408; sharpening that mean at the prespecified temperature 0.5 scored 0.03545, ahead of OKAPI's 0.03868. Probability stacking remained ahead at 0.03193. With a common feature scale and training objective, an unrestricted logistic stacker scored 0.03190 against 0.03416 for a stacker constrained to commute with every class permutation. These controls expose two distinct capabilities. The constrained head learns one weight per provider. The unrestricted head can learn class-specific slopes, intercepts and cross-class relationships. The contrast does not identify which coefficient structure matters. [Method means](observed-brier.csv) · [Per-split results](external-results/scores.csv)
- **Provider identity.** On the same digit splits, OKAPI selected a GP tree eight times in the heterogeneous pool, and a flat or augmented logit head eight times in the bootstrap LR pool. The best of the five primary methods changed from probability stacking to logit stacking. On Wine, the external full-versus-symmetric mean favored the unrestricted fit, although it won only three splits out of eight. Test-split overlap makes dataset-level confidence from these counts inappropriate.

The synthetic panel used 144 banks and 240 s with four workers. Building the 24 external banks took about 10 s and evaluating the nine fusion methods took about 29 s with one numerical thread. A line-search warning in the exploratory computation was traced to a numerically stationary solution of the symmetric diagnostic. The rerun used an independent gradient check and a strong-convexity bound of at most 1e−12 in the summed training objective as the condition for accepting a failed solver status. All 144 training fits passed. [Synthetic settings](synthetic-results/settings.json) · [External timing and overlap](external-results/run.json)

A subsequent external-validity test could add class-pair confusion summaries and the distribution of runner-up probabilities on **fusion-fit** rows, then compare their forecast error with the six-feature rule frozen here. Strong providers make some confusion cells very sparse; pooling counts or restricting the number of features would reduce estimation noise. New datasets and provider pools held out from generator design would be needed to determine whether those summaries genuinely predict performance elsewhere.

## Building the real prediction banks

A raw dataset supplies features and labels. A fusion benchmark needs the predictions of a *specified collection of trained providers* on rows those providers did not train on. These are different statistical objects: changing the providers changes the task faced by a fusion rule, even on identical images. The earlier digits result motivated this study. [Original generator](../../../examples/digits_predictions.py) · [Earlier digits experiment](../pattern-iterations/digits/README.md)

For heterogeneous digits, divide all 1,797 images into three disjoint stratified sets on each split. Fit three classifiers to 898 flattened 8×8 images with pixel intensities divided by 16: multinomial logistic regression (C=1, maximum 2,000 iterations), uniform five-nearest-neighbors, and a 100-tree random forest. Freeze them. Each emits ten probabilities per image, giving arrays of shape (449,3,10) for training fusion and (450,3,10) for evaluation. Both stacking and OKAPI learn from the first array and its labels. The base logistic regression learns from image pixels. The logistic stacker learns from all 30 provider probabilities for each of the 449 images. No prediction used to fit or score fusion is an in-sample prediction from a base estimator. The saved row IDs make the separation inspectable.

The second digits bank uses three logistic regressions trained on stratified bootstrap samples of the same 898 images, with exactly the same fusion and test rows. The separate three-class Wine bank uses the original three classifier families with a standardizer fitted on 89 base-training rows and 44/45 rows for fusion-fit/test. A dataset-level generalization claim would require more independent tasks and provider pools. Digits partitions overlap across seeds and its two pools reuse the same images; Wine has few held-out examples. [Manifest and split hashes](external-results/fit_manifest.json) · [Reference scenario coverage](inventory.md)

## Use a released bank directly

The fit/test probability tensors, labels and row IDs from all 24 real banks are included in the repository. Export one split to the standard FusionBench dataset format, then compare other fusion methods on those same predictions. This example uses the public methods' default settings. The research driver below sets the study's exact search seeds.

```sh
python docs/ensemble-study/external-coverage/export_bank.py --suite digits --seed 3140000 --out /tmp/fusion-banks
fusionbench evaluate --dataset /tmp/fusion-banks/digits_3140000 --methods simple_average top_n stacking_lr okapi_native_pattern_l2 --out /tmp/fusion-results
```

Use `digits_lr_bag` with seeds 3140000–3140007 or `wine` with seeds 3150000–3150007 for the other banks. The [manifest](external-results/fit_manifest.json) names each output column's provider and records the partition fingerprints.

## Reproduction

Use the research checkout with its optional pinned OKAPI dependency (e.g. `uv sync --extra okapi` and activate its environment). Run from the FusionBench root. The default numerical backend in the study is NumPy with one BLAS thread per worker. An honest forecast requires running the fit-only and synthetic stages, freezing the forecast, and then evaluating the target test data. All necessary raw image and Wine measurements come with scikit-learn.

```sh
export BACKEND=numpy OKAPI_LOG_LEVEL=ERROR OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
study=docs/ensemble-study/external-coverage
out=/tmp/fusion-coverage
python "$study/external.py" --profiles-only --out "$out/external-results"
python "$study/synthetic.py" --workers 4 --out "$out/synthetic-results"
python "$study/forecast.py" --synthetic "$out/synthetic-results" --profiles "$out/external-results/fit_profiles.csv" --out "$out/predictions"
python "$study/external.py" --out "$out/external-results"
python "$study/evaluate_forecasts.py" --root "$out"
```

The new generator is available as `generate_dataset("multiclass_provider_panel", ...)`. The standard v1 reference pack remains pinned to the original nineteen. [Scenario knobs and paired interventions](../../site/scenarios.md) · [Frozen protocol](protocol.md)

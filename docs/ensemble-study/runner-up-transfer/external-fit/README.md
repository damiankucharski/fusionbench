# Real provider predictions for forecasting

Each prediction tensor has shape (examples, providers, classes) and dtype float32. The corresponding label and source-row arrays have dtype int64. Load a fit split with `np.load("fit_banks.npz")` and keys such as `digits_3300000_inputs`, `digits_3300000_labels` and `digits_3300000_rows`. The eight seeds per task begin at the roots in the table. The fusion-fit counts mirror the synthetic strata. The held-out test counts follow from each source dataset's size. [Arrays](fit_banks.npz) · [Split hashes, versions and provider order](fit_manifest.json) · [Fit profiles](fit_profiles.csv) · [Provider profiles](fit_providers.csv)

| Dataset | Seed root | Classes | Base train / fusion fit / test | Providers, in tensor order |
|---|---:|---:|---:|---|
| scikit-learn digits | 3300000 | 10 | 898 / 449 / 450 | Gaussian naive Bayes (variance smoothing 0.01), distance-weighted 11-nearest neighbors, 100 extremely randomized trees |
| scikit-learn Iris | 3310000 | 3 | 74 / 44 / 32 | Logistic regression (C=1), uniform 5-nearest neighbors, 100-tree random forest |
| UCI Seeds | 3320000 | 3 | 104 / 44 / 62 | Logistic regression (C=1), uniform 5-nearest neighbors, 100-tree random forest |

The first stratified split draws base-training rows at the bank seed. The second uses seed + 1 to draw fusion-fit rows from the remainder. Each source model trains exclusively on its base-training partition. Digits pixel values are divided by 16. Iris and Seeds use a feature standardizer fitted on base-training rows. Both forest types use the bank seed. The logistic iteration cap is 2,000. The fit-only stage computes provider probabilities only for fusion-fit rows. Test row IDs have a recorded hash so the split can be checked before external scoring. The same source row can occur in multiple repetitions.

## Seeds source and credit

M. Charytanowicz, J. Niewczas, P. Kulczycki, P. Kowalski and S. Lukasik (2010), *Seeds*, UCI Machine Learning Repository. The original has seven continuous kernel measurements and 210 observations, with 70 each from three wheat cultivars. Row and feature order are preserved. Published cultivar IDs 1, 2 and 3 are recoded to 0, 1 and 2. The probabilities are derived from fitted classifiers. The ZIP archive is cached outside this repository, and both its SHA-256 and the checksum of its sole data member are pinned in the manifest and driver. [Dataset and citation](https://doi.org/10.24432/C5H30K) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) · [Official ZIP](https://archive.ics.uci.edu/static/public/236/seeds.zip)

From the FusionBench repository root, use the study's pinned research environment and one numerical thread. Supplying `--seeds-archive PATH` reuses an offline copy of the official ZIP.

```sh
export BACKEND=numpy OKAPI_LOG_LEVEL=ERROR LOGURU_LEVEL=ERROR
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1
python docs/ensemble-study/runner-up-transfer/external.py --profiles-only --out results/runner-up-transfer
```

# Digits on the same source population

Regularized OKAPI beat the arithmetic mean on all eight held-out splits at both fusion-training sizes. Increasing the fusion-training sample from 100 to 449 digits reduced its average test Brier from **0.03039 to 0.02750**. Six splits improved, with paired changes ranging from −0.00934 to +0.00386. Logistic stacking had the lowest average test Brier at both sizes. Its advantage over OKAPI was 0.00304 at 100 examples and 0.00510 at 449.

| Method | Fit Brier, 100 | Test Brier, 100 | Fit Brier, 449 | Test Brier, 449 |
|---|---:|---:|---:|---:|
| Arithmetic mean | 0.06568 | 0.05632 | 0.05960 | 0.05632 |
| Ranked-prefix Top-N | 0.04310 | 0.03257 | 0.03461 | 0.03044 |
| Probability logistic stacking | 0.00939 | 0.02735 | 0.00905 | 0.02240 |
| Regularized OKAPI | 0.03788 | 0.03039 | 0.03108 | 0.02750 |
| Holdout selection, positive gain | 0.04739 | 0.04048 | 0.03108 | 0.02750 |
| Holdout selection, gain above one standard error | 0.04739 | 0.04048 | 0.03108 | 0.02750 |

Regularized OKAPI selected its GP finalist in all 16 fits. The separate source-only comparison with the mean made the same deployment decisions under both thresholds. At 100 training digits it selected the mean on three splits, raising average test Brier from 0.03039 to 0.04048. At 449 training digits it retained OKAPI on every split. Logistic stacking scored below OKAPI on six of eight small-sample tests and seven of eight larger-sample tests. Although its mean training loss was lower than its test loss at both sizes, it still achieved the best average test loss. A fit-to-test gap alone would give an incomplete method comparison here.

## Data and evaluation

Each of eight prespecified seeds, 256100–256107, gives a class-stratified partition of sklearn’s 1,797 ten-class digits. The existing generator divides pixel values by 16 and uses 898 digits to train logistic regression (2,000 maximum iterations), five-nearest neighbors, and a 100-tree random forest. A separate 449 digits train the fusion methods and 450 are held out. For the small-sample condition we took a stratified 100-digit subset of the 449, using random state `seed+2`. Both sample sizes share the same fitted base models and test rows. All labels were used as supplied. Source-only holdout selections use the same helper as the synthetic check and deploy either the mean or the regularized fit trained on the full available fusion sample. Search and native split seeds are zero.

The test splits share 98–129 individual digits per pair. The table reports descriptive averages across these partitions of one dataset. The paired changes and individual held-out scores are available for inspection. A separate check reproduced the split indices and labels, aggregated all 46,368 per-digit fit and test losses, and verified 96 fit/test score pairs and 32 gate decisions.

[Protocol](protocol.md) · [Runner](digits.py) · [Original generator](../../../examples/digits_predictions.py) · [Split scores](results/digits-scores.csv) · [Paired changes](results/digits-paired.csv) · [Fit choices and input hashes](results/digits-choices.csv)

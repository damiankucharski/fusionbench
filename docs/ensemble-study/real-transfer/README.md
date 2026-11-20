# Can one supervised network recover an ensemble improvement on real data?

Neither dataset met all three frozen criteria. On smartphone activity recognition, the wider network recovered 22 of 33 classification errors fixed by the selected ensemble. It also made 34 new errors on windows that the chosen component and ensemble both classified correctly. On Wisconsin, the selected fusion and component made identical class decisions on every test example.

## Held-out results

Lower Brier is better. The loss sums across classes before averaging examples. HAR weights each test subject equally. The strongest component and the fusion were chosen separately on the selection split.

| Test set | Validation-chosen component | Validation-chosen fusion | Single 72-unit MLP | Mean fusion | Lowest component risk in hindsight |
|---|---:|---:|---:|---:|---:|
| Wisconsin, 114 cases | .06158 | .06647 (mean) | .06382 | .06647 | .06158 |
| HAR, 2,947 windows from 9 held-out people | .09848 | .08812 (OKAPI) | .09143 | .08483 | .08914 |

The HAR selection set contained four people. It favored OKAPI (.22275) over averaging (.22384). On unseen people, averaging had lower risk. The lowest individual risk on test is included to show sensitivity to member selection. Choosing that member from test scores would require test labels.

| Primary comparison | Wisconsin gain [lower bound] | HAR gain [lower bound] |
|---|---:|---:|
| Chosen fusion vs selected component | −.00489 [−.01056] | +.01036 [−.00070] |
| Single model vs selected component | −.00224 [−.01473] | +.00706 [−.01035] |
| Single model vs chosen fusion | +.00265 [−.00697] | −.00330 [−.02682] |

Positive gains favor the first model in the comparison. The one-sided bootstrap lower bounds use a Bonferroni correction across six checks. The single model met the prespecified .01 Brier noninferiority criterion on Wisconsin, where there was no established ensemble improvement to recover. HAR's positive estimate for ensemble gain is compatible with zero at the prespecified threshold. The data also leave open a student shortfall greater than .01. They do not establish equivalence. The small number of subjects limits inference about the lower tail. These intervals condition on one split and one initialization of each replacement model.

On HAR, test accuracies were 93.68% for the validation-selected member, 94.28% for OKAPI and 94.60% for the primary student. The per-window count of 33 repairs includes 16 accompanying ensemble harms. These raw counts weight each window once; accuracy and Brier give each person equal weight. On Wisconsin, the ensemble and component agreed on all 114 decisions and both got 111 right. The student introduced one error among those 111 examples.

## Resource comparison and controls

All students trained from random weights using true labels. No teacher output entered their loss. The primary MLP could use the labels that had been reserved for GP fitting. The base-only 72-unit control reached .06903 on Wisconsin and .09259 on HAR. With the same expanded pool as the primary MLP, the 24-unit control reached .07103 and .09463.

On HAR, a single 72-unit model has 40,902 parameters and 40,824 dense multiply-adds per example. Three 24-unit components together have 40,914 parameters and 40,824 dense multiply-adds. The single model used 1,850 updates, compared with 5,550 smaller updates across three components. At inference a three-node OKAPI tree needs at most two distinct components, so the wide network's compute budget is matched to the full three-member pool and can exceed the selected tree's cost. The single model used 1.93 seconds of training time on HAR. The three members used 4.58 seconds combined; GP fitting took 0.39 seconds. End-to-end computation for both datasets and all controls took 16.02 seconds wall time and 15.97 seconds of process CPU. Times include initialization overhead and exclude downloading the data. [Raw training costs](fit_costs.csv)

## Reproduce

Use the FusionBench research branch with OKAPI revision `968baa3955f151902d12aecae261f5e05a82d78b`. The protocol was committed as `3131fe5` before training. From a checkout with its dependencies installed, run:

```sh
curl -fL 'https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip' -o har.zip
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python examples/real_transfer.py --har-archive har.zip --out results/real-transfer
```

Wisconsin uses the original binary labels and bundled features. HAR uses the released 561 sensor-window features and all six activities. Both network types see the same feature representation. This experiment does not evaluate a recognizer trained on raw accelerometer waveforms or clinical use. The driver verifies the public archive's hash and the absence of split overlap. [Wisconsin source](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) · [HAR source](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)

[Frozen protocol](protocol.md) · [All test scores](scores.csv) · [Prespecified contrasts](contrasts.csv) · [Split fingerprints, timing and denominators](details.json)

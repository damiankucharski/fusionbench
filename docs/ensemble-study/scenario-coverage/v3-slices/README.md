# How does fusion behave on the cases each scenario was designed to expose?

Six paired synthetic banks per arm used 512 fit and 4096 test rows. The four new defaults and their controls share a test protocol with the earlier ten-class scenario. Values are the unweighted mean of bank-level full-class Brier scores. Lower is better. Methods fit on provider outputs and labels. The evaluator alone uses the hidden region, environment, difficulty and realized test labels to identify subsets. [Design and metrics](protocol.md) · [Scores and counts by bank](banks.csv) · [Paired differences, ranges and counts](paired.csv)

| Default scenario | Native OKAPI L2 | Mean | Top-N | Probability LR | Observable oracle |
| --- | ---: | ---: | ---: | ---: | ---: |
| Rare specialist | 0.3763 | 0.4470 | 0.3772 | 0.3772 | 0.3716 |
| Redundant pool | 0.4191 | 0.4094 | 0.4082 | 0.4204 | 0.4022 |
| Environment shortcut | 0.7157 | 0.6832 | 0.6832 | 0.6835 | 0.2805 |
| Selective fallback | 0.2584 | 0.2904 | 0.2889 | 0.2523 | 0.2482 |
| Similar classes 3/8 | 0.0161 | 0.0660 | 0.0620 | 0.0200 | unavailable |

**The rare expert helps on a small identifiable subset.** The private rare region supplied 42–56 fit rows, including 10–17 in native validation, and 367–450 test rows per bank. Native OKAPI reduced rare-region Brier against the generalist from 0.3794 to 0.3527 in aggregate and improved it in all six banks. On common rows it moved from 0.3771 to 0.3790. Setting the specialty gain to zero preserved region memberships and generalist predictions. In this control native Brier on the rare rows averaged 0.3929. The observed inputs include a noisy region cue; the generator's posterior marginalizes the true region.

**An extra candidate can be either redundant or sufficient.** With the canonical null panel the anchor is the true posterior. Native OKAPI's mean squared distance from that posterior was 0.01647 across banks. Permuting the providers' positions independently across banks gave 0.01281 in the null arm. With the same positions and a label-informative candidate in the matched positive arm, the empirical Brier fell from 0.4151 to 0.2787. Top-N matched the planted posterior on every positive-control test row in all six banks. The positive arm puts that exact posterior in one available column.

**A shift in the environment mixture reverses the deployment result.** The native learner saw 43–67 reversed-environment rows in fit. A stationary test mixture gave it 0.2852 Brier versus 0.3772 for the stable provider. The shifted test had 3234–3301 reversed rows per bank and gave native 0.7157 versus the same stable score. The native model's Brier stayed high within reversed rows in both settings (0.8151 stationary, 0.8093 shifted). Its test squared distance to the split-specific observable posterior rose from 0.00741 to 0.43447. The known stable provider outperformed it on every shifted bank.

**Losing a report on difficult examples changes which method wins within the fallback slice.** Both fallback mechanisms censored exactly 128 fit rows and 1024 test rows per bank. In the default difficulty-dependent arm native OKAPI scored 0.4970 versus LR's 0.5033 on censored rows and 0.1788 versus 0.1686 on available rows. Randomizing censoring at the same frequency gave 0.3757 versus 0.3759 on censored rows. These conditional scores average over different selected cases. Across the full paired banks the reference provider retained 0.3713 in both arms.

**Hard cases explain most of the native advantage on the multiclass control.** On hard examples native OKAPI scored 0.1042 versus 0.1303 for LR and won on all six banks. On easy examples both rounded to 0.0005. Across labels 3 and 8 the comparison was 0.0185 versus 0.0203, with native lower on four banks. The intersection of hard examples and labels 3/8 contained 113–133 test cases and only 1–7 native-validation cases per bank. The bank table includes separate rows for each label and for the other eight labels.

For groups defined by latent state or realized labels, squared distance to the observable posterior describes fidelity to that predictor. A group-conditioned learner with access to additional information can attain a different conditional risk. The finite-bank paired loss versus the observable oracle is recorded separately. These six replicates characterize the supplied synthetic mechanisms. They do not establish external performance or a uniform method ranking.

## Reproduction

From the source checkout with its pinned OKAPI extra installed, run

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. \
python docs/ensemble-study/scenario-coverage/v3-slices/run.py --workers 2
```

The program writes one bank-level record per method and slice plus paired differences (native minus comparator). A single-worker replay of the first replicate reproduced all 194 published bank–method–slice records exactly at the stored precision. For each partition, its test-count-weighted slice scores recover that bank's pooled test Brier. Only subsets defining a partition should be combined this way. For example, hard cases and labels 3/8 overlap. [Evaluator](run.py)

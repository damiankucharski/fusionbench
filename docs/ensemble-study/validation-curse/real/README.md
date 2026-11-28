# Conservative OKAPI search on four small real classification datasets

Subtracting one paired standard error from each discovery score changed held-out full-class Brier from **0.075916 to 0.075957** when each dataset had equal weight. The penalty helped on UCI Seeds (−0.00389) and hurt on a heterogeneous Digits pool (+0.00600). The matched unpenalized search reproduced the deployed incumbent's prediction digest on 48 of 48 partitions. [Frozen protocol](protocol.md) · [Round-two lock](../round2-lock.md) · [All paired differences](results/pool-differences.csv)

Each partition had three trained providers. We fitted the fusion methods using 64 labeled examples on Digits and 44 on Iris, Wine and Seeds. For the primary comparison, we used the same native initialization, split, GP allowance, L2 strength (.25), family selection and deployment procedure for both values of the penalty. We kept all three final candidates per arm. Every method saw the same development and test partitions. Top-N ranked the three providers once on development Brier and tested all three prefixes. [Bank creation, row IDs and split checks](source-audit.json) · [Driver](run.py)

## Held-out full-class Brier

Lower is better. Rows average eight paired partitions. The four-dataset mean first averages the three provider pools trained on Digits. The six-pool mean gives the three Digits pools half of the total weight. Probability and logit stacking use multinomial logistic regression with C=1. All methods use only the small development set to fit fusion.

| Pool | Mean | Top-N | Probability LR | Logit LR | OKAPI z=0 | OKAPI z=1 | Raw outer choice | Guarded outer choice |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Digits A | .06408 | .04073 | .04092 | .06221 | .04037 | .04010 | .04354 | .04599 |
| Digits B | .07213 | .04876 | .05593 | .09609 | .05486 | .06086 | .07167 | .08021 |
| Digits logistic bag | .08620 | .08813 | .07897 | .11572 | .08380 | .08417 | .08095 | .08468 |
| Iris | .07613 | .08884 | .09957 | .07581 | .08109 | .08357 | .08400 | .07613 |
| Seeds | .12225 | .12660 | .12762 | .13383 | .13286 | .12898 | .12701 | .12225 |
| Wine | .04481 | .03605 | .02178 | .03943 | .03004 | .02957 | .03440 | .03497 |
| **Four datasets** | **.07933** | **.07767** | **.07689** | **.08510** | **.07592** | **.07596** | **.07770** | **.07591** |
| Six pools | .07760 | .07152 | .07080 | .08718 | .07050 | .07121 | .07359 | .07404 |

Across the 48 comparisons, z=1 improved Brier 20 times, tied 13 times within 1e−12 and worsened it 15 times. Four-dataset accuracy was 94.99% at z=0 and 95.00% at z=1, against 95.09% for the mean. Nominally identical native allowances include the same 600 GP discovery evaluations per bank. Adaptive feature screening, validation and head fitting made the total number of score calls 204,713 for z=0 and 217,330 for z=1 (+6.16%). [Exact Brier](results/pool-brier.csv) · [Accuracy for every method and pool](results/pool-accuracy.csv) · [Score-call accounting](results/budget.csv)

The largest worsening occurred on Digits B partition 3300007. Both procedures selected their GP finalist. That finalist's held-out Brier rose from .07780 to .13769. The penalized search's augmented candidate scored .07979 and the fixed arithmetic mean .08065. The example illustrates both a large change in available GP predictions and an opportunity for a different terminal choice. Its test outcomes were opened after fitting. Across the four datasets, giving a retrospective analyst the best test Brier among the three frozen finalists yielded .07069 at z=0 and .07060 at z=1. The corresponding gaps from the actual choice were .00522 and .00536. Choosing after observing the test labels can inflate the apparent benefit of selecting another finalist. [All 144 final-family comparisons, including the incumbent](results/final-family-oracle.csv) · [Aggregate retrospective headroom](results/pool-oracle.csv)

An additional replay after the prespecified comparison refitted all 96 primary models and compared the *same pre-refit winner* on its internal validation examples and on the untouched test. Every final deployment digest was reproduced. In the four-dataset average, the observed gain over the arithmetic mean for z=0 fell from .02665 on validation to .00347 on test. At z=1 it fell from .02654 to .00338. Relative optimism was therefore .02319 and .02317 respectively. This diagnostic removes the effect of head refitting from the comparison and shows how much the advertised internal-validation gain shrank on these archives. [Individual replay and call breakdown](results/discovery-replay.csv) · [Summary by dataset and provider bank](results/pool-discovery.csv)

## A separate holdout for the final choice

Both outer selectors saw the same four frozen predictions on the same 14 or 20 gate examples. The raw selector used the lowest gate Brier. The guarded selector accepted the best challenger only when its estimated gain over the mean exceeded a shared threshold from 4,095 centered sign-flip draws. The native search and refits for these candidates used the remaining 30 or 44 development examples. The inner incumbent used its ordinary family choice on these reduced training inputs. Its four-dataset test Brier was .07627 compared with .07592 for the full-development incumbent. [Gate decisions and relative optimism](results/gate.csv)

| Pool | Mean chosen, raw / guarded (of 8) | Guarded minus raw Brier | Partition range |
|---|---:|---:|---:|
| Digits A | 0 / 1 | +.00245 | .00000 to +.01961 |
| Digits B | 1 / 4 | +.00854 | .00000 to +.03088 |
| Digits logistic bag | 0 / 6 | +.00374 | .00000 to +.00673 |
| Iris | 2 / 8 | −.00787 | −.03870 to +.00269 |
| Seeds | 2 / 8 | −.00476 | −.02653 to +.01227 |
| Wine | 1 / 3 | +.00057 | −.00209 to +.00668 |

The guard selected the arithmetic mean on 30/48 partitions, up from 6/48 with raw selection. Relative selection optimism is the increase from gate to test in the selected model's Brier difference against the same fixed mean. It fell from .01269 to .00364 with equal dataset weights. The guard also improved held-out Brier by .00178 over raw at those weights. With equal provider-pool weights the guard worsened Brier by .00045. It chose the mean on all 16 Iris and Seeds banks, where that fallback was helpful on average. On the three Digits pools it never improved on the corresponding raw choice. [Paired ranges](results/paired-direction.csv) · [Every selected family and Top-N prefix](results/selected-frequencies.csv) · [Gate and test decomposition](results/pool-gate.csv)

## Scope and reproduction

All 48 partitions separate base-model fitting, fusion development and test observations. The test observations repeat across the eight partitions of each pool. The three Digits pools contain 10,800 test positions covering 1,776 distinct images, with Digits A and the logistic bag using identical base, development and test IDs on all eight paired partitions. The source audit found 159 distinct test examples on Wine, 129 on Iris and 198 on Seeds. The four datasets had been used in earlier public comparisons, including the known strength of averaging on Iris and Seeds. These descriptive differences do not support independent-replicate uncertainty intervals, transfer to a new population or a claim about MedMNIST or STARCOP. Their aligned historical probability banks were unavailable for this run. [Independently reconstructed partitions](verify_real.py) · [Historical baselines](../../class-capacity/README.md)

With the matched OKAPI and FusionBench revisions on the Python import path, run from the FusionBench root:

```bash
python docs/ensemble-study/validation-curse/real/verify_real.py
python docs/ensemble-study/validation-curse/real/run.py --workers 2
python docs/ensemble-study/validation-curse/real/summarize.py
python docs/ensemble-study/validation-curse/real/discovery_replay.py
python docs/ensemble-study/validation-curse/real/verify_results.py
```

The compact archive contains prediction digests and Brier for every fitted model and candidate, the selected source example IDs, every held-out row's Brier for each candidate, choices and retrospective comparisons. [Per-bank scores](results/scores.csv.gz) · [Source IDs and choices](results/banks.jsonl.gz) · [Per-example scores](results/test-row-losses.npz)

An independent verifier aggregated 273,976 archived per-example losses and reproduced all 1,104 reported Brier scores. It also reproduced 144 retrospective selections and checked the 192 historical-incumbent versus z=0 hashes for the deployed prediction plus its three candidates. Reconstructing all four cheap baselines from the source probabilities reproduced all 192 prediction hashes. Two independently refitted guard decisions cover an acceptance (Digits A partition 0, .04283 gain against a .03082 threshold) and a fallback to the mean (Iris partition 0, .00652 gain against a .05042 threshold). [Numerical verifier](verify_results.py) · [Verification details](verification.json)

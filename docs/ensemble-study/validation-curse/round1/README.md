# Round 1. A larger menu can make the selected predictor look better while its true risk rises

With 32 validation rows, the 64-candidate selector rejected the optimal arithmetic mean in **32 of 32 independent banks**. Its validation Brier was 0.3512, its independent audit Brier was 0.4171, and its exact population Brier was 0.4200. The mean's population Brier was 0.4000. In the paired setting with a genuinely better provider, the same selector found the optimal rule on 5 of 32 banks. With 512 validation rows it found that rule on all 32. [Frozen protocol](protocol.md) · [Raw selections](results/selections.csv)

## Changing the selection budget and validation size

Twelve complementary binary provider predictions average to a known anchor probability. The menu starts with this mean. The other 63 fixed candidates add or subtract 0.10 based on observable provider contrasts. In the null setting the anchor is the true probability. In the information setting the first provider is the true probability and candidate 1 returns it. Both candidate menus and validation samples are nested. The expectation over all 256 feature states gives each candidate's exact risk under both source and target laws. [Population risks](results/population.csv)

| Null validation size | Expected regret, menu 1 | Menu 4 | Menu 16 | Menu 64 | Mean selected at menu 64 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 0 | 0.01125 | 0.02000 | 0.02000 | 0/32 |
| 128 | 0 | 0.00438 | 0.01563 | 0.02000 | 0/32 |
| 512 | 0 | 0 | 0.00125 | 0.00500 | 24/32 |

The entries average each *selected* predictor's exact source risk minus the best available predictor's exact risk. Moving from one to 64 candidates raised the null regret by 0.020 at 32 validation rows. The extra optimism in apparent gain over the fixed anchor was 0.07660 (paired 95% interval 0.07089–0.08231). At 512 rows the extra optimism was 0.00527 (0.00183–0.00871). The optimism contrasts subtract the fixed anchor's audit-minus-validation gap on the same samples, which removes some sampling variation in example difficulty. [All validation, audit, IID and shifted means](results/selection-means.csv) · [Paired comparisons](results/selection-contrasts.csv)

| Information setting, menu 64 | Correct candidate | Source selection regret | IID test Brier | Shifted test Brier |
| ---: | ---: | ---: | ---: | ---: |
| 32 validation rows | 5/32 | 0.03375 | 0.41558 | 0.37908 |
| 128 validation rows | 18/32 | 0.01750 | 0.39827 | 0.39575 |
| 512 validation rows | 32/32 | 0 | 0.38055 | 0.41286 |

Moving from 32 to 512 validation rows improved IID test Brier by 0.03503 (0.02934–0.04072) and worsened shifted test Brier by 0.03378 (0.02828–0.03928). The shifted domain gives more mass to extreme anchor values and reverses the informative provider contrast. Successful source selection therefore carries its own target-domain cost. The intervals average independent bank differences across 32 seeds. [Split seeds and hashes](results/banks.csv)

## Development-fitted references

The four methods below received the twelve provider outputs. The table uses the eight paired banks for which all four methods were evaluated. The 64-candidate menu comprises the separately specified fixed rules. In the information setting, ranked Top-N could select the informative provider alone.

| IID full-class Brier | Arithmetic mean | Brier Top-N | Probability LR | Regularized native OKAPI |
| --- | ---: | ---: | ---: | ---: |
| Mean optimal | 0.40173 | 0.40641 | 0.40745 | 0.41337 |
| One provider informative | 0.40303 | 0.38368 | 0.38885 | 0.39424 |

All four references used 640 development rows, and their predictions on validation, audit and both test sets were recorded. The three inexpensive controls were also fitted on the remaining 24 banks of each arm. This round isolates choosing among prespecified functions on independent rows. Selection among candidates trained adaptively on the same validation labels, dependence within real groups and uncertainty about an unknown evaluation distribution require separate designs. [Individual comparator scores](results/references.csv) · [Matched control summaries](results/reference-summary.csv)

## Reuse and reproduction

`fusionbench.eval.validation_curse.audit_menus` accepts row-by-candidate loss matrices for validation, an independent clean audit, an IID test and a shifted test. Pass the candidate-prefix sizes and validation-prefix sizes to be compared. Supply population risks when known to obtain exact selection regret. The function reports the independent audit gap when population risks are unavailable. The study runner uses a tolerance of 10⁻¹² in mean Brier to recognize mathematical ties in floating point. The separate verifier replays every decision on the integer grid.

```sh
study=docs/ensemble-study/validation-curse/round1
BACKEND=numpy OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python "$study/run.py" --out /tmp/fusion-curse-r1
python "$study/summarize.py" --out /tmp/fusion-curse-r1
python "$study/verify.py" --out /tmp/fusion-curse-r1
```

The archived run took 10.02 seconds. The independent integer calculation reproduced all 768 winners, their held-out scores and exact regrets. The archive contains 320 split records and 208 fitted-comparator records. [Audit implementation](../../../../fusionbench/eval/validation_curse.py) · [Runner](run.py) · [Verification](verify.py) · [Run manifest](results/run.json)

# Eight terminal probes on the 25 standard scenarios

Across 100 fresh banks, offering distinct provider pairs within each operator gave a mean test Brier of **0.31831**, compared with **0.31893** for draws with replacement. The paired difference (unique minus IID) was **−0.00062**, with a 95% interval from **−0.00153 to +0.00019**. Three scenario means improved, three worsened and nineteen tied. The policies made byte-identical predictions on 91 of 100 test banks. This result leaves the incremental value of avoiding repeats unresolved. [Protocol](r3-protocol.md) · [Paired contrasts](results/r3-contrasts.csv)

Both terminal menus improved on the frozen GP finalist, which scored **0.32872**. Offering distinct pairs lowered test Brier by **0.01041** (95% interval 0.00552–0.01537). Offering IID pairs lowered it by **0.00979** (0.00532–0.01435). The largest family gains from adding the distinct-pair menu occurred on `parity` (−0.08813), `weakest_link` (−0.08145), and `hardrescue` (−0.08113). Compared directly with IID, the largest gain was on `hardrescue` (−0.01030), and the largest regression was on `hardveto` (+0.00250). Each family mean uses four data draws. [All family contrasts](results/r3-family-contrasts.csv) · [Scores by bank](results/r3-scores.csv)

| Method | Mean held-out Brier |
| --- | ---: |
| Ordinary native GP finalist | 0.32872 |
| GP plus eight IID pair probes | 0.31893 |
| GP plus eight probes from shuffled pair cycles | 0.31831 |
| Arithmetic mean | 0.39149 |
| Brier-ranked Top-N | 0.35381 |
| Probability logistic stacking | 0.26085 |
| Regularized native-pattern OKAPI | 0.23170 |

Across all banks, the two policies offered 6.74 and 7.04 distinct operator–pair combinations on average, respectively. Each paid for eight fit evaluations per bank. Some scenarios have only two or three providers and exhaust their pair catalog within four slots for one operator. The fit-selected winner was a probe in 15 banks under IID sampling and 14 under distinct-pair sampling. In the other banks the ordinary finalist remained deployed. Distinct-pair deployment improved test loss relative to this finalist in eleven banks and worsened it in three. [Offered programs and their fit and test losses](results/r3-proposals.csv) · [Method means](results/r3-means.csv)

## Design and reproduction

We generated 512 fit and 512 held-out test observations per scenario and data seed. A single ordinary native GP search was fitted on each bank at search seed zero. Both post-search menus reused its frozen finalist. Each menu interleaved four native MIN and four MAX trees across the full provider palette. Selection minimized unrounded fit Brier with the finalist first on exact ties. Each chosen tree was then evaluated on fresh test inputs. The bootstrap resampled four paired data-seed indices independently within each fixed family. The same data seeds also determined the pair schedules. The panel describes these generators, menu size and search seed.

The 100 GP searches averaged 205.24 objective evaluations and 227.48 offspring proposals each. The 200 terminal menus added 1,600 fit evaluations. For diagnostics, all 1,600 offered trees also received a test forward pass after the deployment choices were frozen. All controls and bookkeeping took part in the same 59.4-second run using one CPU worker with one numerical thread. [Run settings and timing](results/r3-manifest.json)

From FusionBench, set `PYTHONPATH` to the [matching OKAPI checkout](https://github.com/damiankucharski/OKAPI/commit/92f0e96) and the FusionBench root. The smoke flag checks binary and multiclass native evaluations, a one-pair palette and exact-loss ties.

```sh
export PYTHONPATH="/path/to/OKAPI:."
export BACKEND=numpy OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OKAPI_LOG_LEVEL=ERROR
python docs/ensemble-study/pair-coverage/r3_panel.py --smoke
python docs/ensemble-study/pair-coverage/r3_panel.py
```

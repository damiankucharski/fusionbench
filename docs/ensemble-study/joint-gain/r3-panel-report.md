# Does breeding on joint gain improve performance across 25 scenarios?

With the same terminal selection rule, joint-gain breeding scored **0.32307** full-class test Brier and fold-gain breeding scored **0.32514**. The paired difference was **−0.00207** (95% interval −0.01186 to +0.01043). Eight scenario means improved, four tied and thirteen worsened. The interval spans zero and more families worsened than improved. The result misses both prespecified criteria for recommending the change across this suite. [Protocol](protocol.md) · [Paired contrasts](results/r3-panel-contrasts.csv)

The largest improvement came from `weakest_link` (−0.10718 on average, with three of four data seeds improving). The largest regressions were `hardveto` (+0.06793, with three losses and one tie), `two_pathways` (+0.00574) and `compositional_switch` (+0.00502). At the individual bank level the two procedures issued identical predictions in 51 of 100 comparisons. Joint breeding improved test Brier in 25 banks and worsened it in 24. [Family contrasts](results/r3-panel-family-contrasts.csv) · [Bank-level scores and prediction hashes](results/r3-panel-scores.csv)

| Procedure | Mean test Brier | Mean number of distinct providers used |
| --- | ---: | ---: |
| Joint breeding, joint terminal | 0.32307 | 4.36 |
| Fold breeding, joint terminal | 0.32514 | 4.13 |
| Fold breeding, fold terminal | 0.32543 | 4.32 |
| Ordinary pooled GP | 0.31803 | 4.28 |
| Initial anchor alone | 0.38621 | 8.58 |
| Arithmetic mean | 0.39426 | 10.36 |
| Brier-ranked Top-N | 0.35780 | 4.24 |
| Probability logistic stacking | 0.26092 | 10.36 |
| Regularized native-pattern OKAPI | 0.23037 | 9.73 |

Against ordinary pooled GP, the joint procedure scored +0.00504 (−0.00486 to +0.01479). Against fold breeding and fold deployment, it scored −0.00236 (−0.01219 to +0.01018). On the joint-bred trajectories, the joint terminal rule and the pooled-gain veto selected identical native programs on all 100 banks. [All 18 methods](results/r3-panel-means.csv) · [Method means by family](results/r3-panel-family-means.csv)

## Design and cost

Each family supplied four fresh data draws (seeds 288400–288403), each with 512 fit and 512 held-out test rows. Each fit row formed one fold-assignment group. The three searches shared their founders, anchor, folds and search seed zero. The four terminal rules reused each frozen path. The panel therefore ran 300 native searches and assessed 1,200 path/terminal combinations, 100 standalone anchor decisions, 100 ordinary pooled terminal decisions and 400 reference fits or predictions. Each winning native program received an independent held-out forward pass that reproduced its recorded Brier. The interval resamples the four paired data seeds within each of the 25 fixed families. It describes variation across these data draws at the fixed search seed. [Generation traces](results/r3-panel-trajectories.csv.gz) · [Run manifest](results/r3-panel-manifest.json)

Pooled, fold and joint breeding used 204.36, 204.96 and 206.65 objective evaluations per path on average. They offered 227.58, 227.93 and 228.29 offspring respectively. All fits, scoring and aggregation took 169 seconds on one CPU worker with one numerical thread. The search-cost fields shared by the four terminal deployments describe their common fitted path. The initial anchor can be selected using only the founder scan. [Path costs](results/r3-panel-path-costs.csv)

From the FusionBench checkout, substitute the path to the [matching OKAPI checkout](https://github.com/damiankucharski/OKAPI/commit/697335b) in the following command.

```sh
PYTHONPATH="/path/to/OKAPI:." BACKEND=numpy OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  OKAPI_LOG_LEVEL=ERROR python docs/ensemble-study/joint-gain/round3_panel.py
```

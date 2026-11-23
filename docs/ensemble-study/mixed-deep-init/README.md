# Six balanced starters among thirty shallow OKAPI proposals

At a fixed budget of 600 GP evaluations, replacing the last six shallow starters with balanced eight-input trees improved held-out Brier on CompositionalSwitch in all eight fresh data banks compared with six size-matched chains. The paired mean difference was −0.07073 (95% t interval −0.11240 to −0.02906), after averaging two search starts per bank. The balanced group and the chain group used exactly the same eight provider occurrences and seven MIN/MAX operators per starter. Only the grouping changed. Both retained the native singletons and the same first 24 shallow starters. [Frozen design](protocol.md)

| Full-class test Brier, lower is better | CompositionalSwitch IID | Mean of 19 families IID | Mean of 9 predefined shifts |
| --- | ---: | ---: | ---: |
| Standard OKAPI | .35176 | .27526 | .32662 |
| Six deep chain starters | .33869 | .27248 | .32406 |
| Six deep balanced starters | **.26796** | .25811 | .31668 |
| All-provider mean | .39982 | .37813 | .38677 |
| Ranked-prefix Top-N, 384 fit labels | .35413 | .33406 | .36588 |
| Logit stack, 384 fit labels | .33265 | .22589 | .31761 |
| Ranked-prefix Top-N, 640 fit labels | .35413 | .33297 | .36582 |
| Logit stack, 640 fit labels | .33027 | **.21995** | **.31642** |

Each GP arm used 384 labels for evolution and 256 separate labels to select its final tree. The 640-label baselines fit on their union. IID and shifted tests contained 2,048 held-out examples per bank. On the focal shift all inputs came from the interior grid. The effect of grouping shrank to −.01924 (95% interval −.03173 to −.00675). Four independent data banks in each of the remaining 18 families give limited precision for individual regressions. The generator's default hidden roles were fixed across focal banks. The suite average weights each family equally and describes this registry at this budget. [Each family and split](scenario-means.csv) · [Paired intervals](paired.csv)

The balanced arm had lower point estimates than the chain arm in 13 of the 19 IID families. TwoPathways showed the largest gain (−.08864) and NativeCloseSwitch the largest loss (+.00634). The selected balanced-start winners used an average of 15.97 nodes and 4.73 distinct provider IDs across families, versus 9.28 nodes and 3.95 IDs for ordinary OKAPI. These are counts in the tree. Runtime of the base models and the effect of pruning were outside the experiment. On the focal task none of the 16 balanced-arm winners was structurally identical to one of its six inserted trees. This comparison cannot establish ancestry or rule out functional equivalence.

The one campaign ran 480 trajectories in 71.5 seconds of wall time on three local CPU workers (181.7 summed search seconds). Including verification and reporting the round took about four minutes from protocol publication. The independent verifier checked 960 pairs of deep starters, all 288,000 training evaluations by recorded counts, 2,760 score rows and all 66 paired estimates. The separate prefix check replayed 160 complete rankings and matched 400 prediction fingerprints. Each run used the same fixed native six-operator repertoire. Generated models and Brier scores are archived with per-bank and per-restart identifiers. [Scores](scores.csv.gz) · [Programs and budgets](trees.jsonl.gz) · [Top-N prefix losses](topn-prefixes.jsonl)

To reproduce from the FusionBench root, use the research environment with OKAPI checked out at 3eae0e5 as a sibling directory.

```sh
export PYTHONPATH=.:../OKAPI OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
python docs/ensemble-study/mixed-deep-init/run.py --workers 3
python docs/ensemble-study/mixed-deep-init/verify.py
python docs/ensemble-study/mixed-deep-init/check_topn.py
```

The remaining question is how many distinct providers can be removed from these same frozen winners while retaining the observed advantage on independently held-out examples.

# Which part of a six-proposal mutation budget buys coverage?

At a fixed parent and companion, proposing all six native operator families once gave a slightly better fit-selected child on fresh IID examples than drawing six families with replacement. The **gain in selected IID excess Brier was 0.00179** (95% paired interval 0.00102 to 0.00256). The probability of offering *any* truly improving child increased from 60.59% to 63.89%, a difference of 3.30 percentage points (−0.48 to +7.08). The two co-primary conditions therefore did not both pass. The one-step pilot covered 24 new data banks, three synthetic laws and eight proposal seeds per bank. The intervals use independent banks.

The largest difference came from a predeclared descriptive comparison. Keeping exactly the original random operator sequence and drawing companions afresh gave **78.30%** probability of offering an IID improvement, up 17.71 percentage points (11.31 to 24.11). Fit-selected IID gain improved by **0.00956** (0.00645 to 0.01266) over random operators with one fixed companion. All menus had six proposals and the parent could survive.

| Six proposals from the same singleton | Different families | Different companions | Any fit improvement | Any IID improvement | Fit-selected winner improves on IID | Fit-selected IID gain over parent |
|---|---:|---:|---:|---:|---:|---:|
| Uniform families with replacement, one companion | 3.94 | 1 | 53.30% | 60.59% | 51.22% | 0.01419 |
| Every family once, one companion | 6 | 1 | 57.12% | 63.89% | 54.51% | 0.01598 |
| Same random families, fresh companions | 3.94 | 3.77 | 71.18% | 78.30% | 68.23% | 0.02375 |

The operator-only menu guaranteed a FAR proposal, but this supplied little extra value on the FAR law. Averaged over fresh fits and IID tests its gain over the singleton was 0.01056 versus 0.01072 for random operators at the same companion. Varying the companions raised this to 0.02247. On the MIN law covering every family did help in point estimate, raising the gain from 0.02317 to 0.02725; it guaranteed an IID-improving proposal in this sample, and the fit selector deployed an improving child 83.33% of the time. An exploratory count after selection shows one reason operand diversity matters in this design. Both hidden signal providers, in either order, were actually offered in 23.61% of the fixed-companion menus and 75.00% of the varying-companion menus. There were four distractor providers.

## Prediction quality against inexpensive references

Mean IID excess Brier by label law. Lower is better. These comparisons share predictions, fit labels, and test observations.

| Model | FAR switch | Logistic | MIN |
|---|---:|---:|---:|
| Fixed best-fit singleton | 0.08052 | 0.05133 | 0.07977 |
| Six random families, one companion | 0.06981 | 0.04263 | 0.05661 |
| Six families once, one companion | 0.06996 | 0.04119 | 0.05253 |
| Six random families, fresh companions | 0.05805 | 0.03815 | 0.04417 |
| Arithmetic mean of all six providers | 0.09032 | 0.05765 | 0.06638 |
| Learned Top-N | 0.04006 | 0.01114 | 0.04059 |
| Probability-feature logistic stacking | 0.02488 | 0.00756 | 0.01663 |
| Logit-feature logistic stacking | 0.02257 | 0.00483 | 0.01973 |

Every one-step mutation policy remained behind regularized stacking on all three laws. On the lower-score covariate shift, the corresponding average excess risks for the random fixed, balanced fixed, and varying-companion policies were 0.03720, 0.03530, and 0.03391. These are descriptive outcomes on the three specified laws.

There were 10,368 native proposal evaluations and 1,728 menu decisions in 10.7 seconds on one CPU thread. An independent result checker recomputed every retained slot, checked equality of matched parameter draws and risks, and reproduced both primary intervals from the raw candidate records.
## What one step measures

The frozen parent is the provider with lowest Brier on 256 fit labels. Its sole root fixes the append site. The challenger is an addition of one native operator and one new provider. Inputs and labels are common within a bank; the new provider and schedule are independent of labels. The native weighted mean draws its mixing weight uniformly, and native FAR and CLOSE draw their threshold uniformly. Every arm received six actual native constructor calls and six training-objective evaluations. Repeats count in full. A fit-selected proposal replaces the parent only if its Brier improves. IID excess Brier is squared error to the simulation's known conditional probability; only auditing uses that probability. Ordinary validation/test Brier and the lower-score covariate shift are recorded separately.

The experiment concerns the birth of candidate structures at a singleton root. It says nothing direct about survival in a population, later mutations, crossover, deeper trees, or generalization to observed data. The validation and test sets played no part in choosing families, operands or winners.

## Reproduction

Use OKAPI revision `4c784f6a217675d1305d378eaeb525738c37761d`. From the FusionBench checkout run:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OKAPI_LOG_LEVEL=ERROR \
python examples/operator_coverage.py --out docs/ensemble-study/operator-coverage/results
python docs/ensemble-study/operator-coverage/verify.py
```

[Preregistered protocol](https://github.com/damiankucharski/fusionbench/blob/be0eb9b6eb3cda347753b8351e17ee6533f8eeb7/docs/ensemble-study/operator-coverage/protocol.md) · [All candidates](results/candidates.csv.gz) · [Every selected slot](results/menus.csv.gz) · [Paired contrasts](results/contrasts.csv) · [Per-law averages](results/means.csv) · [All baselines](results/baselines.csv.gz) · [Measured cost](results/cost.json)

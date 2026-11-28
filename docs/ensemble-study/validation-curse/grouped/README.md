# Can evolution lose a known optimal program before the final generation?

We generated clustered multiclass prediction banks with 24 independent training units, eight correlated annotations per unit, eight provider columns and five classes. Provider outputs are constant within each unit; labels share a draw 80% of the time. In one arm the true class probability is the arithmetic mean. In the other, a designated provider genuinely improves on the mean. These categorical banks simulate scarce independent information and correlated observations. MedMNIST and STARCOP were not used.

All methods trained on the same 192 labeled rows. Native GP started from its eight singleton programs and kept eight survivors. We scored the entire surviving population at generations 0, 4 and 16 on 2,048 independent feature vectors per bank. Given the generating probability q, the exact excess full-class Brier conditional on a feature vector is the sum of squared differences between a prediction and q. Integrating out label noise this way makes comparisons between individual trees more precise. Variation in sampled test features remains. Population minima are diagnostic and unavailable for selection. [Protocol and amendments](protocol.md)

## Round one

On the 12 mean-optimal banks the average best-in-population excess Brier improved from 0.12419 at initialization to 0.04773 at generation 16. The final empirical winner had excess 0.05816, leaving only 0.01044 between it and the best surviving program. Selecting the best survivor retrospectively therefore cannot recover the remaining 0.04773. In the signal arm the pattern reversed for the ceiling: the best survivor moved from 0.00514 to 0.01261 excess Brier, although the population median improved from 0.23732 to 0.01901. Convergence, improvement of a typical program and retention of the best rule are different endpoints. [Per-bank results](r1-banks.csv)

## Round two on fresh banks

The table gives full-class Brier against sampled test labels, averaged equally across 12 banks per condition. The conventional solver uses empirical Brier. Penalized solvers subtract one standard error of each program's paired training gain over the mean; data, starting population, final population size and generation budget are identical. The row penalty treats 192 annotations as independent. The group penalty uses the 24 underlying independent units. This diagnostic uses the native paired-fitness primitive with the ordinary affine Brier-score scaling and three-decimal solver selection. The full-suite experiment separately evaluates the regularized GP-plus-head procedure.

| Generating law | Mean | Top-N | Probability LR | Conventional GP | Row penalty | Group penalty |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Mean optimal | 0.70687 | 0.73022 | 1.08104 | 0.76929 | 0.75959 | 0.75544 |
| One informative provider | 0.70692 | 0.64586 | 0.98725 | 0.64309 | 0.65217 | 0.64738 |

On mean-optimal banks the group-aware penalty improved expected excess Brier in 9/12 draws by 0.01226 on average (paired t interval for guarded minus ordinary −0.02515 to +0.00063). It still left 0.04634 excess Brier. Even when both populations used the identical raw training-loss rule for their final pick, the group-penalized trajectory improved from 0.05860 to 0.05189. On the signal control the group penalty cost 0.00454 on average (interval −0.00088 to +0.00995). The tradeoff is visible on observed test Brier too. [Round-two bank scores](r2-banks.csv)

## Did evolution fail to construct the mean, or discard it?

We inserted an exact arithmetic-mean tree at initialization in place of the last singleton. This kept the initial population size, GP budget and all other selection mechanisms fixed. On the 12 mean-optimal banks, ordinary GP had lost all functionally mean-equivalent survivors by generation 4 (criterion: excess Brier below 10⁻¹²). The group-aware penalty retained the mean in five banks at generation 4 and one bank at generation 16. At generation 16 even the ordinary population's best remaining tree had average excess 0.01525; the group penalty reduced it to 0.00922. The arithmetic mean can be expressed natively and was explicitly present at the outset. [Seeded witness](r2-seed-mean-banks.csv)

After observing this loss we replayed the same 12 null banks with the seed and noise-free conditional expected loss on the training inputs at every evaluation. The mean survived in all 12 final populations. This attribution control shows that label noise interacting with search and survival can eliminate an optimal program even while holding size pressure and budget fixed. The control uses knowledge of the simulator and cannot be deployed on unknown real labels. [Attribution check](r2-seed-mean-population-banks.csv)

Run from the FusionBench checkout with the pinned OKAPI extra installed and a single BLAS thread per worker:
```sh
export PYTHONPATH=. BACKEND=numpy OKAPI_LOG_LEVEL=ERROR OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python docs/ensemble-study/validation-curse/grouped/run.py 1
python docs/ensemble-study/validation-curse/grouped/run.py 2
python docs/ensemble-study/validation-curse/grouped/run.py 2 --seed-mean
python docs/ensemble-study/validation-curse/grouped/run.py 2 --seed-mean --population-control
```

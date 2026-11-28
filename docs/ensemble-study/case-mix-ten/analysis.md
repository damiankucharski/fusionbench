# Reading the case-mix study

Each source bank produces one fitted method and one set of training decisions. The 30 standard families are fixed and receive equal weight. Each focal test integrates over the exact joint distribution of reports and labels. All four target worlds reuse the same fitted forecast on the same set of report vectors.

For maximum regret, subtract each world's own full-information Bayes Brier from the fitted model's expected Brier. Take the larger regret at 0% and 100% shortcut activity separately for each method on each source seed, then average or compare the paired maxima. The worst Brier endpoint omits the Bayes subtraction. It answers a different question because the two target worlds have different irreducible loss. Negative method-minus-reference differences favor the method.

| Output | Interpretation |
|---|---|
| `rXX-levels.csv`, `rXX-families.csv` | Absolute risk and per-family means. Use `metric` to distinguish ordinary Brier, worst Brier and maximum regret. |
| `rXX-contrasts.csv` | Paired risk differences, including comparisons with simple averaging, ranked Top-N, logistic stacking, unrestricted weighting and clipping. |
| `rXX-profile-pairs.csv` | Profile minus mixed on matched source seeds, both in absolute risk and relative to the unweighted native procedure. |
| `rXX-support.csv` | Observed environment/context cell coverage on each source bank, before seeing evaluation outcomes. |
| `rXX-weights.csv`, `rXX-reasons.csv` | Effective source size, weight concentration, choice of full/clipped/uniform weights and fraction of cases where a configuration changed the fit. The `changed_*` fields are counts of banks. |
| `r03-stage-pairs.csv`, `r04-stage-pairs.csv` | Exact within-bank comparisons of weighting at individual stages against ordinary native and full weighting. |

The row-level effective sample size is the squared sum of weights divided by their sum of squares. A cap bounds each raw ratio before the weights are normalized to mean one. For scene-grouped data the supplementary group count uses the same formula after aggregating weights within each source scene. Its ratio is relative to the number of distinct groups. Neither count measures target representativeness.

`changed_path` compares the ordered sequence of distinct tree keys evaluated during GP discovery. `changed_tree` compares the returned program's structural key, which can include numerical parameters. `changed_component` records selection among the fitting procedure's candidate families. `changed_forecast` checks the exact output values on the common target support. A path change need not change the selected program, and a structural change can leave its predictions unchanged.

Percentile intervals use 10,000 paired draws at 103000 plus the round index. Source seeds are resampled within fixed standard families or within a focal profile. Matched profile comparisons resample the common seed once for both profiles. The number of test rows is never treated as the number of independent source draws.

Round 9 uses empirical margins only to screen the three candidates, then ranks eligible rules by mixed maximum regret. `selection.json` stores the input hash, scores and eligible set. Round 10 verifies this lock against Git before fitting. `results/confirmation.json` gives the primary outcome and each prespecified safeguard separately. The independent verifier reconstructs paired intervals and representative fits and checks the committed rule.

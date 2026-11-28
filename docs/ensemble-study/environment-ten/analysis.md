# Reading the environment study results

The standard score is the equally weighted mean across the 30 fixed benchmark families. Focal scores refer to one specified source design and one target law. Every method sees the same source rows and target features on a given seed. The test Brier on a focal bank integrates over the exact joint input-label law. Its variability across seeds reflects source sampling, provider ordering and the resulting fit. Confidence intervals resample source seeds within each fixed family with 10,000 paired draws at 83000 plus the round index.

For *worst endpoint*, take the larger of the two Brier scores at 0% and 100% shortcut activity for each fitted method on each source seed. Then compare the two methods on that seed and average their differences. This order allows the worst endpoint to differ between methods. A negative contrast favors the method in the `method` column. A rule can improve worst-endpoint Brier and still lose at the favorable endpoint.

Each screening decision acts on one source bank. The same decision is deployed in every target world. In the role and decision-pair tables, a *shortcut drop* is the removal of the particular provider secretly assigned that role by the generator. The specialist-drop rate counts mistaken removal of the stable specialist. The reported rates divide by all source banks with a known role, including those where the rule retained all providers. The standard tasks supply proxy environment tags and lack simulator role labels. The paired source-design tables compare matched seeds and also report the difference between each design's method-minus-native effects.

| Output | Contents |
|---|---|
| `rXX-levels.csv` and `rXX-families.csv` | Absolute Brier, macro-F1 and deployed provider counts, with the 30-family breakdown |
| `rXX-contrasts.csv` | Paired Brier differences and 95% percentile intervals for each reference |
| `rXX-drops.csv` | Which planted provider was removed and how often a rule abstained |
| `rXX-drop-pairs.csv`, `rXX-drop-effects.csv` | Paired role transitions and changes in exclusion rates between screening rules |
| `rXX-design-pairs.csv` | Matched changes in exclusion frequency, Brier and gain relative to native under changes to the source design |
| `rXX-verification.json` | Reconstructed source evidence, reference fits and paired intervals, with exact target-law identity checks |

`python summarize.py 9` selects on the mixed design's worst endpoint and saves the raw CSV hash in `selection.json`. The round-10 runner checks this decision against Git before fitting. `python summarize.py 10` applies the frozen thresholds to the confirmation draws and saves `results/confirmation.json`. `python verify.py R` independently rebuilds the environment scores and paired intervals and refits the three cheap comparators on a fixed set of representative seeds. The verifier checks that removing the true shortcut makes risk invariant to its subsequent activity and that intermediate target laws give the linear interpolation of their endpoint risks.

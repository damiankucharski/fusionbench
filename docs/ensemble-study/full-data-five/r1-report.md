# What does withholding 30% of the fusion examples cost?

The full-data control ran ordinary constructive OKAPI for eight generations, using every source example for evolution and final Pareto-front selection. The second arm removed 30% before training, then used the same native return rule on the reduced fit. The third reused that exact final population, including its ordinary champion, and selected the tree with the lowest Brier on the reserved examples. This separates the change in the data available to evolution from the change in how a fixed population is ranked. Ties kept the ordinary champion first and then the first occurrence of each distinct tree. The test data were independent of all three choices. [Implementation](engine.py) · [Protocol](protocol.md)

## Change in test Brier

Negative changes are improvements. Intervals are paired 95% bootstrap intervals across data draws. The standard-v5 estimate averages the fixed 27 families with equal weight. Its interval resamples the two seeds within each family. The four digits resplits overlap, so their summary is descriptive.

| Setting | All rows | Reduced fit | Reduced fit + holdout selection | Reduced minus all | Holdout selection minus reduced |
| --- | ---: | ---: | ---: | ---: | ---: |
| equal | .5222 | .5452 | .5495 | +.0230 [+.0071, +.0412] | +.0042 [−.0046, +.0179] |
| weak | .5296 | .5433 | .5409 | +.0138 [−.0079, +.0402] | −.0025 [−.0124, +.0070] |
| strong | .4287 | .4278 | .4174 | −.0009 [−.0237, +.0204] | −.0104 [−.0295, +.0007] |
| scene16 | .1475 | .1555 | .1597 | +.0080 [−.0014, +.0179] | +.0042 [−.0014, +.0110] |
| scene128 | .1469 | .1510 | .1517 | +.0041 [−.0056, +.0119] | +.0007 [−.0042, +.0050] |
| standard | .3981 | .4000 | .3996 | +.0019 [−.0019, +.0058] | −.0004 [−.0017, +.0008] |
| digits_449 | .0348 | .0347 | .0346 | −.0001 | −.0001 |

With 16 scenes of eight rows each, holding out five complete scenes increased test Brier by **.01226** versus the full-data procedure (95% interval .00270 to .02157). Test Brier rose in 10 of 12 draws. Withholding rows while keeping the ordinary final selection accounted for .00802 of the observed mean difference. Changing selection on the same reduced fit added .00424. The individual intervals include zero. In eight of 12 draws the held-out labels selected a different structure. This lowered Brier on the five selection scenes by .02844 on average, including draws where the winner stayed the same. The mean improvement on the selection sample failed to transfer to fresh scenes. [Paired differences](results/round1-contrasts.csv) · [Terminal population decisions](results/round1-fits.jsonl)

The withheld set contained 40 rows. On average, 3.92 of them had a positive label and those positives occupied 2.17 of the five scenes. One of the 12 withheld sets had no positives. Each fit and test scene came from the same generator. Test scenes were distinct from every fusion-training scene. In the 128-scene variant, training used 128 rows distributed across 128 scenes and reserving 39 complete scenes cost .00479 Brier after terminal reselection (−.00558 to +.01450). This study does not identify how much of the combined split cost would recur on clinical or satellite images. [Scene construction and event counts](scene-cohort.md) · [Seeded split implementation](engine.py)

When all providers were equally competent in the finite-population law, using only 89 of the 128 rows increased Brier by .02303. Restricting to 89 rows had an unresolved cost under weak specialization. With strong specialization, selecting a champion on the holdout improved test Brier by .01039 on average relative to the reduced-data champion, with the interval extending up to +.00072. Across the full standard suite, reduced fit followed by holdout selection improved seven family means, worsened ten and tied ten compared with all-row OKAPI. The largest gain was on compositional_switch (−.03155) and the largest regression was on compound (+.04277). Each family had only two independent draws. [All 27 families](results/round1-families.csv)

## Secondary measures on rare events

The native search optimized Brier in every arm. Higher macro-F1 and positive-class average precision (AP) are better.

| Setting and metric | All rows | Reduced fit | Reduced fit + holdout selection | Reduced minus all | Holdout selection minus reduced |
| --- | ---: | ---: | ---: | ---: | ---: |
| scene16 macro_f1 | .6500 | .6567 | .6639 | +.0067 [−.0089, +.0254] | +.0071 [−.0097, +.0257] |
| scene16 average_precision | .4475 | .4303 | .4220 | −.0172 [−.0409, +.0084] | −.0083 [−.0242, +.0050] |
| scene128 macro_f1 | .6439 | .6226 | .6326 | −.0213 [−.0378, −.0050] | +.0100 [−.0024, +.0254] |
| scene128 average_precision | .4531 | .4233 | .4213 | −.0298 [−.0575, −.0058] | −.0020 [−.0237, +.0207] |

On the 16-scene design, holdout selection after reduced-data evolution improved macro-F1 relative to all-row OKAPI by .01382 (.00047 to .02666), even as Brier worsened. AP fell by .02556 (−.06024 to +.00703). Brier measures probability accuracy, macro-F1 scores the resulting class decisions and AP measures the positive-class ranking. Their directions can differ on the same forecasts. With 128 independent training scenes, removing 39 rows directly reduced AP by .02978 (−.05751 to −.00580).

## Where the usual comparisons stand

On the 16-scene design, arithmetic averaging had the best Brier (.13908), highest macro-F1 (.70447) and highest AP (.54217) among all seven arms. Ranked Top-N, all-row OKAPI and logistic stacking had Brier .15174, .14748 and .16010 respectively. With 128 training scenes, arithmetic averaging attained exactly the same scores because it uses no training labels and the two conditions share test scenes and provider predictions. All-row OKAPI had Brier .14688, ranked Top-N .14911 and logistic stacking .15907. [Absolute scores and references](results/round1-levels.csv)

For the 27-family mean, all-row OKAPI reached .39810 versus .40403 for arithmetic averaging, .38645 for ranked Top-N, .29076 for logistic stacking and .26037 for the historical eight-tree regularized pipeline. The historical method has a different search space, auxiliary logit heads and a fixed row-level internal split. With 16 source scenes, that internal split placed rows from all 16 on both sides on each of the 12 draws. Its independent test scenes were disjoint. R1's controlled holdout-by-scene comparison kept entire scenes together.

This round recorded 826 method scores across 118 banks in 120.6 seconds using one numerical thread. The search seed was fixed and intervals describe variation over source and evaluation banks as specified in the protocol. [Runtime](results/round1-run.json) · [Individual scores](results/round1-scores.csv)

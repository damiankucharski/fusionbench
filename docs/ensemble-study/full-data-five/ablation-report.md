# What changed the selected native tree?

Giving ordinary GP the arithmetic mean as a starting tree helped even on banks where it later selected a different forecast. Penalizing uncertain improvements had a different effect. It frequently returned the exact arithmetic mean on the small matched banks.

Each comparison below changes one part of its own round. Changes are mean held-out full-class Brier, with negative values favoring the intervention. Each controlled column has 12 paired banks. The standard column gives equal weight to the 27 families, with two banks per family. Digits uses four overlapping partitions. [Paired intervals and all controls](results/matched-ablations.csv)

| Intervention and matched control | Weak | Strong | 16 fit scenes | 128 fit scenes | Standard-v5 | Digits 449 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Round 2, starting mean vs duplicate starting provider | −.02845 | −.01061 | +.00794 | −.00122 | −.05417 | −.00173 |
| Round 3, squared-distance penalty vs zero penalty with the same scaling | −.00304 | +.00406 | −.00293 | +.00051 | +.00171 | +.00002 |
| Round 4, paired margin vs zero margin with the same score mapping | −.00533 | +.01214 | −.00266 | −.00221 | +.00385 | +.00037 |

## Giving search an additional starting tree

In round 2 both policies received one additional initial slot and the same population quota. The control duplicated the first initial provider. Native structural deduplication can remove that copy during evolution. Putting the all-provider mean in the slot instead won on all 12 weak-specialization draws, even though the final predictions of both arms differed from the mean on every draw. An initial mean can change the search trajectory without being the deployed solution. We did not trace ancestry, so the results do not establish which winning programs descended from it. On the 16-scene control, the same change raised test Brier by .00794 (95% paired interval −.00090 to +.01816). The amount of independent scene information matters when interpreting the focal gain.

The additional slot alone also changed the search. Across all 27 families the duplicate-slot control worsened Brier relative to ordinary GP by .01194. The gain of .05417 against that duplicate is therefore larger than the gain against ordinary GP.

## Changing fitness at every generation

In round 3 the squared-distance penalty modestly improved weak-signal prediction versus its same-scale control (−.00304, interval −.00637 to −.00005). The strong-signal cost was +.00406 (+.00066 to +.00717). Each deployed tree differed from the arithmetic mean on all 12 weak cases. Three-decimal native selection can change when scores are rescaled: even the same objective with a different scale changed the suite result by −.00374 versus the ordinary seeded fit.

In round 4 the paired margin selected the exact mean on **8/12 weak**, **6/12 with 16 scenes**, **3/12 with 128 scenes**, and **0/12 strong** banks. The matched zero-margin objective selected the mean on none of these banks. Weak Brier improved by .00533 (interval .00159 to .00906) and 16-scene Brier improved by .00266 (.00003 to .00542). Strong Brier worsened by .01214 (.00398 to .02409) even though all its deployed trees differed from the mean. Across the full suite the margin raised Brier by .00385 (.00164 to .00606). A margin alters the complete evolutionary fitness ordering, including which non-mean programs can win.

Mean equivalence was checked over all class probabilities on both fit and audit inputs with absolute tolerance 1e−12. On these banks the count also matched the number of deployed trees with the exact starting mean's structural key. The seed's survival in intermediate generations remains unmeasured. [Decisions and number of providers](results/mean-deployment.csv) · [Reproduce these summaries](ablation.py)

# Recalibrating the output of one fitted tree

Round 3 asked whether the probability scale explained part of the remaining Brier loss. We first ran the previous full-data GP with its mean seed and distance penalty, or the corresponding GP with a compact seed. We then fixed its selected tree and tried nine inverse temperatures from 0.5 to 2. Each acts on the final class distribution. A multiplier above one sharpens the distribution; a multiplier below one softens it. At one the original fitted forecast is returned exactly. [Source fitting and deployment](calibrate_gp.py)

For binary probabilities strictly inside the native clipping range, the calibrated probability is `sigmoid(beta × logit(p))`. A positive multiplier preserves ordering and the 0.5 decision threshold in exact arithmetic. It cannot correct a wrong ranking or learn an intercept for positive prevalence. The native operator clips the input at 1e−6 and 1−1e−6; floating-point ties can also affect ranking or argmax on discrete score grids. In the multiclass case the same scalar raises each clipped class probability to the power beta and renormalizes. Argmax is preserved in exact arithmetic away from clipping and ties. One operator suffices to apply the selected change to the final forecast. It reads the same providers as the uncalibrated program.

All fit examples participated in both evolution and fitting the multiplier. The penalized method minimized source full-class Brier plus 0.25 × log2(beta)² / sqrt(U), where U is the number of fit scenes or the number of available examples if no scene IDs were supplied. The sources with 16 and 128 independent scenes therefore impose different costs on extreme temperatures at the same row count. The unpenalized control used exactly the same grid. The calibrated arithmetic-mean control separates a shared probability-scale effect from any benefit specific to evolved trees. [Protocol](protocol.md)

## Round 3

The table gives observed Brier on the same held-out predictions before and after calibration. The eight related configurations within a factorial seed are averaged first. The suite gives equal weight to its 30 fixed families. There were eight factorial seeds and two suite seeds per family.

| Fitted program | Factorial | All 30 families | Earlier 16-scene problem |
| --- | ---: | ---: | ---: |
| Tree with all-provider mean seed and penalty | .14545 | .33299 | .14349 |
| Same tree, penalized temperature | .14440 | .31759 | .14138 |
| Same tree, unpenalized temperature | .14460 | .31961 | .13730 |
| Compact-seed GP | .14680 | .34539 | .15035 |
| Same compact-seed GP, penalized temperature | .14582 | .33352 | .14949 |
| Arithmetic mean | .14978 | .37219 | .14127 |
| Same mean, penalized temperature | .14654 | .35312 | .13797 |
| Probability logistic stacking | .17288 | .27050 | .16658 |

Compared with its own uncalibrated tree, penalized calibration of the all-provider mean-seeded GP improved crossed-scenario Brier by **.00105** (95% paired interval .00045 to .00170) and suite Brier by **.01540** (.01390 to .01689). On the eight weak-specialization fits it worsened risk by .00140 (interval for the increase .00000 to .00366). Penalized and unpenalized temperature gave similar average results in the factorial (penalized minus unpenalized −.00020, −.00246 to +.00232). On the previous 16-scene task the unpenalized grid did better by .00408 (.00258 to .00535). These are exploratory comparisons across fresh training banks. [All contrasts](results/r3-contrasts.csv)

The simple average also benefited from one scalar, by .00324 (.00221 to .00406) on the factorial and .01907 (.01781 to .02037) on the suite. Its calibrated loss was lower than LR on the factorial and higher over the complete suite. On this factorial the generator uses 1.3 times the latent signal for ordinary provider logits and 1.8 times the signal for the event logit. The scale mismatch is one plausible source of the benefit. Which slope works best after pooling noisy providers also depends on provider noise, correlation and specialization. [Scenario source](../../../fusionbench/scenarios/grouped_rare_event.py) · [Absolute scores](results/r3-levels.csv)

The penalized all-provider-seeded tree chose identity on 41 of 64 factorial fits and sharpened the forecast in the other 23. The calibrated simple mean kept identity in 20 and sharpened in 44. Binary AP and macro-F1 were unchanged on all 64 factorial fits for each calibrated method. Every calibrated tree in the round used exactly the same number of providers as its parent. When its multiplier was one, its fit and test prediction hashes were identical to the parent's hashes. On discrete finite-support controls, occasional last-bit differences and clipping changed some tied predictions; the maximum absolute AP change for the penalized GP was .00046 across the weak and strong controls. [Choices, classification and ranking](results/r3-temperature.csv) · [Reproduce the diagnostic](temperature_summaries.py)

## Does the calibrated tree improve on calibrating the arithmetic mean?

After selecting the temperature-calibrated GP for confirmation, we registered a separate control before its confirmation scores were available. We fitted the same penalized scalar to the arithmetic mean on all 408 assigned banks. All 816 fit and test hashes for the uncalibrated mean matched their original records. This comparison was descriptive and played no role in selection or in the four primary confirmation checks. [Control protocol](calibration-control-protocol.md) · [Separate replay](mean_control.py)

| Confirmation cohort | Arithmetic mean | Calibrated mean | Confirmed calibrated GP | Logistic stacking |
| --- | ---: | ---: | ---: | ---: |
| Eight crossed settings, observed Brier | .14725 | .14409 | .14334 | .16499 |
| All 30 families | .37593 | .35671 | .32150 | .27063 |
| Original 27 families | .40212 | .38118 | .34272 | .28587 |
| Earlier 16-scene control | .13888 | .13379 | .14076 | .16976 |
| Earlier 128-scene control | .13888 | .13188 | .13977 | .15334 |

On the factorial, the difference in **conditional expected Brier** between the calibrated mean and calibrated GP was +.00077 (95% paired interval −.00149 to +.00262). The observed-label difference was +.00075 (−.00169 to +.00272). Calibrating the mean reduced its own conditional expected Brier by .00304 (.00241 to .00360); the calibrated mean beat stacking by .02080 (.01228 to .03016). The improvement over the raw mean therefore includes a probability-scale effect available without GP. The two calibrated approaches produced different test predictions and can still differ by mechanism. [Control levels](results/calibration-control-levels.csv) · [Paired contrasts by setting](results/calibration-control-contrasts.csv)

The calibrated GP achieved lower suite Brier than the calibrated mean by .03521 (.03117 to .03923) over all 30 families and by .03846 (.03399 to .04292) on the original 27. The order reversed in the 16-scene control, where the calibrated mean scored .00697 lower (.00423 to .01041). These comparisons describe the joint effect of selecting a tree and fitting its temperature; they do not separate GP's structural choices from changes in the optimal calibration of the selected output.

On the 192 factorial fits, calibrated GP chose identity 106 times and sharpened 86 times. Calibrated mean chose identity 74 times, sharpened 117 times and softened once. Neither calibration changed binary AP or predicted classes on any factorial fit. Provider counts were identical to the corresponding raw programs. [All control fits and decision changes](results/calibration-control-scores.csv) · [Confirmed GP temperatures](results/r4-temperature.csv)

Reproduce these descriptive checks from the same FusionBench and OKAPI checkouts with the environment used by the main runner:

```sh
study=docs/ensemble-study/mechanism-four
python "$study/temperature_summaries.py" 3
python "$study/temperature_summaries.py" 4
python "$study/mean_control.py"
```

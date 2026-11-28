# When label noise and provider noise have opposite effects

With a disappearing shortcut, corrupting 20% of the **training labels** reduced regularized OKAPI’s exact clean-target Brier by **0.1192** (paired 95% interval −0.1735 to −0.0633). Independently flipping 20% of the **generalist and specialist votes** during training increased the same loss by **0.0530** (0.0222 to 0.0808). The gain from label noise depended on the target. With the shortcut fully available at test, label noise in training increased clean-target Brier by 0.0587. [Frozen protocol](protocol.md) · [Paired contrasts](results/noise-contrasts.csv)

All fits used 512 source rows with a 5% specialist region, a routing cue that identifies the region and a shortcut that predicts the clean label with 90% accuracy. At test the region prevalence was 50%. Each training state used the same underlying labels and random draws within each of 16 banks. Each fitted model supplied one forecast on each of the 16 possible vote vectors. We integrated those same forecasts under nine target laws, crossing shortcut availability with clean reports and labels, 20% provider flips, or 20% label flips. The provider flips act independently on the two core votes. The shortcut and routing cue retain their original reports. Label flips occur after all four reports. Intervals use 10,000 paired resamples of the 16 training banks at fixed data laws and fixed native search seed zero. [Fitting records](results/noise-fits.csv) · [Forecasts](results/noise-predictions.csv)

## When the shortcut disappears

Differences below subtract the clean-trained model’s full two-class Brier on the same clean target. Negative values favor the modified training condition.

| Method | Clean training, target Brier | Training labels flipped, difference | Training provider votes flipped, difference |
|---|---:|---:|---:|
| Arithmetic mean | 0.4282 | 0.0000 | 0.0000 |
| Fit-Brier Top-N | 0.7991 | −0.3176 | +0.0209 |
| Probability logistic stacking | 0.6471 | −0.1147 | +0.0695 |
| Regularized OKAPI | 0.6526 | −0.1192 | +0.0530 |

The clean-target Bayes Brier was 0.2322. With training labels flipped, regularized OKAPI reached 0.5335 and still trailed the arithmetic mean. The improvement occurred in both subgroups. Its common-region Brier fell from 0.6738 to 0.5416 and its specialist-region Brier from 0.6315 to 0.5253. Training with corrupted provider votes instead increased those losses to 0.7321 and 0.6791. Regularized OKAPI’s average empirical training Brier and exact same-source Brier were 0.1680 and 0.1698 under clean training, 0.3805 and 0.3928 under noisy training labels, and 0.1799 and 0.1803 under noisy provider reports. Each of those source scores concerns its own observed outcome. [Population and subgroup scores](results/noise-scores.csv) · [Empirical training scores](results/noise-fits.csv)

## What changes as the shortcut survives

| Target shortcut accuracy | Regularized OKAPI, clean training and clean target | Difference from training-label flips (95% interval) | Difference from training-provider flips (95% interval) |
|---|---:|---:|---:|
| 50% | 0.6526 | −0.1192 [−0.1735, −0.0633] | +0.0530 [+0.0222, +0.0808] |
| 60% | 0.5313 | −0.0747 [−0.1171, −0.0300] | +0.0413 [+0.0168, +0.0633] |
| 90% | 0.1672 | +0.0587 [+0.0388, +0.0897] | +0.0063 [−0.00003, +0.0122] |

The frozen forecasts reveal how much each model used the shortcut. For every setting of the other three report signs, we switched only the shortcut sign and measured the absolute change in the probability regularized OKAPI assigned to class one. Averaging equally across the eight sign settings and 16 banks gave 0.668 after clean training, 0.419 after training-label flips, and 0.707 after training-provider flips. Top-N selected a single provider on 15 of 16 clean fits and all 16 provider-noisy fits. It selected at least two on all 16 label-noisy fits. These diagnostics describe this source shortcut and these training rules. [Frozen forecasts](results/noise-predictions.csv) · [Provider counts](results/noise-fits.csv)

## Changing noise only at evaluation

Keep the shortcut absent on target and freeze the clean-trained regularized model. Flipping target labels raised mean Brier from 0.6526 to 0.6961. Flipping target provider reports raised it to 0.6960. Their Bayes floors were 0.4036 and 0.3930, so their observed-label excess risks were 0.2925 and 0.3030. Against the underlying clean outcome the label-flip arm still had excess 0.4204 because its input reports, forecasts and clean outcome law were all unchanged. [Population scores](results/noise-scores.csv)

The corresponding **training mismatch** held target noise fixed and compared clean fitting with fitting under the same type of noise as the target. With target labels flipped, clean fitting increased regularized OKAPI’s target Brier by 0.1253 [0.0799, 0.1682] relative to fitting on flipped labels. With target provider votes flipped, clean fitting reduced target Brier by 0.0366 [0.0175, 0.0538] relative to fitting on flipped provider votes. Region prevalence and shortcut accuracy still differed between source and target in both comparisons. [Paired contrasts](results/noise-contrasts.csv)

For a fixed scalar forecast $p(X)$, let $L_0$ be full two-class Brier against clean labels and $L_e$ the Brier after independently flipping the label with probability $e$. Then

$$
q_e(X)=e+(1-2e)q_0(X),\qquad
L_e=(1-2e)L_0+2e\,\mathbb E[p(X)^2+(1-p(X))^2].
$$

The forecast-dependent term can reverse comparisons. With the shortcut fully present and the same clean-trained forecasts, probability stacking beat the arithmetic mean under clean labels (0.1617 versus 0.2872). Flipping 20% of target labels put the arithmetic mean ahead (0.3987 versus 0.4315), with the ordering reversing in all 16 banks. Both identities held within $2.3\times10^{-16}$ over every scored source, target and subgroup. [Pairwise orderings](results/noise-rankings.csv) · [Numerical checks](results/noise-manifest.json)

## Reproduction

From the FusionBench checkout, activate the study environment with the companion OKAPI research revision on the Python path and run the following command. This regenerates the six noise result files, including a runtime in the manifest. [Runner](noise.py) · [OKAPI revision](https://github.com/damiankucharski/OKAPI/commit/92f0e96d7ba29f047134ffb3640f444fff5adc7a)

```sh
BACKEND=numpy LOGURU_LEVEL=ERROR OKAPI_LOG_LEVEL=ERROR python docs/ensemble-study/transfer-factorial/noise.py
```

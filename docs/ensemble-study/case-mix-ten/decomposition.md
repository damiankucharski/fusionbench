# Separating systematic error from variation across source draws

On the mixed source, full case-mix weighting moved the mean trained forecast closer to the active shortcut world's Bayes forecast. Its squared bias fell by 0.03016 there, while variation across source draws increased by 0.00501. When the shortcut was inactive, both components increased. The benefit depends on the direction in which weighting moves the learned forecast.

| Source | Target | Weights | Change in squared bias | Change in variance | Change in mean Brier excess |
| --- | --- | --- | ---: | ---: | ---: |
| Mixed, 512 examples | Inactive | Full | +0.01355 | +0.00715 | +0.02071 |
| Mixed, 512 examples | Active | Full | −0.03016 | +0.00501 | −0.02515 |
| Mixed, 128 examples | Inactive | Full | +0.02392 | +0.01297 | +0.03688 |
| Mixed, 128 examples | Inactive | Capped at 2 | +0.01728 | +0.00591 | +0.02319 |
| Extreme case mix | Specialist common | Full | −0.00190 | +0.01568 | +0.01378 |
| Extreme case mix | Specialist common | Capped at 2 | −0.00522 | −0.00059 | −0.00581 |

The extreme source had region prevalences of 3.5% and 96.5% at its two sites. The sharpest contrast in variability occurred when the target shortcut was inactive. Full weighting increased the variance component by 0.02181 (paired 95% interval 0.01246 to 0.02862). Capping weights at two gave a change of −0.00030 (−0.00552 to 0.00583). Capping curtailed this cost of sparse overlap. Each table entry is relative to ordinary native OKAPI on the same source draws. [All 96 estimates and paired intervals](results/r10-decomposition.csv) · [Checks against archived forecasts and scores](results/r10-decomposition.json)

## Calculation

For each source profile, compare the 20 independently trained forecasts on the exact same 16 binary vote patterns. The roles of the four providers are permuted before training. The audit restores the role order and checks the target joint probabilities against the generator's canonical enumeration. It verifies every archived forecast hash and reconstructs each bank's expected Brier excess before averaging. No additional model is fitted. [Calculation](decompose.py)

Let $p_i(x)$ be the two-class forecast after training on source draw $i$, let $\bar p(x)$ be its mean over draws, and let $q(x)$ and $m(x)$ be the Bayes class vector and probability mass in one fixed target world. The mean excess above that world's exact Bayes Brier is

$$
\sum_x m(x)\frac1n\sum_i\|p_i(x)-q(x)\|^2
=\underbrace{\sum_x m(x)\|\bar p(x)-q(x)\|^2}_{\text{squared bias of mean forecast}}
+\underbrace{\sum_x m(x)\frac1n\sum_i\|p_i(x)-\bar p(x)\|^2}_{\text{variation across fitted forecasts}} .
$$

Both components describe these realized 20 forecasts. Dividing by $n$ makes the decomposition exact. With finitely many fits, the squared bias of their mean also contains sampling error; the table gives no bias-corrected estimate for infinitely many source draws. Variation includes the sampled source rows and provider ordering. The GP seed is fixed throughout, so the study does not estimate variability over optimization seeds.

Each reported change subtracts the corresponding component of native OKAPI on the same source draws and target. Percentile intervals use 10,000 bootstrap resamples of whole, paired source draws and recompute the mean forecast in every resample. These exploratory intervals describe fixed target worlds. The registered endpoint first takes each fit's larger regret across two worlds. This nonlinear maximum cannot be recovered by adding the two components of either world.

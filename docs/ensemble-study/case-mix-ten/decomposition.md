# Separating systematic error from variation across source draws

For each source profile, compare the 20 independently trained forecasts on the exact same 16 binary vote patterns. The roles of the four providers are permuted before training. The audit restores the role order and checks the target joint probabilities against the generator's canonical enumeration. It verifies every archived forecast hash and reconstructs each bank's expected Brier excess before averaging. No additional model is fitted. [Calculation](decompose.py)

Let $p_i(x)$ be the two-class forecast after training on source draw $i$, let $\bar p(x)$ be its mean over draws, and let $q(x)$ and $m(x)$ be the Bayes class vector and probability mass in one fixed target world. The mean excess above that world's exact Bayes Brier is

$$
\sum_x m(x)\frac1n\sum_i\|p_i(x)-q(x)\|^2
=\underbrace{\sum_x m(x)\|\bar p(x)-q(x)\|^2}_{\text{squared bias of mean forecast}}
+\underbrace{\sum_x m(x)\frac1n\sum_i\|p_i(x)-\bar p(x)\|^2}_{\text{variation across fitted forecasts}} .
$$

Both components describe these realized 20 forecasts. Dividing by $n$ makes the decomposition exact. With finitely many fits, the squared bias of their mean also contains sampling error; the table gives no bias-corrected estimate for infinitely many source draws. Variation includes the sampled source rows and provider ordering. The GP seed is fixed throughout, so the study does not estimate variability over optimization seeds.

Each reported change subtracts the corresponding component of native OKAPI on the same source draws and target. Percentile intervals use 10,000 bootstrap resamples of whole, paired source draws and recompute the mean forecast in every resample. These exploratory intervals describe fixed target worlds. The registered endpoint first takes each fit's larger regret across two worlds. This nonlinear maximum cannot be recovered by adding the two components of either world.

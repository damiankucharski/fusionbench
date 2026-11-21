# When a calibrated source needs a negative fusion weight

Both binary providers are exact marginal posteriors. The stronger provider has population accuracy 84.1%, the weaker one 69.1%. When correlation between their errors conditional on the class reaches 0.85, the weaker provider should receive a negative coefficient in a logit pool. At fixed output from the stronger provider, an increase in the weaker provider's logit can be evidence that the two shared an unusually large noise realization. A signed stacker learned this direction in all 24 banks. On fresh IID rows it achieved positive-class Brier 0.08481 against 0.11304 for the same fit restricted to nonnegative effective slopes. The paired difference was −0.02823 (95% t interval −0.02897 to −0.02749). The archived [protocol](protocol.md) predates the fits.

## A population bound for monotone fusion

Let the two observed logits be Z. Given the binary class, Z is Gaussian with mean +d/2 or −d/2 for d=(4,1), equal class priors and common covariance with diagonal (4,1) and off-diagonal 2ρ. Each marginal log likelihood ratio is exactly its observed logit. The joint log likelihood ratio is wᵀZ with w=Σ⁻¹d. The coefficient on the weaker logit is (1−2ρ)/(1−ρ²). At ρ=0 it is one, at 0.5 it is zero, and at 0.85 it is −2.523.

For ρ>0.5, fix the stronger source. The true joint posterior η decreases with the weaker source. Any predictor g nondecreasing in that same input has conditional covariance with η at most zero, since oppositely ordered functions of the same random variable have nonpositive covariance. Write c for the strong source's calibrated probability, which equals the conditional mean of η. Relative to c, the conditional Brier loss difference is E[(g−c)²] − 2 Cov(g,η), hence is nonnegative. Selecting the stronger source attains the bound. This covers positive-weight logit pools even with an intercept and optimal tuning. OKAPI's full palette also contains nonmonotone operators, so the bound applies to a restricted tree family. The weight formula is a special case of Kahn’s Gaussian aggregation rule. The benchmark quantifies the price of the monotonicity restriction and of carrying the fitted dependence across domains. [Kahn, UAI 2004, equations 5 and 10–12](https://arxiv.org/pdf/1207.4144)

## What happens when the dependence changes?

Twenty-four independent banks, 384 labeled fit rows per bank, 256 validation rows, and 8,192 test rows per domain. IID and shift predictions use the same frozen fits and randomized column order. The shift changes only the class-conditional correlation. All learned methods see the same labeled examples. Lower Brier is better. Rows report means over banks.

| Method | IID, ρ=0 | IID, ρ=0.5 | IID, ρ=0.85 | Fit at 0.85, test at 0 |
| --- | ---: | ---: | ---: | ---: |
| All-provider mean | .12250 | .13397 | .14241 | .12275 |
| Ranked-prefix Top-N | .11274 | .11233 | .11233 | .11254 |
| Stacking, probabilities | .09724 | .11482 | .09234 | .17673 |
| Stacking, logits | .09538 | .11348 | .08481 | .20536 |
| Same logit fit, nonnegative slopes | .09538 | .11336 | .11304 | .11327 |
| Strong source, identity known | .11233 | .11233 | .11233 | .11254 |
| True posterior for test law | .09443 | .11233 | .08293 | .09471 |

The two logit fits use the same standardized binary class columns, an unpenalized intercept and the same C=1 penalty on all four coefficients. Nonnegative effective slopes are enforced by signs on the complementary feature pairs. This leaves the minimum ridge penalty at every feasible effective slope unchanged. Both stackers optimize penalized log loss and are scored on Brier. At ρ=0 the two logit arms differ by less than 0.000001 in mean IID Brier. At ρ=0.85 the fitted signed coefficient on the weak source averaged −1.920. All 24 constrained fits put it at zero. After shifting 0.85 to 0, retaining the signed coefficients *increased* Brier by 0.09209 vs the constrained fit (descriptive 95% paired interval +0.08662 to +0.09756). There was no target-domain tuning.

The current OKAPI weighted-logit search initializes positive weights and clips its mutations at zero. Direct constrained parameter fitting isolates that sign restriction. The full evolutionary search is evaluated in separate studies. The bound also explains why more generations within that restriction cannot close this IID gap. The next comparison is a signed OKAPI node on this generator, followed by testing whether unlabeled joint predictions can detect when the subtraction ceases to help. An extension should include shifts in class priors and individual provider calibration to challenge such a detector.

## Reproduction

From the FusionBench root, run the following on CPU.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m fusionbench.eval.calibrated_suppressor --out /tmp/suppressor
python -m pytest -q tests/test_calibrated_suppressor.py tests/test_oracle_and_reliability.py
fusionbench generate --scenarios calibrated_suppressor --n 384 --seed 113 --out /tmp/suppressor-data
```

The full numerical run took 1.71 seconds. Fresh-process replay of bank 41017 reproduced all 84 loss records and all 15 fit records apart from wall time, within CSV precision. Unit checks compare the scenario posterior with a separate Gaussian density calculation and test the change of sign. [Registered scenario](../../../fusionbench/scenarios/correlated.py) · [Experiment](../../../fusionbench/eval/calibrated_suppressor.py) · [Per-bank losses](results/scores.csv) · [Fit coefficients, timing and selected N](results/fits.csv) · [Original OKAPI weighted-logit implementation](https://github.com/damiankucharski/OKAPI/blob/248c1dd/okapi/node.py#L666)

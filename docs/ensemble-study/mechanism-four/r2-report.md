# A smaller starting tree for native evolution

A compact pool can imitate the arithmetic mean without using any labels. Begin with an empty set of providers. Try adding each remaining provider and keep the addition whose equal-weight average best matches the full mean on the source rows. Stop at the ceiling of the square root of the provider count, using at least two when available. An eight-provider task gets three operands. A second policy draws the same number of providers uniformly without replacement with seed 421. Both trees order their selected operands by source column. [Native implementation](https://github.com/damiankucharski/OKAPI/blob/3d1358e39bf2cce543d97bb710f4fdd4c21154ec/okapi/sparse_mean.py) · [Experiment adapter](sparse_gp.py)

The head-to-head GP comparison replaces only the initial mean tree in the previous full-data search. The training penalty in all three arms uses the full provider mean on all source rows. Each final program can deploy with its own operands; evolution can add providers that the starting tree omitted. The random seed fixes one subset for each pool size. Its GP result describes that draw rather than an average over all possible starting subsets.

## Why matching the full mean can change true loss

Let $a$ and $s$ denote the full and compact means on a given audit row, and let $q$ be the generator's true label distribution there. The conditional full-class Brier difference is
$$
R(s)-R(a)=\|s-a\|^2+2\langle s-a,a-q\rangle.
$$
The first term measures the cost of departing from the mean. The second measures the direction of that departure relative to the error already in the mean. A compact tree can improve a biased mean if its change points far enough toward the true probability. All selection here used source predictions. Known $q$ entered only this retrospective interpretation of the same round-2 forecasts.

A uniformly drawn subset of $k$ distinct providers has expected forecast $a$. Its expected second term therefore vanishes. Its expected loss above $a$ equals the full pool's average squared distance to $a$ times $(M-k)/(k(M-1))$, where $M$ is the total number of providers. This identity gives the average standalone loss over all 56 triples of eight providers without training more models. The identity has no corresponding guarantee for GP descendants. [Decomposition and forecast-hash replay](sparse_geometry.py)

### What survived outside the source bank?

For a typical three-provider subset the exact expected Brier cost relative to using all eight was .00595 across the factorial. Greedy selection reduced that cost to .00270 (95% paired interval .00080 to .00415). The distance term was .00543 and the alignment term was −.00273. Hence the chosen subset generally moved away from the full mean in a direction that corrected some of its error. With equally noisy and independent providers this benefit disappeared: the greedy subset cost .00844 versus all eight and .00013 versus the exact uniform-subset expectation in the setting without a specialist. We replayed all 128 compact-pool forecasts on the existing evaluation banks and reproduced their saved hashes. These intervals use eight independent data draws, preserving all eight matched settings of each draw. [Bankwise decomposition](results/r2-sparse-geometry.csv) · [Paired conditional summaries](results/r2-sparse-summary.csv)

The *particular* random control for eight providers selected indices 0, 1 and 2. In the high-heterogeneity cells these were the three lowest-noise providers. The fixed subset always missed the specialist (index 3). Its mean conditional Brier beat the full eight-provider average by .00303 across the factorial, despite a mean squared distance of .00605. Its larger correction term (−.00909) explains the apparent advantage. A different fixed choice could give a different answer. The true uniform-subset expectation, integrated over all triples, was worse than greedy selection by .00325 (.00186 to .00501).

## What survived evolution?

| Round 2 setting | Full-mean seed | Greedy subset seed | Fixed random seed | Greedy minus full-mean seed |
| --- | ---: | ---: | ---: | ---: |
| Factorial, conditional Brier | .14977 | .15033 | .15066 | +.00056 [−.00243, +.00285] |
| Factorial, deployed provider count | 3.02 | 2.30 | 2.30 | −.72 [−1.16, −.25] |
| All 30 scenarios, observed Brier | .33657 | .34364 | .33741 | +.00706 [+.00199, +.01211] |
| All 30 scenarios, deployed provider count | 4.23 | 2.85 | 2.87 | −1.38 [−1.67, −1.10] |

On the historical 27 families alone the Brier change was +.00819. Across all 30 family means, six improved, five tied and 19 worsened. Largest changes went in different directions on parity (−.14994) and needle (+.23199). The compact policy also cost .00630 observed Brier on the previous 16-scene task and .01830 on weak specialization. It failed the registered standard-panel and 16-scene screening limits. A smaller initial tree reduced average deployment cost in this experiment but provided a weaker starting point for several tasks. [Matched comparisons](results/r2-contrasts.csv) · [All scenarios and cheap controls](results/r2-families.csv)

# Which empirical lattice winners can generalize?

This exploratory analysis reuses the frozen 72 data cells and their recorded searches after the focal comparison. It asks whether choosing a different tree tied at the minimum training Brier could recover the true law. On both eight-provider lattice tasks, all 12 banks strictly favored other predictions over the true conditional law on the 256 training labels. On both four-provider tasks, that law was the unique empirical minimizer on all 12 banks. [Focal protocol](protocol.md) · [Per-bank cut results](posthoc-cut.csv.gz)

## Empirical minimizers under the stated decimal levels

The table averages 12 banks per row. Lower and upper denote the pointwise least and greatest empirical MIN/MAX minimizers. The midpoint averages their probabilities on every specimen and is a two-model ensemble. Fit Brier is the same for both endpoints. The additive conditional mean lies outside the MIN/MAX family. All losses use the positive channel.

| Law | Providers | Banks with distinct endpoints | True law is an empirical minimizer | Fit Brier endpoints / midpoint | IID Brier lower / upper / midpoint | Shifted Brier lower / upper / midpoint |
|---|---:|---:|---:|---|---|---|
| Branch | 4 | 0/12 | 12/12 | .18034 / .18034 | .18453 / .18453 / .18453 | .16844 / .16844 / .16844 |
| Branch | 8 | 11/12 | 0/12 | .17373 / .17342 | .20458 / .20468 / .20373 | .18331 / .18297 / .18247 |
| Chain | 4 | 0/12 | 12/12 | .16714 / .16714 | .17164 / .17164 / .17164 | .14988 / .14988 / .14988 |
| Chain | 8 | 11/12 | 0/12 | .16249 / .16233 | .18460 / .18407 / .18390 | .15865 / .15813 / .15806 |
| Additive | 4 | 1/12 | Outside family | .23212 / .23180 | .25641 / .25641 / .25609 | .23694 / .23683 / .23665 |
| Additive | 8 | 12/12 | Outside family | .20750 / .20706 | .25675 / .25747 / .25573 | .23834 / .23790 / .23708 |

For the eight-provider branch law, the endpoints differ on an average of 5.50 Boolean corners out of 256. They give different probabilities on 3.51% of the IID population and 3.03% under shift. Their mean squared difference is .003620 and .002651, respectively. The corresponding chain figures are 2.67 corners, 1.71% and 1.48% population mass, with squared differences .001754 and .001309. On the training labels the true branch and chain probabilities had Brier .00881 and .00621 above the achievable minima, respectively. [Summary for all six cells](posthoc-cut-means.csv)

## A certificate applicable to every empirical minimizer

At every input any minimizing tree, and any convex average of such trees, predicts in the interval between the two extremal probabilities. The squared distance from the true conditional probability to that interval therefore lower-bounds conditional excess Brier. Integrating over the 4⁸ inputs gives a lower bound for each bank. Pointwise nearest predictions need not jointly be attainable by a single minimizing tree.

On the eight-provider branch law, the lower bound averaged **.01790** under IID. The validation-selected BRANCH method attained **.00593**. On 10 of 12 banks its mean loss across both starts fell below the bound for every empirically minimizing tree on that bank. On the eight-provider chain law the corresponding lower bound was **.01164**, and CHAIN attained **.00966**, below the bound on 7 of 12 banks. This result applies to the specified losses, input distribution, training sample sizes and two fixed search seeds. [Per-bank bounds and frozen GP comparisons](posthoc-cut-bounds.csv)

## Numerical definitions and checks

The focal search fits squared loss with integer numerator on denominator 400n. The levels are exactly 2, 7, 13 and 18 divided by 20. We built an integer-capacity cut and obtained both extremal solutions from its residual graph. The public weighted min-cut instead works exactly on the binary64 representations supplied to it. Its winners attained the same nominal minimum in all 72 cells. Its two extremal rules gave a strictly narrower interval on at least one Boolean corner in 16 cells, including 5 of 12 eight-provider branch banks and 4 of 12 eight-provider chain banks. The original report's lower-rule IID Brier for the eight-provider branch law, .20413, uses the binary64 convention. Both conventions appear in the raw table.

For all 36 four-provider cells we independently scored all 166 functions using exact arithmetic under both conventions and checked that the intersection and union of every tied truth table matched the inferred endpoints. For every cell we checked monotonicity, pointwise ordering, attainment of the certified optimum by both endpoints, and the squared-loss identity for the midpoint ensemble. [Reproduction code](posthoc_cut.py)

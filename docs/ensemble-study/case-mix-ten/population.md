# How much can correcting the case mix achieve?

Suppose training data were unlimited and a fusion rule could learn the exact probability of a positive label for every pattern of four provider reports. On the mixed source, rebalancing the two training environments reduces the worst endpoint regret from **0.08328 to 0.04194**. The reduction is possible because the source makes the shortcut most reliable precisely where the regional specialist already works well. The opposite association is also harmful. Rebalancing the reversed source reduces regret from **0.09298 to the same 0.04194**. These are calculations for population Bayes predictors. The native OKAPI experiments separately measure how much a finite tree search gains. [Exact results](results/population.csv)

| Source | Unweighted max regret | Cap 2 | Full weighting | Effective sample fraction, cap 2 / full |
|---|---:|---:|---:|---:|
| Mixed | .08328 | .05496 | .04194 | .716 / .510 |
| Reversed | .09298 | .04418 | .04194 | .716 / .510 |
| Extreme difference in regional prevalence | .10105 | .08722 | .04194 | .814 / .135 |
| Mixed, 25% errors in context metadata | .08328 | .07625 | .07625 | .878 / .878 |
| Regional prevalence alone varies | .04194 | .04194 | .04194 | .716 / .510 |
| Shortcut activity alone varies | .04194 | .04194 | .04194 | 1 / 1 |

Regret is expected full-class Brier minus the Bayes Brier in each deployment world. The maximum uses the two worlds in which the shortcut is inactive or always active. Both have 50% regional prevalence. On the mixed source, full weighting *increases* regret in the inactive world from .01913 to .02876 and decreases it in the active world from .08328 to .04194. On the reversed source the inactive world initially limits performance (.09298 vs .01746). The same correction can thus help different endpoints under the two designs. [Both endpoint risks and regrets](results/population.csv)

## What changes under weighting

Each source environment $e$ has known regional prevalence and shortcut activity. Let $c$ denote the observed context metadata and $x$ one of the 16 patterns of provider reports. For noiseless metadata $c$ is the regional cue in $x$. Give each case weight $w(e,c)=P(e)P(c)/P(e,c)$. For capping, truncate at 2 before renormalizing to unit mean. The reweighted population that the estimator would learn is

$$
J_w(x,y)=\frac{\sum_{e,c}w(e,c)P(e,c,x,y)}{\mathbb E[w]},\qquad
p_w(x)=\frac{J_w(x,1)}{J_w(x,0)+J_w(x,1)}.
$$

In the mixed source the two equally likely environments have regional prevalences .15 and .85. The shortcut is inactive in the first and active in the second. Conditional on being outside versus inside the region it is therefore active on 15% versus 85% of cases. Full weighting restores equal representation of the environments within both contexts, giving 50% activity in each. The resulting **joint distribution** agrees on all 16 patterns and both outcomes with the target generator at shortcut activity .5. Capping at 2 gives 37.5% and 62.5% activity. Reversing the source exchanges these pairs. In these symmetric designs, capped and uncapped weighting preserve the original marginal proportions of both source environments and observed contexts. This preservation under clipping depends on the design.

The two target worlds disagree about the correct probability at some report patterns. A single forecast minimizing maximum regret in the two worlds learns the mixture at activity .54890, with regret .03568 at both endpoints. Balancing a clean source to independence produces activity .5, giving worst regret .04194, a gap of .00626 to this theoretical optimum. The balancing rule has no access to the actual target law. Its benefit depends on which endpoint risks matter in deployment. [Minimax derivation](../provider-use-ten/oracle.md)

## Missing information and the cost of weighting

**Weight concentration.** With cell proportions .15 and .85 within each environment, full weights are .5/.15 and .5/.85. Their population effective sample fraction is $(\mathbb E[w])^2/\mathbb E[w^2]=4(.15)(.85)=.51$. Thus 512 observations have an asymptotic effective count of 261, or 65 for 128 observations with the same distribution. In the extreme .035/.965 design the fraction falls to .135, equivalent to 69 of 512 observations. Capping retains 81.4% there but leaves shortcut activity at .123 outside the region and .877 inside. Finite counts and the study's minimum-four cell fallback can change the realized result.

**Noisy context.** The source metadata are flipped with probability .25 independently of everything else conditional on the true region. The true cue remains in the reports. For a fixed pattern we therefore integrate the two weights with probabilities .75 and .25. The rarer observed cell within an environment now has probability .325, so the largest raw weight is .5/.325 = 1.538. Capping at 2 changes nothing. Even full weighting leaves shortcut activity at .201 outside and .799 inside the true region.

**Controls.** If only regional prevalence differs while shortcut activity stays fixed, the population posteriors conditional on reports are identical in both environments. Reweighting cannot improve them, yet effective sample size falls. If only shortcut activity differs and regional prevalence is the same everywhere, the population weights are all one. When both training environments completely lack an active shortcut, the reweighted Bayes predictor keeps ignoring it and incurs .12322 excess Brier when it becomes active at deployment. This last example is a population control outside the empirical panel. [Enumeration and numerical checks](population.py)

From the repository root, run the following command to regenerate all 36 population rows and their numerical checks.

~~~sh
PYTHONPATH=. python docs/ensemble-study/case-mix-ten/population.py
~~~

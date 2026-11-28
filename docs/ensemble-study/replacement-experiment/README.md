# Does separating breeding from replacement prevent validation overfitting?

**Keep the distinction between breeding and deployment as an experimental control. The KL replacement rule tested here was too conservative for general use.** On 24 standard scenarios it left the initially selected provider in place on 140 of 144 banks. On the same open search trajectory, replacing the incumbent by observed improvement instead of a confidence bound lowered test Brier by 0.08434. The population and the selected models were genuine native OKAPI trees. The 72-offspring budget scoped the experiment to a short native tree search. Other higher-budget or regularized OKAPI pipelines have different scores. [Frozen design](protocol.md) · [Full results and independent controls](panel/README.md)

## Controlled mechanism

We compared ordinary native search, empirical replacement with open breeding, certified replacement with open breeding, and certification for both breeding and replacement. Each live run offered 12 offspring per generation for six generations. Open variants shared the same parents, mutations and offered trees. The deployment decision used six fresh confirmation blocks. Each block was revealed after its proposals were fixed. Every decision ended with a single executable tree. Retaining the initial mean in a controlled study retained its full provider dependency.

* **Protecting a good incumbent.** In 48 binary-classification banks whose optimal rule was the three-provider mean, both certified variants preserved that optimum throughout. On the same open proposal path, empirical replacement left the optimum at least once on 41 banks and finished away from it on 22. Ordinary search finished away from the optimum on 24. [Exact conditional risks and paths](controlled/README.md)
* **Rejecting a genuine improvement.** On the matched binary law with posterior MIN(C, MAX(A,B)), open breeding produced an improvement over the mean in 47/48 banks. Empirical replacement finished ahead of the mean in 41/48, open certified replacement in 1/48. We then bypassed proposal and offered the exact program. It was worth 0.140625 in population Brier. Raw comparison accepted it on all 48 fresh blocks and certified comparison accepted it on zero. For the 12-slot rule, even the most favorable conceivable labels on each of these 48 blocks' observed features could not clear the bound. This is a failure of statistical power at the prescribed block size. Live search also failed to offer that exact program within budget. [Witness and power audit](controlled/README.md)
* **Breeding through neutral intermediates.** On a separate XOR law we supplied two native intermediate trees, each exactly tied with the starting mean in population. Their native MAX is the true posterior. Keeping both available for breeding allowed the complete tree to pass the same deployment test on 30/32 banks. Requiring each intermediate to pass the test immediately closed the route on 32/32. In unassisted GP no arm discovered exact XOR within budget. Open certified breeding did deploy a partial improvement on 5/32 banks, and strict breeding on zero. The five improved deployments had mating ancestry containing an unconfirmed offspring. Those edges record reproduction. Attributing an improvement to inherited material would require a separate intervention. [Natural runs and supplied-route replay](barrier/README.md)

## Full-suite regression

All 24 generators contributed six independent banks, with 1,024 test observations per bank. Equal-weight suite mean full-class Brier is lower for better predictors. The four main arms all had 256 discovery and six disjoint 64-row confirmation blocks. The unsplit ordinary control used all 640 for discovery. Top-N and logistic stacking in the table were fit on all 640. The panel also reports their 256-row controls. [All seeds, scores and cost accounting](panel/README.md)

| Deployment | Test Brier |
|---|---:|
| Probability logistic stacking | 0.24431 |
| Ranked-prefix Top-N average | 0.33838 |
| Ordinary native GP, 640 discovery rows | 0.35892 |
| Ordinary native GP, 256 discovery rows | 0.36053 |
| Open breeding, empirical replacement | 0.36252 |
| Uniform mean | 0.37756 |
| Open breeding, certified replacement | 0.44686 |
| Strict breeding and certified replacement | 0.45041 |

Open certified minus open empirical was +0.08434 (paired 95% interval +0.06123 to +0.10481). Open minus strict certified was −0.00355 (−0.00985 to +0.00177). The latter interval crossed zero and the two policies tied on 138 banks. Intervals resample the six independent banks within each of the 24 fixed scenarios. On average the open certified final program referenced 1.03 providers, compared with 1.94 under open empirical replacement. Keeping the initial best singleton cheaply often gave up genuine ensemble gain. Safety for conditional expected gain on observed features is distinct from generalization to new rows and distribution shifts. Exact-quota scenario designs, such as selective fallback, can also induce dependencies between units. Their results are empirical evaluations. [Scenario-level contrasts](panel/results/scenario-contrasts.csv) · [Deployment and compute budgets](panel/results/mean-budgets.csv)

## Reproduce

Use FusionBench from this research branch with its OKAPI extra and the pinned [native implementation](https://github.com/damiankucharski/OKAPI/commit/1d7ad2d1660627585329bc584a9c4634915e8bb1). Each component report gives its exact command, seeds and archived outcomes. The common [FusionBench runner](../../../fusionbench/eval/replacement_search.py) returns every proposal, native mating parent IDs, eligibility, raw and lower gains, active survivors, incumbent history and path digests. The default solver is unchanged.

The archived [cross-study verifier](verify_study.py) reconstructs the controlled population risks, all generation-level decisions on the supplied XOR path, the suite's paired contrasts and a multiclass native replay. The full suite took 14.8 seconds on two workers, the matched 384-search controlled run took 9.4 seconds, and the XOR runs took about five seconds. Protocol registration through independent verification took about seven minutes, excluding research design and the subsequent raw-score precision check.

In an explicitly post-result floating-point check we repeated the empirical-promotion arm on all 240 panel and matched-control banks, requiring a gain above 1e−12. All native paths matched the original runs. Harmful null replacement remained 41/48. The number of signal banks finishing better than the mean changed from 41 to 40. All 144 full-suite test losses remained identical to within 1e−12. [Numerical sensitivity and replays](roundoff/README.md)

The next worthwhile test is power at a fixed label budget. On known beneficial, neutral and harmful candidates, measure whether a paired bound that uses each case's own loss range can admit genuinely beneficial changes at the same false-promotion target. The known witness provides a cheap preflight before another live search. Track both the cost of keeping fresh rows for confirmation and inference-time provider count.

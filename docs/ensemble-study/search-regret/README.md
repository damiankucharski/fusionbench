# Where OKAPI loses accuracy on a four-provider benchmark

With 128 fit labels on TwoPathways, 48 generations of MIN/MAX genetic programming found an empirical optimum in **0 of 16 runs**. Across these runs it evaluated 317 trees per run on average and encountered 13.1 of 166 distinct decision functions. Exhaustive empirical selection gave 18.5% exact IID population error. The GP archive winner gave 32.5%. The generator's irreducible error was 18%. With 32 labels, exhaustive selection itself gave 27.5% IID error. The finite benchmark exposes both a search failure at the larger fit size and sampling error at the smaller size.

Each row below averages search restarts within each of eight independent training draws. Entries are exact population error probabilities averaged over those draws. Every candidate's risk integrates all 16 inputs and label noise. The raw file also includes independent, sampled validation and IID/shifted test errors. The shift changes four input prevalences and preserves the labeling law.

| Task | Fit labels | Exact MIN/MAX ERM | GP with MIN/MAX | GP, six operators | Arithmetic mean | Logistic stack | Bayes |
|---|---:|---:|---:|---:|---:|---:|---:|
| TwoPathways | 32 | .275 | .355 | .350 | .260 | .340 | .180 |
| TwoPathways | 128 | .185 | .325 | .312 | .260 | .280 | .180 |
| XOR | 128 | .360 | .405 | .380 | .540 | .490 | .180 |
| Additive | 128 | .291 | .344 | .347 | .315 | .278 | .259 |

- **Optimization.** At 128 TwoPathways labels, the MIN/MAX archive winner incurred an average 12.44 additional training errors over exhaustive ERM. Uniformly drawing as many distinct functions as the GP visited yielded 4.25 extra errors and .222 IID risk. Only two random reference orders were used. The full six-operator search reached .312 IID error with this small population. The MIN/MAX optimum only certifies the matching grammar.
- **Generalization.** At 32 TwoPathways labels, exhaustive ERM achieved .168 mean empirical error and .275 mean IID error. The sample admitted 7.75 empirical optima on average. Their mean per-seed minimum and maximum IID risks were .225 and .325. At 128 labels the same search admitted 1.25 optima on average. Under the prevalence shift, the exact ERM, MIN/MAX GP, six-operator GP, average and logistic stack had respective error rates .182, .329, .272, .212 and .276.
- **Misspecification.** On noisy XOR the best possible monotone tree has .340 IID risk. The Bayes error is .180. For 32 labels the MIN/MAX GP hit the empirical optimum in 13 of 16 runs despite this unavoidable gap. The additive control instead favored regularized logistic stacking over the empirical tree estimator.
- **Coverage.** Four single-provider trees initialized the actual OKAPI engine. The hit rates measure the specified small population and operator configuration. Each tree was scored on one representative row per input cell using fit-only counts; this yields the same 0–1 fitness as scoring all original rows. Unobserved cells contributed zero to the objective. Validation-based selection used an additional 128 labeled rows and appears separately in the data.

The registered plan, test settings and raw observations accompany this report. Source-data seeds, GP seeds and elapsed wall time are included. An independent test enumerates all 65,536 truth tables and checks that our certified set contains every nonconstant monotone function and only those functions. The GP runtime used OKAPI research revision f850e5f; its relevant optimizer, operator and lattice enumeration source files match the optional dependency pin a7c23e8.

```bash
python -m fusionbench.eval.search_regret --out results --workers 2
python -m pytest -q tests/test_search_regret.py
```

[Protocol](protocol.md) · [Configuration](results/config.json) · [Summary](results/means.csv) · [Raw runs](results/runs.csv.gz)

A follow-up on 12 new datasets measured the number of essential providers for every visited decision function. [Independent coverage analysis](coverage.md)

Changing only the starting trees at equal fitness-call budgets tests whether broader initial coverage changes optimization and population error. [Starting-tree intervention](initialization.md)

A separate confirmation repeats the diagnosis with four probability levels per provider and expected Brier risk. [Graded probability benchmark](graded.md)

On continuous outputs, an integer minimum cut computes the fit optimum without enumerating the trees. The eight-provider study includes the mean, tuned Top-N, logistic stack and both native GP selection policies. [Exact Brier certificate on continuous predictions](continuous.md)

A paired replay before and after OKAPI's tree identity correction measures its effect on observed search paths, exact regret and held-out predictions. [Structural key audit](structural-key/README.md)

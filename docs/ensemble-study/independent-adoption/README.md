# Does requiring independent evidence improve tree deployment?

The prespecified conservative deployment rule never selected a GP tree in this experiment. In the nonlinear, tree-favorable population, it missed every beneficial nomination at all three validation budgets. On the intermediate population it prevented several costly switches. Choosing a default rule requires accounting for missed gains as well as mistaken adoptions.

## Equal-budget comparison

Each row averages two banks of four actual OKAPI searches on each of 12 independent labeled training sets. Fit size was 256. With 512 validation examples, the gate used 256 to nominate a baseline (mean or fixed regularized logit stacker) and a tree, then 256 fresh labels to test that pair. Naive full validation used all 512 examples to rank the same six candidates. All methods saw identical expert probability channels. Lower binary Brier is better.

| Population | Mean | Fixed stacker | Unconditional nominated GP | Naive full validation | Independent gate | Stacker refit on all 768 |
|---|---:|---:|---:|---:|---:|---:|
| Stacking law | .18024 | .14948 | .16972 | .14948 | .14948 | .14809 |
| Halfway between laws | .20484 | .18595 | .19211 | .18696 | .18595 | .18449 |
| Maximum of three channels | .22943 | .17343 | .15793 | .15787 | .17343 | .17172 |

The halfway case illustrates selection optimism. The nominated GP averaged .18860 on its fit records, .18927 on the validation set that partly selected it, and .19211 on fresh IID covariates, versus .18595 for the stacker. On the stacking law the corresponding GP numbers were .16402, .16811 and .16972. On the maximum law IID GP risk was lower than its realized fit loss (.15793 versus .16409). The population comparison integrates out binary label noise conditional on held-out covariates. Realized label losses are available for every policy.

## Errors at the decision level

Counts below refer to 24 decisions per cell. “Harmful” compares the deployed tree with the nominated baseline on independent IID risk; “missed” counts an advantageous nominated tree rejected by the gate. Signs close to zero are estimated on 8,192 independent covariates, so these counts have Monte Carlo uncertainty.

| Population | Validation labels | Naive tree adoptions | Harmful among naive adoptions | Gate adoptions | Gate misses / beneficial nominees |
|---|---:|---:|---:|---:|---:|
| Stacking | 32 / 128 / 512 | 5 / 0 / 0 | 3 / 0 / 0 | 0 / 0 / 0 | 2/2, 2/2, 0/0 |
| Halfway | 32 / 128 / 512 | 9 / 6 / 6 | 8 / 6 / 5 | 0 / 0 / 0 | 5/5, 2/2, 2/2 |
| Maximum | 32 / 128 / 512 | 18 / 22 / 24 | 0 / 0 / 0 | 0 / 0 / 0 | 24/24, 24/24, 24/24 |

On the maximum law with 512 labels, simply taking the better of the two frozen nominees on the second half chose GP 24/24. Requiring the log e-value to pass log(20) ≈ 2.996 changed this to 0/24. The largest observed value was 2.987. Relative to the conservative gate, naive full validation lowered IID risk by .01556 (paired 95% t interval .01371 to .01740 over 12 data seeds). On the halfway law it raised risk by .00101 (−.00033 to .00234). These intervals share data across comparisons and are descriptive. The two GP banks for each data seed share all examples. In the stacking law with 32 validation labels, the gate's fallback itself scored .15214 against .14948 for the fixed stacker. Independently testing the tree does not remove uncertainty from baseline selection.

## Shift

Replacing the maximum law with the stacking law at deployment leaves the distribution of all inputs unchanged. At validation size 512, risks moved to .25557 for naive full validation, .24145 for the conservative fallback and .18070 for the arithmetic mean. In the opposite shift, the unconditional nominated tree scored .24287, the gate .26748 and the mean .22960. A gate trained and tested on IID validation data has no guarantee under this change in the label law.

## Scope and reproduction

The finite-sample false-adoption bound pertains to one frozen pair with IID labels, conditional on nominations. Its proof follows from a nonnegative product with expectation at most one under the null; it does not promise high power. The gate saved no validation labels here and spent the same search budget as the other selectors. The full-data stacker control shows that reserving validation labels has a cost even when model selection often chooses the stacker. These three constructed populations are designed to expose contrasting decision errors; they do not establish their frequencies on real expert pools.

FusionBench ran 288 GP fits in a 40.35-second study on CPU. Mean fit-selected tree sizes across the three populations were 7.34, 7.39 and 4.82 nodes, with a common cap of 11. The full run includes all model predictions, policy choices and summaries. Raw artifacts record per-candidate and per-policy metrics rather than individual prediction tensors.

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OKAPI_LOG_LEVEL=ERROR \
python -m fusionbench.eval.independent_adoption --out results/independent-adoption
```

Use the CPU dependencies of FusionBench and OKAPI revision `f850e5f78fef7e71d723bade611617c8c602fc1f`. The protocol was published before training in `b3ff784`. [Protocol](protocol.md) · [Policies and adoption counts](results/summary.csv) · [Paired differences](results/paired.csv) · [Per-candidate losses](results/candidates.csv) · [Individual decisions](results/policies.csv) · [Per-search costs](results/search.csv)

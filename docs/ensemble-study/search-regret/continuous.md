# An exact fit certificate for continuous MIN/MAX ensembles

The problem is to find a MIN/MAX tree with the smallest empirical Brier loss for binary labels and arbitrary continuous expert probabilities. On eight experts we can now compute the optimum over every tree size using a single cut on 256 subsets of providers. The resulting rule gives the attaining predictions. This makes the search gap directly measurable on the existing two-pathway generator.

## Why one cut suffices

For a fixed threshold, every tree answers a Boolean question about the set of providers whose scores reach that threshold. MIN acts as AND and MAX as OR. Let f(S) denote that Boolean answer for the subset S. The accepted subsets form an upward-closed collection, with f(empty)=0 and f(all)=1. Conversely, the maximum of the minimum scores in each minimal accepted subset realizes any such collection. This gives exactly the family of pure MIN/MAX trees with no constants and identity postprocessing.

For one labelled observation, sort the provider values into t₁ ≤ … ≤ tₖ, set t₀=0, and let Sⱼ contain the providers scoring at least tⱼ. The squared loss at the final tree score p has the identity

```text
(p − y)² = (0 − y)² + Σⱼ [(tⱼ − y)² − (tⱼ₋₁ − y)²] f(Sⱼ).
```

The nonzero accepted increments end at the tree's attained value. Equal consecutive thresholds contribute zero. Across observations, all increments involving the same subset can therefore be summed into one signed weight. Minimizing this linear function of Boolean decisions over upward-closed subsets is minimum-weight closure. Each upward lattice edge has capacity greater than the sum of the absolute unary weights. Two additional constraints anchor the full and empty subsets. The residual graph identifies both the least and greatest optimizing rules.

Binary64 probabilities are dyadic rationals. Converting them to a common power-of-two denominator before squaring gives integer capacities. Summed loss is verified for both extreme attaining rules against the flow value. The implementation returns the optimum as a rational number. Enumeration of all 166 functions at four inputs is a verification tool only. The graph uses 2ᵏ vertices plus two terminals and k·2ᵏ⁻¹ upward edges, so the implementation currently limits the pool to 12 experts.

## TwoPathways with eight continuous inputs

Each of four latent conditions has two noisy calibrated detectors. The label is (A AND B) OR (C AND D). Lowering detector accuracy from .90 to .78 supplies the shifted domain. The benchmark used eight independent training/test draws and three search seeds each. Every native GP started from eight single providers and optimized Brier for 300 fitness calls. Only MIN and MAX were allowed. Fit and independent validation winners were selected before inspecting either test split. Every baseline saw the same fit rows and eight providers. The data also include the winning Top-N prefix, complete order, and all fit-set prefix scores.

At 128 fit labels, the exact empirical loss was **.05191**. The best program seen by native GP averaged **.18045** on fit, leaving a certified optimization gap of **.12855**. The paired 95% t interval across the eight data draws was .11821 to .13889. This isolates search error even though these data also admit overfitting.

| Method | Fit, 128 labels | Independent IID test | Shifted test |
|---|---:|---:|---:|
| Exact MIN/MAX fit optimum | .05191 | .09732 | .17619 |
| Native MIN/MAX GP, fit winner | .18045 | .21254 | .23139 |
| Native MIN/MAX GP, additional validation for selection | .21969 | .22201 | .23788 |
| Arithmetic mean of eight providers | .15337 | .15528 | .19271 |
| Ranked-prefix Top-N mean | .14745 | .16191 | .19694 |
| Regularized logistic stack | .11745 | .13997 | .17795 |
| Analytic Bayes predictor supplied by the simulator | .03269 | .03775 | .12508 |

The exact fit rule improved over each of the three inexpensive baselines on mean IID Brier. Its mean advantage over the logistic stack decreased from .04265 IID to .00176 under shift, with the shifted comparison spanning zero across draws (−.00192 to +.00544). Top-N selected 5–7 detectors at 128 fit labels. With 32 labels, the exact tree's fit loss was .03222, its IID test loss .11330 and the arithmetic mean's IID loss .15528. The gap between fit and test remains even after search is solved.

The raw file contains full rational certificates and selected prefix metadata. The recorded prefix scores use the FusionBench two-class Brier sum, twice the positive-channel loss in this table; the ordering and chosen prefix are invariant to that factor. A validation fold was available only to the explicitly named GP validation-selection policy. The other deployed rules were frozen after fitting. The supplied Bayes predictor has privileged knowledge of the simulator and serves as a reference only.

The oracle can output a larger tree. At 128 labels its DNF description averaged 11.375 conjunctions and 43.25 provider literals. These canonical-form counts do not measure minimum possible compiled circuit size. Search, unlike the exact comparator, rounds the fitness used for internal selection. Larger ensembles or the full OKAPI grammar with averaging and threshold operators can have different optima. Certifying the fit minimum does not certify the population optimum; the test risks here are independent finite-sample estimates.

## Reproduction

```bash
uv sync --extra lattice --extra okapi
uv run python -m fusionbench.eval.search_continuous --workers 2 --out continuous
uv run python -m pytest -q tests/test_lattice_brier_oracle.py
```

`lattice_brier_exact` is an opt-in comparator in the normal FusionBench method registry, including evaluation on imported binary probability data. The standard CLI reports the sum of both squared class-probability errors; divide that value by two to match this report's positive-channel Brier. The solver needs the lattice extra and works independently of the OKAPI search package.

[Protocol and dated Top-N amendment](continuous-protocol.md) · [Summary](continuous/means.csv) · [Raw results](continuous/runs.csv.gz) · [Solver](../../../fusionbench/eval/lattice_brier_oracle.py)

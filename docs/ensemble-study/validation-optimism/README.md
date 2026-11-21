# How much of the winning GP tree's validation gain survives a new sample?

Selecting among twelve OKAPI trees on 64 validation cases improved performance on an independent IID test, but the improvement visible on the selection cases was larger. Across 16 new training banks, the apparent Brier gain over the first tree was **0.00738**, and the independent test gain was **0.00288** (95% paired bank interval 0.00126 to 0.00450). The excess validation optimism attributable to the larger menu was **0.00379** (0.00158 to 0.00600). Both prespecified one-sided Bonferroni tests passed. A probability-feature logistic stack trained on the same 384 fit examples still had lower test risk than any of the GP selection policies.

## Same models on three independent samples

Each bank contained twelve native fit-selected champions evolved on the same 384 examples. Seven experts supplied probabilities, with their strongest member inverted for a random 10% of observations. For each of twelve repeats per bank we fixed a random order of the twelve champions, selected the best on sample A among the first k, then scored that exact tree on independent sample B, and on independent test observations. Each bank's common test integrated clean and inverted outcomes with the source weights. A 50% inversion test changes only these weights. The confidence intervals use one mean per independent training bank, aggregating the twelve selections within it.

| Pool size | Fit Brier | A, used to select | B, untouched | IID test Brier | Apparent gain on A | Gain on IID test | Excess B−A over first |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.07404 | 0.07689 | 0.07647 | 0.07670 | 0 | 0 | 0 |
| 4 | 0.07218 | 0.07210 | 0.07328 | 0.07433 | 0.00479 | 0.00236 | 0.00160 |
| 12 | 0.07066 | 0.06952 | 0.07289 | 0.07382 | 0.00738 | 0.00288 | 0.00379 |

B−A for the winning tree was 0.00337. For the first fixed tree it was −0.00042. Subtracting these directly paired offsets gives 0.00379. The independent gain on B was 0.00359. By definition, apparent gain less excess optimism equals the gain on B. The separate large test estimates the same IID effect on different observations.

With twelve exact copies of the first champion, every pool size returned identical predictions by construction. The copy pool retained the fixed tree's −0.00042 B−A sampling offset; its excess optimism and search gain were zero at every repeat. In the secondary n=256 comparison, twelve real programs gave more independent test gain (0.00402) and less excess optimism (0.00153) in point estimate. After shifting the inversion rate from 10% to 50%, the 64-case selector scored 0.18206 versus 0.17889 for the first tree. Its shift gain of −0.00317 had a descriptive interval spanning −0.01058 to +0.00424.

## Equal-label comparisons

| Method | Training labels | IID test Brier | 50% inversion Brier |
|---|---|---:|---:|
| All-provider arithmetic mean | None | 0.11048 | 0.14292 |
| Learned Top-N | 384 fit | 0.07635 | 0.16805 |
| Probability-feature logistic stacking | 384 fit | 0.05383 | 0.14002 |
| Logit-feature logistic stacking | 384 fit | 0.05448 | 0.13741 |
| Best of 12 GP champions on A | 384 fit + 64 selection | 0.07382 | 0.18206 |
| Learned Top-N, refitted | 384 fit + 64 A | 0.07619 | 0.16836 |
| Probability-feature logistic stacking, refitted | 384 fit + 64 A | 0.05348 | 0.13971 |
| Logit-feature logistic stacking, refitted | 384 fit + 64 A | 0.05409 | 0.13694 |

All comparisons share the same provider predictions for each observation. Top-N ranks experts and checks every equal-weight prefix. Refit baselines make full use of the selection labels as training data, with untouched B and test. The A column for these refits in the raw files is an in-sample score.

## Previously archived banks and scope

We separately revisited the previously published 16 banks, pairing each even-numbered validation repeat with the following odd-numbered repeat. The 64-case, 12-tree result showed excess optimism 0.00249 and independent IID gain 0.00372. The original study measured expected test gains over random subsets and how rankings change under shift. The paired A/B replay directly identifies each selected champion. Those old tests had already been analyzed, and are excluded from the new study's confidence intervals.

These results concern one synthetic inversion law and a fixed GP search budget. They quantify selection among trained trees. The shifted test illustrates how this conclusion depends on the deployment law. Every selected GP program was evaluated directly. The 192 new searches made 59,985 full fit-objective calls, averaged 6.76 nodes per champion (range 1–33), and took 18.8 seconds combined. The entire analysis took 30.0 seconds on one CPU thread.

## Reproduction and data

Use the benchmark's pinned OKAPI revision a7c23e8e8c8f0bfe608bc7740b1a74a26eef6e8b. Run from the FusionBench checkout. Each decision identifies its GP seed, which joins to the per-bank tree and fitness-call count in search.csv. Existing archive validation and conditional test scores are distributed in the benchmark's results.zip.

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OKAPI_LOG_LEVEL=ERROR \
python examples/validation_optimism.py --out docs/ensemble-study/validation-optimism/results
python docs/ensemble-study/validation-optimism/verify.py
```

[Preregistered protocol](https://github.com/damiankucharski/fusionbench/blob/d2a29ea6e545db1eac546afe14af56b5fa95d2ce/docs/ensemble-study/validation-optimism/protocol.md) · [Prior subset study](../seed-selection-pl.md) · [Related adoption study](../independent-adoption/README.md) · [All intervals](results/estimates.csv) · [Reference methods](results/baseline_means.csv) · [Per-candidate validation scores](results/candidates.csv.gz) · [Frozen selections](results/decisions.csv.gz) · [Search costs](results/search.csv)

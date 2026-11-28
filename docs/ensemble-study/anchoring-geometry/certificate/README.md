# How often can the Brier certificate beat the mean?

The certificate never promoted any challenger at 16 labels, even when the grid contained the true probabilities. With the same strong signal it promoted a beneficial challenger 54.15% of the time at 64 labels and 99.99994% at 256. These are probabilities over all possible independent label draws on a fixed two-context design.

| Labels | Strong-signal power, raw | Strong-signal power, certified | Expected true gain, raw | Expected true gain, certified | Null wrong promotion, raw | Null wrong promotion, certified |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 0.998504 | 0 | 0.307788 | 0 | 0.494568 | 0 |
| 64 | 0.9999993 | 0.541501 | 0.319741 | 0.172243 | 0.039721 | 9.49 × 10⁻¹⁰ |
| 256 | ≈1 | 0.9999994 | 0.320000 | 0.319999 | 9.85 × 10⁻⁶ | 1.72 × 10⁻¹⁵ |

Each row contains equal numbers of labels from each context. The anchor predicts 0.5 in both. The eight challengers use 0.1, 0.5 or 0.9 in each context. The strong law has true probabilities (0.1, 0.9), with maximum full binary Brier gain 0.32. The null law has true probabilities (0.5, 0.5).

For each challenger, the verifier subtracts the empirical full Brier loss from the anchor's loss. The certified rule then deducts the one-sided Hoeffding charge `sqrt(log(160) × sum(range_i²) / (2 n²))`, where `range_i = 4 × abs(p_i − 0.5)` is the difference between the two possible label-dependent gains on row i. The factor 160 comes from eight challengers, equal weights and a family error target of 0.05. Both rules keep the anchor unless a challenger's score is strictly positive. Labels in the two contexts are independent, so every count pair has a product-binomial probability. The verifier sums these probabilities as integer weights and rounds only reported decimals.

At 16 labels, even the most favorable label counts cannot clear the charge. A challenger that departs from the mean in both contexts can gain at most 0.48 empirically, and its charge is about 0.637. Changing one context gives at most 0.24 against a charge of about 0.451. At 64 labels the two-context charge falls to 0.319, close to the strong law's 0.32 expected gain.

With probabilities (0.4, 0.6), every challenger overshoots or moves in the wrong direction. Raw selection promoted a harmful tree with probabilities 0.5970, 0.2182 and 0.0200 as sample size grew. Certified selection reduced these to 0, 2.07 × 10⁻⁷ and 3.96 × 10⁻⁹. With probabilities (0.3, 0.7), changes in the correct direction tie the anchor in population. At 256 labels the raw rule promoted a non-improving tree with probability 0.7627, but its probability of strictly harmful promotion was only 1.37 × 10⁻²⁰. The corresponding certified probabilities were 0.000409 and 1.66 × 10⁻³⁰. The full table distinguishes harmless ties, harmful promotion and expected loss for every law. [All 24 comparisons](results.csv)

For this menu, the certificate is unsuitable as a default at 16 labels and misses nearly half of the strong improvements at 64. The 256-label strong result supports a targeted trial where a comparable signal is plausible. The calculation gives no power estimate for larger candidate menus, trees fitted on the certification labels, dependent labels, or fresh random context proportions.

From the repository root, run the following command to verify the saved table. [Frozen protocol](protocol.md)

```sh
python docs/ensemble-study/anchoring-geometry/certificate/verify.py
```

# Bernoulli KL certificate on the fixed binary menu

## Protocol frozen before enumeration

This exploratory follow-up was specified after the Hoeffding certificate failed to promote any tree at 16 labels. Reuse the eight raw programs, anchor 0.5, four probability laws, sample sizes 16, 64 and 256, equal context counts, one-sided family error target 0.05, uniform program weights, four reported metrics and exact binomial enumeration from the [first protocol](../protocol.md).

On a row where a challenger moves from 0.5 to 0.1 or 0.9, the full Brier gain is 0.48 when the label agrees with the direction of the change and −1.12 otherwise. On the other rows the gain is zero. For M active rows among N, let v be the fraction that agree. Take the lower root p ≤ v of M KL(v ‖ p) = log(160), with p = 0 at v = 0 and p = 160^(−1/M) at v = 1. The lower bound on expected full Brier gain is (M/N) × 1.6 × (p − 0.7). Keep the anchor when all bounds are nonpositive. Resolve a positive tie by lexicographic order of the challenger probability pair. Use this rule without adjusting any threshold after enumeration. Compare its results with raw selection and the published Hoeffding certificate.

The prespecified small-sample check is whether the strong-law certificate succeeds on the event that all 16 labels agree with the true challenger, which has probability 0.9¹⁶.

## Results

The KL bound recovers exactly the predicted small-sample event. With 16 active rows, 16 agreements yield a lower alignment probability of 0.7282 and a positive certified gain of 0.0451. With 15 agreements the lower probability is 0.5896. Even eight agreements on all eight active rows of a one-context challenger give just 0.5303. Therefore only perfect agreement in both contexts can trigger promotion. The strong-law probability of beneficial promotion is exactly 0.9¹⁶ = 0.185302. The null permits four perfectly agreeing two-context directions, with total probability 4/2¹⁶ = 0.0000610352.

| Labels | Strong power, Hoeffding | Strong power, KL | Expected gain, Hoeffding | Expected gain, KL | Null wrong promotion, Hoeffding | Null wrong promotion, KL |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 0 | 0.185302 | 0 | 0.059297 | 0 | 6.10 × 10⁻⁵ |
| 64 | 0.541501 | 0.831555 | 0.172243 | 0.257674 | 9.49 × 10⁻¹⁰ | 4.93 × 10⁻⁷ |
| 256 | 0.999999406 | 0.999999960 | 0.319998975 | 0.319998907 | 1.72 × 10⁻¹⁵ | 4.41 × 10⁻¹⁴ |

The largest KL probability of non-improving promotion across these cases was 0.003331 at the 16-label boundary law. Its probability of strictly harmful promotion was 0.00000757. At that law, a tree can tie the anchor's true Brier score. Under overshoot at 16 labels every challenger is harmful and KL promoted one with probability 0.000305. The table includes all four metrics for the raw, Hoeffding and KL rules. [All 36 comparisons](results.csv)

At 64 labels the KL certificate recovered 83.16% of the strong improvements, which supports preferring it for this fixed binary menu. At 16 labels even the sharpened rule misses more than four out of five. Power for an evolved tree family, or for other probability moves and loss ranges, needs its own analysis.

## Validity and numerical verification

For a fixed challenger let the independent active-row agreement indicators have probabilities pᵢ and average p̄. At exponent s ≥ 0, the moment-generating function of their sum is the product of (1 − pᵢ + pᵢ exp(s)). By the arithmetic–geometric mean inequality this product is at most (1 − p̄ + p̄ exp(s))ᴹ. Optimizing the exponential tail bound gives exp(−M KL(v ‖ p̄)) for v ≥ p̄. Inversion gives a one-sided lower bound on p̄ with failure probability at most 1/160 for each fixed challenger. Summing over eight challengers gives at most 0.05 family error. This conditions on the design and allows unequal agreement probabilities across rows.

The script brackets each lower root on a grid of width 2⁻⁵⁶ by exponentiating the KL inequality and comparing integers. It checks each winning score interval clears every distinct competing interval. Programs with identical active and agreement counts tie exactly. Independent double-precision inversion of the log equation checks the roots. The probability sums use integer product-binomial weights. Run the following command from the repository root to reproduce the saved results.

```sh
python docs/ensemble-study/anchoring-geometry/certificate/kl/enumerate_kl.py
```

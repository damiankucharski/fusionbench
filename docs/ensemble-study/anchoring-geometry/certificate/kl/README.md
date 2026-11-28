# Bernoulli KL certificate on the fixed binary menu

## Protocol frozen before enumeration

This exploratory follow-up was specified after the Hoeffding certificate failed to promote any tree at 16 labels. Reuse the eight raw programs, anchor 0.5, four probability laws, sample sizes 16, 64 and 256, equal context counts, one-sided family error target 0.05, uniform program weights, four reported metrics and exact binomial enumeration from the [first protocol](../protocol.md).

On a row where a challenger moves from 0.5 to 0.1 or 0.9, the full Brier gain is 0.48 when the label agrees with the direction of the change and −1.12 otherwise. On the other rows the gain is zero. For M active rows among N, let v be the fraction that agree. Take the lower root p ≤ v of M KL(v ‖ p) = log(160), with p = 0 at v = 0 and p = 160^(−1/M) at v = 1. The lower bound on expected full Brier gain is (M/N) × 1.6 × (p − 0.7). Keep the anchor when all bounds are nonpositive. Resolve a positive tie by lexicographic order of the challenger probability pair. Use this rule without adjusting any threshold after enumeration. Compare its results with raw selection and the published Hoeffding certificate.

The prespecified small-sample check is whether the strong-law certificate succeeds on the event that all 16 labels agree with the true challenger, which has probability 0.9¹⁶.
